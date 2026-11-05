// Package cluster junta varios motores de inferencia detrás de un router, que
// es como se sirve un modelo en producción (Clase 19).
//
// Hay dos formas de armar el clúster:
//
//   - Colocado: cada réplica hace prefill y decode de sus pedidos, y el
//     router elige la réplica de cada pedido (Router).
//   - Desagregado (DistServe, Mooncake): unas réplicas sólo hacen prefill,
//     otras sólo decode. El KV cache del prompt viaja de una a otra por la
//     red, y cada grupo se dimensiona por separado.
//
// Todo corre en tiempo virtual: cada motor tiene su reloj y el simulador
// avanza siempre el que va más atrasado, así que las decisiones del router
// ven el estado del clúster en el momento en que llega cada pedido.
package cluster

import (
	"fmt"
	"math"
	"sort"

	"github.com/fiubaTA050/servir-tokens/engine"
	"github.com/fiubaTA050/servir-tokens/kvcache"
	"github.com/fiubaTA050/servir-tokens/prefix"
)

// Replica es un motor con su caché de prefijos.
type Replica struct {
	E     *engine.Engine
	Cache *prefix.Cache // nil si no hay caché
	block int
}

// CachedTokens cuántos tokens del prompt de r ya están en el caché de la réplica.
func (rep *Replica) CachedTokens(r engine.Request) int { return rep.E.PeekCached(r) }

// Router elige la réplica de cada pedido.
type Router interface {
	Name() string
	Pick(r engine.Request, reps []*Replica) int
}

// RoundRobin reparte en ronda, sin mirar nada.
type RoundRobin struct{ next int }

func (*RoundRobin) Name() string { return "ronda" }
func (rr *RoundRobin) Pick(_ engine.Request, reps []*Replica) int {
	rr.next = (rr.next + 1) % len(reps)
	return rr.next
}

// LeastLoaded elige la réplica con menos trabajo pendiente.
type LeastLoaded struct{}

func (LeastLoaded) Name() string { return "menor carga" }
func (LeastLoaded) Pick(_ engine.Request, reps []*Replica) int {
	return argmin(reps, func(rep *Replica) float64 { return load(rep) })
}

// Sticky manda todos los turnos de una conversación a la misma réplica, por
// hash de la sesión: aprovecha el caché sin mirar la carga.
type Sticky struct{}

func (Sticky) Name() string { return "por sesión (sticky)" }
func (Sticky) Pick(r engine.Request, reps []*Replica) int {
	return int(prefix.Hash(0, uint64(r.Session)) % uint64(len(reps)))
}

// PrefixAware estima el tiempo al primer token en cada réplica —la espera por
// lo que tiene encolado más el prefill de lo que no tiene en caché— y elige
// la menor. Es el algoritmo del Conductor de Mooncake (§6.1, sin el pool).
type PrefixAware struct{ Cost engine.Cost }

func (PrefixAware) Name() string { return "consciente del prefijo" }
func (p PrefixAware) Pick(r engine.Request, reps []*Replica) int {
	return argmin(reps, func(rep *Replica) float64 {
		uncached := r.Prompt - rep.CachedTokens(r)
		return p.Cost.PerPrefillToken*float64(rep.E.QueuedPrefill()+uncached) +
			p.Cost.PerSeq*float64(rep.E.Running())
	})
}

func load(rep *Replica) float64 {
	return float64(rep.E.QueuedPrefill()) + 50*float64(rep.E.Running()+rep.E.Waiting())
}

func argmin(reps []*Replica, f func(*Replica) float64) int {
	best, bestV := 0, math.Inf(1)
	for i, rep := range reps {
		if v := f(rep); v < bestV {
			best, bestV = i, v
		}
	}
	return best
}

// Config describe un clúster.
type Config struct {
	Engine      engine.Config
	KVTokens    int // slots de KV cache por réplica
	Block       int // tokens por bloque de KV cache
	CacheBlocks int // capacidad del caché de prefijos por réplica, en bloques de Engine.PrefixBlock; 0 = sin caché
	PoolBlocks  int // capacidad de un pool compartido (Mooncake); 0 = sin pool
}

func (c Config) replica(pool *prefix.Cache) *Replica {
	ec := c.Engine
	rep := &Replica{block: ec.PrefixBlock}
	if c.CacheBlocks > 0 {
		rep.Cache = prefix.New(c.CacheBlocks)
		ec.Cache = rep.Cache
	}
	ec.Pool = pool
	rep.E = engine.New(ec, kvcache.NewPaged(c.KVTokens, c.Block))
	return rep
}

// SLO son los límites de latencia de un pedido (DistServe, Mooncake).
type SLO struct {
	TTFT   float64 // ms hasta el primer token
	MaxGap float64 // ms entre dos tokens seguidos, como máximo
}

// Summary es lo que mide un clúster.
type Summary struct {
	Requests   int
	Throughput float64 // tokens generados por segundo
	TTFTP50    float64
	TTFTP99    float64
	TBTP99     float64 // percentil 99 del tiempo entre tokens
	HitRate    float64 // fracción de los tokens de prompt que no hubo que calcular
	Computed   int     // tokens de prefill calculados de verdad
	Goodput    float64 // pedidos por segundo que cumplieron el SLO
	SLOMet     float64 // fracción de pedidos que cumplieron el SLO
	Imbalance  float64 // tokens calculados por la réplica más cargada / promedio
}

// Run corre los pedidos en n réplicas colocadas detrás del router.
func Run(c Config, n int, router Router, reqs []engine.Request, slo SLO) (Summary, error) {
	var pool *prefix.Cache
	if c.PoolBlocks > 0 {
		pool = prefix.New(c.PoolBlocks)
	}
	reps := make([]*Replica, n)
	for i := range reps {
		reps[i] = c.replica(pool)
	}
	sim := newSim(reps)
	err := sim.run(reqs, func(r engine.Request) error {
		return reps[router.Pick(r, reps)].E.Submit(r)
	}, nil)
	if err != nil {
		return Summary{}, err
	}
	var results []engine.Result
	for _, rep := range reps {
		results = append(results, rep.E.Results()...)
	}
	return summarize(results, reps, slo, sim.gaps(reqs)), nil
}

// RunDisaggregated corre los pedidos con p réplicas de prefill y d de decode.
// El KV cache de cada prompt tarda transferPerToken ms por token en viajar
// de una a otra (RDMA entre máquinas). Las de prefill se eligen con
// PrefixAware; las de decode, la de menos secuencias.
func RunDisaggregated(c Config, p, d int, transferPerToken float64, reqs []engine.Request, slo SLO) (Summary, error) {
	var pool *prefix.Cache
	if c.PoolBlocks > 0 {
		pool = prefix.New(c.PoolBlocks)
	}
	pre := make([]*Replica, p)
	for i := range pre {
		pre[i] = c.replica(pool)
	}
	dc := c
	dc.CacheBlocks, dc.PoolBlocks = 0, 0
	dec := make([]*Replica, d)
	for i := range dec {
		dec[i] = dc.replica(nil)
	}
	byID := make(map[int]engine.Request, len(reqs))
	for _, r := range reqs {
		byID[r.ID] = r
	}
	router := PrefixAware{Cost: c.Engine.Cost}
	sim := newSim(append(append([]*Replica{}, pre...), dec...))

	// Cada prefill produce el primer token. Cuando termina, el KV cache viaja
	// y el pedido entra al decode con el resto de la respuesta.
	seen := make([]int, p)
	handoffs := func() []engine.Request {
		var out []engine.Request
		for i, rep := range pre {
			rs := rep.E.Results()
			for _, res := range rs[seen[i]:] {
				orig := byID[res.ID]
				if orig.Output > 1 {
					next := orig
					next.Arrival = res.Done + transferPerToken*float64(orig.Prompt)
					next.Prompt, next.Output = orig.Prompt+1, orig.Output-1
					next.KVReady, next.Prefix = true, nil
					out = append(out, next)
				}
			}
			seen[i] = len(rs)
		}
		return out
	}
	toDecode := func(r engine.Request) error {
		i := argmin(dec, func(rep *Replica) float64 { return float64(rep.E.Running() + rep.E.Waiting()) })
		return dec[i].E.Submit(r)
	}
	err := sim.run(reqs, func(r engine.Request) error {
		first := r
		first.Output = 1
		return pre[router.Pick(r, pre)].E.Submit(first)
	}, &relay{collect: handoffs, submit: toDecode})
	if err != nil {
		return Summary{}, err
	}

	// Junta el prefill y el decode de cada pedido en un solo resultado.
	results := make(map[int]engine.Result, len(reqs))
	for _, rep := range pre {
		for _, res := range rep.E.Results() {
			res.Request = byID[res.ID]
			results[res.ID] = res
		}
	}
	for _, rep := range dec {
		for _, res := range rep.E.Results() {
			full := results[res.ID]
			full.MaxGap = max(full.MaxGap, res.MaxGap, res.FirstToken-full.FirstToken)
			full.Done = res.Done
			full.Preempted += res.Preempted
			results[res.ID] = full
		}
	}
	all := make([]engine.Result, 0, len(results))
	for _, res := range results {
		all = append(all, res)
	}
	return summarize(all, pre, slo, sim.gaps(reqs)), nil
}

// relay lleva pedidos de un grupo de réplicas a otro, con demora.
type relay struct {
	collect func() []engine.Request // lo que salió de las réplicas desde la última vez
	submit  func(engine.Request) error
	pending []engine.Request // ordenado por Arrival
}

type sim struct {
	reps   []*Replica
	tokens map[int][]float64 // tiempos de los tokens de cada pedido, de todas las réplicas
}

func newSim(reps []*Replica) *sim {
	s := &sim{reps: reps, tokens: map[int][]float64{}}
	for _, rep := range reps {
		rep.E.OnToken(func(t engine.Token) { s.tokens[t.Request] = append(s.tokens[t.Request], t.Time) })
	}
	return s
}

// stepOne avanza la réplica ocupada de reloj más atrasado, si está antes de
// until. Devuelve false si no había nada que avanzar.
func (s *sim) stepOne(until float64) bool {
	best := -1
	for i, rep := range s.reps {
		if rep.E.Running()+rep.E.Waiting() == 0 {
			continue
		}
		if best < 0 || rep.E.Now() < s.reps[best].E.Now() {
			best = i
		}
	}
	if best < 0 || s.reps[best].E.Now() >= until {
		return false
	}
	s.reps[best].E.Step()
	return true
}

// run es el loop de eventos: en cada vuelta hace lo que pase primero entre
// la llegada del próximo pedido, la llegada de un KV cache transferido y la
// iteración de la réplica más atrasada.
func (s *sim) run(reqs []engine.Request, arrive func(engine.Request) error, rl *relay) error {
	i := 0
	for {
		tArr, tRelay := math.Inf(1), math.Inf(1)
		if i < len(reqs) {
			tArr = reqs[i].Arrival
		}
		if rl != nil && len(rl.pending) > 0 {
			tRelay = rl.pending[0].Arrival
		}
		tStep := s.minClock()
		switch {
		case math.IsInf(tArr, 1) && math.IsInf(tRelay, 1) && math.IsInf(tStep, 1):
			return nil
		case tRelay <= tArr && tRelay <= tStep:
			r := rl.pending[0]
			rl.pending = rl.pending[1:]
			if err := rl.submit(r); err != nil {
				return err
			}
		case tArr <= tStep:
			if err := arrive(reqs[i]); err != nil {
				return err
			}
			i++
		default:
			s.stepOne(math.Inf(1))
			if rl != nil {
				if out := rl.collect(); len(out) > 0 {
					rl.pending = append(rl.pending, out...)
					sort.SliceStable(rl.pending, func(a, b int) bool { return rl.pending[a].Arrival < rl.pending[b].Arrival })
				}
			}
		}
	}
}

// minClock es el reloj de la réplica ocupada más atrasada (o +Inf si no hay).
func (s *sim) minClock() float64 {
	m := math.Inf(1)
	for _, rep := range s.reps {
		if rep.E.Running()+rep.E.Waiting() > 0 {
			m = min(m, rep.E.Now())
		}
	}
	return m
}

func (s *sim) gaps(reqs []engine.Request) []float64 {
	var gaps []float64
	for _, r := range reqs {
		ts := s.tokens[r.ID]
		for i := 1; i < len(ts); i++ {
			gaps = append(gaps, ts[i]-ts[i-1])
		}
	}
	return gaps
}

func throughput(rs []engine.Result) float64 {
	tokens, first, last := 0, math.Inf(1), 0.0
	for _, r := range rs {
		tokens += r.Output
		first = min(first, r.Arrival)
		last = max(last, r.Done)
	}
	return float64(tokens) / ((last - first) / 1000)
}

func summarize(rs []engine.Result, reps []*Replica, slo SLO, gaps []float64) Summary {
	s := Summary{Requests: len(rs)}
	if len(rs) == 0 {
		return s
	}
	ttft := make([]float64, len(rs))
	met := 0
	first, last := math.Inf(1), 0.0
	for i, r := range rs {
		ttft[i] = r.FirstToken - r.Arrival
		if ttft[i] <= slo.TTFT && r.MaxGap <= slo.MaxGap {
			met++
		}
		first = min(first, r.Arrival)
		last = max(last, r.Done)
	}
	s.Throughput = throughput(rs)
	s.TTFTP50 = engine.Percentile(ttft, 0.5)
	s.TTFTP99 = engine.Percentile(ttft, 0.99)
	s.TBTP99 = engine.Percentile(gaps, 0.99)
	s.SLOMet = float64(met) / float64(len(rs))
	s.Goodput = float64(met) / ((last - first) / 1000)

	prompt, reused, maxC, sumC := 0, 0, 0, 0
	for _, rep := range reps {
		st := rep.E.Stats()
		prompt += st.PromptTokens
		reused += st.CachedTokens + st.PooledTokens
		s.Computed += st.ComputedPrefill
		maxC = max(maxC, st.ComputedPrefill)
		sumC += st.ComputedPrefill
	}
	if prompt > 0 {
		s.HitRate = float64(reused) / float64(prompt)
	}
	if sumC > 0 {
		s.Imbalance = float64(maxC) / (float64(sumC) / float64(len(reps)))
	}
	return s
}

// String resume en una línea.
func (s Summary) String() string {
	return fmt.Sprintf("%d pedidos, %.0f tok/s, TTFT p50 %.0f ms p99 %.0f ms, TBT p99 %.0f ms, caché %.0f%%, SLO %.0f%%",
		s.Requests, s.Throughput, s.TTFTP50, s.TTFTP99, s.TBTP99, 100*s.HitRate, 100*s.SLOMet)
}
