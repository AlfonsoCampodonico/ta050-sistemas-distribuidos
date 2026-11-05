// Package engine es el corazón de un servidor de inferencia: el scheduler que
// decide qué secuencias corren en cada iteración del modelo y el administrador
// del KV cache que decide cuántas entran.
//
// El modelo no existe: cada iteración cuesta un tiempo que sale de un modelo
// de costo (Cost) y produce un token por secuencia. Todo lo demás —admisión,
// batching, memoria, preemption, caché de prefijos, prefill en pedazos— es lo
// que hace un servidor real.
//
// El mismo motor se usa de tres maneras: el simulador avanza el reloj virtual
// lo que dura cada iteración (cmd/sim), el servidor HTTP duerme ese tiempo de
// verdad y manda los tokens por SSE (server), y el paquete cluster junta
// varios motores detrás de un router (Clase 19).
package engine

import (
	"fmt"
	"math"
	"sort"

	"github.com/fiubaTA050/servir-tokens/kvcache"
	"github.com/fiubaTA050/servir-tokens/prefix"
)

// Mode es la política de batching.
type Mode int

const (
	// Static es batching a nivel de request, como FasterTransformer: se arma
	// un lote, se corre hasta que termina la secuencia más larga y recién ahí
	// se arma el siguiente. Las que terminan antes siguen ocupando su lugar.
	Static Mode = iota
	// Continuous es batching a nivel de iteración, como Orca: después de cada
	// iteración salen las secuencias que terminaron y entran las que esperan.
	Continuous
)

func (m Mode) String() string {
	if m == Static {
		return "estático"
	}
	return "continuo"
}

// Reserve dice cuánta memoria pide al admitir un administrador contiguo.
type Reserve int

const (
	// ReserveMax reserva la longitud máxima del modelo: Orca (Max) en el paper de vLLM.
	ReserveMax Reserve = iota
	// ReserveExact reserva justo prompt + salida, como si se conociera el
	// largo de la respuesta de antemano: Orca (Oracle), imposible en la práctica.
	ReserveExact
)

// Cost es un modelo de costo de una iteración, en milisegundos.
//
// En decode, cada iteración lee todos los pesos de la memoria de la GPU una
// vez, sin importar cuántas secuencias haya en el lote: ese es Base, y por eso
// batchear sale casi gratis. Cada secuencia agrega un poco (PerSeq: leer su KV
// cache) y cada token de prefill agrega cómputo (PerPrefillToken). Traer un
// token de KV cache ya calculado desde otra máquina (PerLoadToken) es mucho
// más barato que recalcularlo: es la idea de Mooncake, cambiar cómputo por
// almacenamiento.
type Cost struct {
	Base, PerSeq, PerPrefillToken, PerLoadToken float64
}

// DefaultCost son números ilustrativos del orden de un modelo de 70B.
func DefaultCost() Cost {
	return Cost{Base: 20, PerSeq: 0.25, PerPrefillToken: 0.05, PerLoadToken: 0.005}
}

// Step es lo que dura una iteración.
func (c Cost) Step(prefillTokens, seqs int) float64 {
	return c.Base + c.PerSeq*float64(seqs) + c.PerPrefillToken*float64(prefillTokens)
}

// Config configura el motor.
type Config struct {
	Mode     Mode
	MaxBatch int     // secuencias por iteración, como máximo
	MaxLen   int     // longitud máxima del modelo: prompt + salida
	Reserve  Reserve // sólo importa con un administrador contiguo
	Cost     Cost

	// ChunkPrefill limita cuántos tokens de prefill entran en una iteración
	// (Sarathi-Serve). Un prompt más largo se procesa en varias iteraciones,
	// y los decodes del lote no esperan a que termine. 0 = sin límite.
	ChunkPrefill int

	// Cache es el caché de prefijos en la GPU de este motor; Pool, uno
	// compartido entre varios motores, más lento de leer (el pool de KV
	// cache de Mooncake, en la memoria de las CPUs del clúster). nil = sin
	// caché. PrefixBlock es cuántos tokens tiene cada bloque de Request.Prefix.
	Cache, Pool *prefix.Cache
	PrefixBlock int
}

// Request es un pedido: cuándo llegó, cuántos tokens tiene el prompt y
// cuántos va a generar (el motor no lo sabe de antemano salvo con ReserveExact).
type Request struct {
	ID      int
	Arrival float64 // ms
	Prompt  int
	Output  int

	// Session agrupa los turnos de una misma conversación.
	Session int
	// Prefix son los hashes encadenados de los bloques completos del prompt
	// (ver el paquete prefix). Vacío = el prompt no se puede cachear.
	Prefix []uint64
	// KVReady dice que el KV cache del prompt ya fue calculado en otra máquina
	// y viene transferido: la primera admisión no hace prefill (el lado de
	// decode de un servicio desagregado).
	KVReady bool
}

// Token es un token generado.
type Token struct {
	Request int
	Index   int // 0 = el primero
	Time    float64
	Last    bool
}

// Result describe un pedido terminado.
type Result struct {
	Request
	FirstToken float64 // cuándo el cliente vio el primer token
	Done       float64 // cuándo el cliente tuvo la respuesta completa
	Preempted  int     // cuántas veces lo desalojaron por falta de memoria
	MaxGap     float64 // el mayor tiempo entre dos tokens seguidos, ms
}

type seq struct {
	req        Request
	generated  int
	pending    int // tokens de prefill que faltan calcular
	loading    int // tokens que se traen del pool en la próxima iteración
	chunk      int // tokens de prefill que calcula en esta iteración
	produces   bool
	admitted   bool
	estimate   int // tokens de prefill que el router supuso al encolar
	firstToken float64
	lastToken  float64
	maxGap     float64
	preempted  int
	finished   bool
}

// Engine es el motor de inferencia.
type Engine struct {
	cfg     Config
	kv      kvcache.Allocator
	now     float64
	waiting []*seq // por orden de llegada (FCFS)
	running []*seq // por orden de admisión
	results []Result
	onToken func(Token)

	steps, batchSum, preemptions, maxRunning int
	utilSum                                  float64
	promptTokens, cachedTokens, pooledTokens int
	computedPrefill                          int
	queuedEstimate                           int // suma de seq.estimate de la cola
	gaps                                     []float64
}

// New crea un motor.
func New(cfg Config, kv kvcache.Allocator) *Engine {
	if cfg.MaxBatch <= 0 {
		cfg.MaxBatch = 256
	}
	if cfg.PrefixBlock <= 0 {
		cfg.PrefixBlock = 64
	}
	return &Engine{cfg: cfg, kv: kv}
}

// OnToken registra una función que se llama con cada token generado.
func (e *Engine) OnToken(f func(Token)) { e.onToken = f }

// Now es el reloj del motor, en ms.
func (e *Engine) Now() float64 { return e.now }

// Config devuelve la configuración del motor.
func (e *Engine) Config() Config { return e.cfg }

// Submit encola un pedido. Los pedidos tienen que llegar en orden de Arrival.
func (e *Engine) Submit(r Request) error {
	if r.Prompt+r.Output > e.cfg.MaxLen {
		return fmt.Errorf("el pedido %d tiene %d tokens y el modelo admite %d", r.ID, r.Prompt+r.Output, e.cfg.MaxLen)
	}
	if r.Prompt+r.Output > e.kv.Stats().Capacity {
		return fmt.Errorf("el pedido %d no entra en el KV cache ni aunque esté solo", r.ID)
	}
	s := &seq{req: r, estimate: r.Prompt - e.PeekCached(r)}
	e.queuedEstimate += s.estimate
	e.waiting = append(e.waiting, s)
	return nil
}

// Cancel saca un pedido de la cola o del lote y libera su memoria: es lo que
// tiene que pasar cuando el cliente cierra la conexión.
func (e *Engine) Cancel(id int) bool {
	for i, s := range e.waiting {
		if s.req.ID == id {
			e.queuedEstimate -= s.estimate
			e.waiting = append(e.waiting[:i], e.waiting[i+1:]...)
			return true
		}
	}
	for i, s := range e.running {
		if s.req.ID == id {
			e.running = append(e.running[:i], e.running[i+1:]...)
			e.kv.Free(id)
			return true
		}
	}
	return false
}

// Running y Waiting son la cantidad de secuencias en el lote y en la cola.
func (e *Engine) Running() int { return len(e.running) }
func (e *Engine) Waiting() int { return len(e.waiting) }

// QueuedPrefill estima cuántos tokens de prefill tiene por delante este
// motor: los que faltan de las secuencias del lote más los de los prompts de
// la cola que no estaban en su caché cuando llegaron. Es lo que mira un
// router para estimar cuánto va a esperar un pedido nuevo.
func (e *Engine) QueuedPrefill() int {
	n := e.queuedEstimate
	for _, s := range e.running {
		n += s.pending + s.loading
	}
	return n
}

// PeekCached dice cuántos tokens del prompt de r están en el caché de la GPU
// de este motor, sin tocar el caché.
func (e *Engine) PeekCached(r Request) int {
	if e.cfg.Cache == nil || r.KVReady {
		return 0
	}
	usable := min(len(r.Prefix), (r.Prompt-1)/e.cfg.PrefixBlock)
	return e.cfg.Cache.Peek(r.Prefix[:usable]) * e.cfg.PrefixBlock
}

// KV devuelve el uso del KV cache.
func (e *Engine) KV() kvcache.Stats { return e.kv.Stats() }

// Step corre una iteración del modelo y devuelve cuánto duró. Si no hay nada
// que correr pero hay pedidos que llegan más tarde, adelanta el reloj hasta el
// próximo y devuelve 0. idle es true si no hay nada que hacer.
func (e *Engine) Step() (dt float64, idle bool) {
	e.admit()
	if len(e.running) == 0 {
		if len(e.waiting) == 0 {
			return 0, true
		}
		e.now = max(e.now, e.waiting[0].req.Arrival)
		return 0, false
	}

	// Plan de la iteración, en orden de admisión. Cada secuencia en prefill
	// toma lo que pueda del presupuesto de tokens (todo, si no hay
	// ChunkPrefill). Cada secuencia que va a producir un token necesita lugar
	// en el KV cache para él; si no hay, se desaloja la admitida más
	// recientemente (vLLM, §4.5) hasta que alcance.
	budget := math.MaxInt
	if e.cfg.ChunkPrefill > 0 {
		budget = e.cfg.ChunkPrefill
	}
	for i := 0; i < len(e.running); {
		s := e.running[i]
		if s.finished {
			i++
			continue
		}
		s.chunk = min(s.pending, budget)
		budget -= s.chunk
		s.produces = s.chunk == s.pending
		if !s.produces || e.kv.Append(s.req.ID) {
			i++
			continue
		}
		budget += s.chunk
		e.preemptLast()
	}
	if len(e.running) == 0 {
		return 0, false
	}

	prefill, load, seqs := 0, 0, len(e.running) // en Static, las terminadas siguen ocupando lugar
	for _, s := range e.running {
		prefill += s.chunk
		load += s.loading
	}
	dt = e.cfg.Cost.Step(prefill, seqs) + e.cfg.Cost.PerLoadToken*float64(load)
	e.now += dt
	e.computedPrefill += prefill

	e.steps++
	e.batchSum += seqs
	e.maxRunning = max(e.maxRunning, seqs)
	e.utilSum += e.kv.Stats().Utilization()

	for _, s := range e.running {
		if s.finished {
			continue
		}
		if s.loading > 0 || (s.chunk > 0 && s.chunk == s.pending) {
			// El KV cache del prompt quedó completo: se puede reusar.
			e.remember(s.req.Prefix)
		}
		s.loading = 0
		s.pending -= s.chunk
		s.chunk = 0
		if !s.produces {
			continue
		}
		s.generated++
		if s.generated == 1 {
			s.firstToken = e.now
		} else {
			gap := e.now - s.lastToken
			e.gaps = append(e.gaps, gap)
			s.maxGap = max(s.maxGap, gap)
		}
		s.lastToken = e.now
		s.finished = s.generated == s.req.Output
		if e.onToken != nil {
			e.onToken(Token{Request: s.req.ID, Index: s.generated - 1, Time: e.now, Last: s.finished})
		}
	}

	if e.cfg.Mode == Continuous {
		kept := e.running[:0]
		for _, s := range e.running {
			if s.finished {
				e.kv.Free(s.req.ID)
				e.results = append(e.results, e.result(s, s.firstToken))
			} else {
				kept = append(kept, s)
			}
		}
		e.running = kept
	} else if e.allFinished() {
		// Batching estático: el lote termina entero y recién ahí se devuelven
		// las respuestas, sin streaming.
		for _, s := range e.running {
			e.kv.Free(s.req.ID)
			e.results = append(e.results, e.result(s, e.now))
		}
		e.running = nil
	}
	return dt, false
}

func (e *Engine) result(s *seq, first float64) Result {
	return Result{Request: s.req, FirstToken: first, Done: e.now, Preempted: s.preempted, MaxGap: s.maxGap}
}

func (e *Engine) remember(hashes []uint64) {
	if len(hashes) == 0 {
		return
	}
	if e.cfg.Cache != nil {
		e.cfg.Cache.Insert(hashes)
	}
	if e.cfg.Pool != nil {
		e.cfg.Pool.Insert(hashes)
	}
}

func (e *Engine) admit() {
	if e.cfg.Mode == Static && len(e.running) > 0 {
		return
	}
	for len(e.waiting) > 0 && len(e.running) < e.cfg.MaxBatch {
		s := e.waiting[0]
		if s.req.Arrival > e.now {
			return
		}
		limit := e.cfg.MaxLen
		if e.cfg.Reserve == ReserveExact {
			limit = s.req.Prompt + s.req.Output
		}
		// Una secuencia desalojada vuelve con lo que ya generó: hay que
		// recalcular su KV cache con un prefill de prompt + generados.
		if !e.kv.Admit(s.req.ID, s.req.Prompt+s.generated, limit) {
			return // FCFS: si la primera no entra, no se adelanta ninguna
		}
		e.waiting = e.waiting[1:]
		e.queuedEstimate -= s.estimate
		e.planPrefill(s)
		e.running = append(e.running, s)
	}
}

// planPrefill decide cuánto del prompt hay que calcular al admitir: nada si el
// KV cache vino transferido, y si no, todo menos lo que ya está en el caché de
// la GPU (gratis) o en el pool (se trae, a PerLoadToken por token).
func (e *Engine) planPrefill(s *seq) {
	total := s.req.Prompt + s.generated
	first := !s.admitted
	s.admitted = true
	if first {
		e.promptTokens += s.req.Prompt
	}
	if s.req.KVReady && first {
		s.pending, s.loading = 0, 0
		return
	}
	// Siempre se calcula al menos el último token del prompt: de ahí sale
	// el primer token de la respuesta.
	usable := min(len(s.req.Prefix), (s.req.Prompt-1)/e.cfg.PrefixBlock)
	hashes := s.req.Prefix[:usable]
	local := 0
	if e.cfg.Cache != nil {
		local = e.cfg.Cache.Match(hashes)
	}
	pooled := 0
	if e.cfg.Pool != nil {
		pooled = e.cfg.Pool.Match(hashes[local:])
	}
	cached, loaded := local*e.cfg.PrefixBlock, pooled*e.cfg.PrefixBlock
	s.loading = loaded
	s.pending = total - cached - loaded
	if first {
		e.cachedTokens += cached
		e.pooledTokens += loaded
	}
}

// preemptLast desaloja la secuencia admitida más recientemente: libera todo
// su KV cache y la vuelve a poner primera en la cola (recomputación).
func (e *Engine) preemptLast() {
	s := e.running[len(e.running)-1]
	e.running = e.running[:len(e.running)-1]
	e.kv.Free(s.req.ID)
	s.preempted++
	s.pending, s.loading, s.chunk = 0, 0, 0
	s.estimate = s.req.Prompt + s.generated
	e.queuedEstimate += s.estimate
	e.preemptions++
	e.waiting = append([]*seq{s}, e.waiting...)
}

func (e *Engine) allFinished() bool {
	for _, s := range e.running {
		if !s.finished {
			return false
		}
	}
	return true
}

// Results devuelve los pedidos terminados.
func (e *Engine) Results() []Result { return e.results }

// Stats resume la ejecución.
type Stats struct {
	Steps       int
	AvgBatch    float64 // secuencias por iteración, en promedio
	MaxBatch    int
	AvgKVUtil   float64 // fracción de la memoria asignada que guarda tokens
	Preemptions int

	PromptTokens    int     // tokens de prompt de todos los pedidos
	CachedTokens    int     // de esos, los que estaban en el caché de la GPU
	PooledTokens    int     // y los que se trajeron del pool compartido
	ComputedPrefill int     // tokens de prefill calculados de verdad (con recomputaciones)
	TBTP99          float64 // percentil 99 del tiempo entre tokens, ms
}

func (e *Engine) Stats() Stats {
	st := Stats{Steps: e.steps, MaxBatch: e.maxRunning, Preemptions: e.preemptions,
		PromptTokens: e.promptTokens, CachedTokens: e.cachedTokens, PooledTokens: e.pooledTokens,
		ComputedPrefill: e.computedPrefill, TBTP99: percentile(e.gaps, 0.99)}
	if e.steps > 0 {
		st.AvgBatch = float64(e.batchSum) / float64(e.steps)
		st.AvgKVUtil = e.utilSum / float64(e.steps)
	}
	return st
}

// percentile devuelve el percentil p de xs (no modifica xs).
func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[min(len(s)-1, int(p*float64(len(s))))]
}

// Percentile es percentile, para los demás paquetes.
func Percentile(xs []float64, p float64) float64 { return percentile(xs, p) }
