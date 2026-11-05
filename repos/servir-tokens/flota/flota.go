// Package flota modela una flota de servidores de GPU que sirve muchos
// modelos a la vez, la mayoría ociosos casi todo el tiempo (Clase 20).
//
// No hay una GPU por modelo: cuando llega un pedido para un modelo que no
// está cargado, hay que traer su checkpoint a una GPU libre. Eso es un
// arranque en frío, y cuánto tarda depende de dónde esté el checkpoint:
//
//	remoto ──red──▶ SSD ──▶ DRAM ──PCIe──▶ GPU
//
// Cada servidor tiene una DRAM y un SSD que funcionan como cachés LRU de
// checkpoints. El scheduler decide en qué servidor arrancar cada modelo
// (Policy), y cuánto tiempo queda cargado un modelo ocioso antes de liberar
// la GPU (KeepAlive). Es el diseño de ServerlessLLM (Fu et al., OSDI 2024):
// cargador multinivel, ubicación consciente de la localidad y migración en
// vivo de inferencias.
//
// Todo corre en tiempo virtual, con eventos.
package flota

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"sort"
)

// Tier es dónde está un checkpoint cuando hay que cargarlo.
type Tier int

const (
	DRAM Tier = iota
	SSD
	Remote
)

func (t Tier) String() string { return [...]string{"DRAM", "SSD", "remoto"}[t] }

// Hardware son los anchos de banda de un servidor, en GB/s. Son del orden
// del testbed de ServerlessLLM (red de 10 Gb/s, NVMe, PCIe 4.0); lo que
// importa es la forma: cada nivel es varias veces más rápido que el anterior.
type Hardware struct {
	Net  float64 // del almacenamiento remoto al servidor
	SSD  float64 // del SSD a la DRAM
	PCIe float64 // de la DRAM a una GPU
}

// DefaultHardware: 10 Gb/s de red, un NVMe de 6 GB/s y PCIe de 25 GB/s.
func DefaultHardware() Hardware { return Hardware{Net: 1.25, SSD: 6, PCIe: 25} }

// Loader es cómo se lee un checkpoint.
type Loader struct {
	Name string
	// Pipelined: los niveles se solapan (mientras un pedazo pasa de SSD a
	// DRAM, el anterior ya va a la GPU), así que manda el más lento. Sin
	// pipeline, cada nivel espera a que el anterior termine el archivo
	// entero.
	Pipelined bool
	// Efficiency es la fracción del ancho de banda local (SSD y PCIe) que se
	// aprovecha. Un cargador que lee tensor por tensor, con mmap y sin
	// memoria fijada, usa una fracción chica (ServerlessLLM, figura 6b).
	Efficiency float64
}

var (
	// Naive lee tensor por tensor y nivel por nivel, como torch.load.
	Naive = Loader{Name: "ingenuo", Pipelined: false, Efficiency: 0.3}
	// Fast es el cargador de ServerlessLLM: formato pensado para leer en
	// bloques grandes, O_DIRECT, memoria fijada, varios hilos y pipeline.
	Fast = Loader{Name: "optimizado", Pipelined: true, Efficiency: 1}
)

// LoadTime es cuántos segundos tarda en llegar a una GPU un checkpoint de
// gb gigabytes que está en el nivel from.
func (h Hardware) LoadTime(l Loader, from Tier, gb float64) float64 {
	var hops []float64
	switch from {
	case Remote:
		hops = append(hops, h.Net)
		fallthrough
	case SSD:
		hops = append(hops, h.SSD*l.Efficiency)
		fallthrough
	case DRAM:
		hops = append(hops, h.PCIe*l.Efficiency)
	}
	if l.Pipelined {
		slowest := math.Inf(1)
		for _, b := range hops {
			slowest = min(slowest, b)
		}
		return gb / slowest
	}
	t := 0.0
	for _, b := range hops {
		t += gb / b
	}
	return t
}

// Policy decide en qué servidor arranca un modelo que no está cargado.
type Policy int

const (
	// Random elige cualquier GPU libre: el scheduler serverless de siempre.
	Random Policy = iota
	// Locality estima el tiempo de arranque en cada servidor con GPU libre
	// —cola de carga más tamaño sobre ancho de banda del nivel donde está el
	// checkpoint— y elige el menor (ServerlessLLM §6.1).
	Locality
	// Migrate es Locality, pero si el checkpoint está en la DRAM de un
	// servidor sin GPU libre, puede mudar una inferencia en curso de ese
	// servidor a otro para hacerle lugar: migración en vivo (§5).
	Migrate
	// Preempt es como Migrate pero en lugar de mudar la inferencia la corta
	// y la vuelve a empezar en otro lado: pierde lo que llevaba (Shepherd).
	Preempt
)

func (p Policy) String() string {
	return [...]string{"al azar", "consciente de la localidad", "localidad + migración", "localidad + desalojo"}[p]
}

// Config describe la flota.
type Config struct {
	Servers   int
	GPUs      int     // por servidor; cada instancia de un modelo ocupa una GPU
	Batch     int     // pedidos que una instancia atiende a la vez (batching continuo)
	DRAMGB    float64 // DRAM de cada servidor para guardar checkpoints
	SSDGB     float64 // SSD de cada servidor para guardar checkpoints
	ModelGB   float64 // tamaño de cada checkpoint
	Replicas  int     // copias de cada checkpoint repartidas en los SSDs al empezar
	Hardware  Hardware
	Loader    Loader
	Policy    Policy
	KeepAlive float64 // segundos que un modelo ocioso queda cargado; math.Inf(1) = hasta que haga falta la GPU
	// Costo de la inferencia, en milisegundos por token.
	PrefillMs   float64
	DecodeMs    float64
	RecomputeMs float64 // recalcular el KV cache de un token ya generado (es un prefill)
	Seed        uint64
}

// DefaultConfig es la flota de las demos: 4 servidores de 4 GPUs, como el
// clúster de ServerlessLLM (§7.1), con checkpoints de 13 GB
// (un modelo de 6,7B en FP16). En la DRAM de cada servidor entran 10
// checkpoints; en el SSD, 32: cada modelo está en el SSD de dos servidores.
func DefaultConfig() Config {
	return Config{Servers: 4, GPUs: 4, Batch: 8, DRAMGB: 130, SSDGB: 416, ModelGB: 13, Replicas: 2,
		Hardware: DefaultHardware(), Loader: Fast, Policy: Migrate, KeepAlive: 30,
		PrefillMs: 0.25, DecodeMs: 25, RecomputeMs: 2.5, Seed: 1}
}

// Summary es lo que mide la flota.
type Summary struct {
	Requests    int
	GPUs        int
	Cold        float64 // fracción de pedidos que esperaron una carga
	Loads       [3]int  // cargas desde cada nivel (DRAM, SSD, remoto)
	DelayP50    float64 // segundos de demora: lo que tardó de más un pedido respecto de correr solo en una GPU con el modelo cargado
	DelayP99    float64
	DelayMean   float64
	Migrations  int
	Preemptions int
	PauseMean   float64 // segundos que estuvo parada, en promedio, cada inferencia mudada o cortada
	Busy        float64 // fracción del tiempo de GPU haciendo inferencia
	Loading     float64 // fracción cargando checkpoints o recalculando KV
	IdleWarm    float64 // fracción con un modelo cargado y sin trabajo: lo que cuesta el keep-alive
	Throughput  float64 // pedidos por segundo
}

// --- estado de la simulación ---

type slotState int

const (
	free slotState = iota
	loading
	busy
	idle
)

type job struct {
	req       Request
	service   float64 // segundos de GPU que necesita
	start     float64 // cuándo empezó a correr
	end       float64 // cuándo termina, si nada cambia
	pause     float64 // segundos parado por migraciones o desalojos
	cold      bool
	preempted bool
	gen       int // invalida eventos viejos
	slot      *slot
}

func (j *job) generated(t float64, c Config) int {
	dec := t - j.start - float64(j.req.Prompt)*c.PrefillMs/1000
	return max(0, min(j.req.Output, int(dec*1000/c.DecodeMs)))
}

type server struct {
	id         int
	slots      []*slot
	dram, ssd  lru
	loaderFree float64 // las cargas de un servidor van de a una (§6.1)
}

// slot es una GPU: libre, cargando un modelo, con pedidos o con el modelo
// cargado y ocioso.
type slot struct {
	srv      *server
	state    slotState
	since    float64
	model    int
	jobs     []*job // los que corren, o los que esperan que termine la carga
	gen      int
	victim   *slot // si es destino de una migración: de dónde vienen las inferencias
	resuming bool  // destino que ya cargó el modelo y está recalculando el KV cache
	reserved *job  // si es origen de una migración: quién usa la GPU cuando se vayan
}

type event struct {
	t    float64
	kind int
	s    *slot
	j    *job
	gen  int
}

const (
	evArrive = iota
	evLoaded
	evDone
	evExpire
	evMigrated
)

type events []event

func (e events) Len() int { return len(e) }
func (e events) Less(i, j int) bool {
	if e[i].t != e[j].t {
		return e[i].t < e[j].t
	}
	return e[i].kind < e[j].kind
}
func (e events) Swap(i, j int) { e[i], e[j] = e[j], e[i] }
func (e *events) Push(x any)   { *e = append(*e, x.(event)) }
func (e *events) Pop() any {
	old := *e
	x := old[len(old)-1]
	*e = old[:len(old)-1]
	return x
}

type sim struct {
	c           Config
	servers     []*server
	q           events
	pending     []*job
	requeued    []*job
	dispatching bool
	rng         *rand.Rand
	time        [4]float64 // tiempo de GPU en cada estado
	now         float64
	s           Summary
	delays      []float64
	pauses      []float64
}

// Run sirve los pedidos con la flota descripta por c.
func Run(c Config, reqs []Request) Summary {
	m := &sim{c: c, rng: rand.New(rand.NewPCG(c.Seed, 23))}
	for i := range c.Servers {
		srv := &server{id: i, dram: lru{cap: c.DRAMGB}, ssd: lru{cap: c.SSDGB}}
		for range c.GPUs {
			srv.slots = append(srv.slots, &slot{srv: srv})
		}
		m.servers = append(m.servers, srv)
	}
	// Ubicación inicial en los SSDs: en ronda, Replicas copias de cada uno,
	// los más populares primero, hasta llenar (ServerlessLLM §7.1).
	models := 0
	for _, r := range reqs {
		models = max(models, r.Model+1)
	}
	for k := range c.Replicas {
		for mod := range models {
			srv := m.servers[(mod+k)%c.Servers]
			if srv.ssd.used+c.ModelGB <= srv.ssd.cap {
				srv.ssd.insert(mod, c.ModelGB)
			}
		}
	}
	for _, r := range reqs {
		service := (float64(r.Prompt)*c.PrefillMs + float64(r.Output)*c.DecodeMs) / 1000
		heap.Push(&m.q, event{t: r.Arrival, kind: evArrive, j: &job{req: r, service: service}})
	}
	for m.q.Len() > 0 {
		e := heap.Pop(&m.q).(event)
		m.now = e.t
		switch e.kind {
		case evArrive:
			m.pending = append(m.pending, e.j)
			m.dispatch()
		case evLoaded:
			if e.s.gen == e.gen {
				m.loaded(e.s)
			}
		case evDone:
			if e.j.gen == e.gen {
				m.finished(e.j)
			}
		case evExpire:
			if e.s.gen == e.gen && e.s.state == idle {
				m.set(e.s, free)
				e.s.gen++
			}
		case evMigrated:
			if e.s.gen == e.gen {
				m.migrated(e.s)
			}
		}
	}
	for _, srv := range m.servers {
		for _, s := range srv.slots {
			m.set(s, s.state) // cierra la contabilidad
		}
	}
	total := m.time[0] + m.time[1] + m.time[2] + m.time[3]
	s := m.s
	s.GPUs = c.Servers * c.GPUs
	if total > 0 {
		s.Busy, s.Loading, s.IdleWarm = m.time[busy]/total, m.time[loading]/total, m.time[idle]/total
	}
	if s.Requests > 0 {
		s.Cold /= float64(s.Requests)
		s.Throughput = float64(s.Requests) / m.now
	}
	s.DelayP50, s.DelayP99, s.DelayMean = percentile(m.delays, 0.5), percentile(m.delays, 0.99), mean(m.delays)
	s.PauseMean = mean(m.pauses)
	return s
}

func (m *sim) set(s *slot, st slotState) {
	m.time[s.state] += m.now - s.since
	s.state, s.since = st, m.now
}

func (m *sim) push(e event) { heap.Push(&m.q, e) }

// tier es dónde tiene el servidor el checkpoint del modelo.
func (m *sim) tier(srv *server, model int) Tier {
	switch {
	case srv.dram.has(model):
		return DRAM
	case srv.ssd.has(model):
		return SSD
	}
	return Remote
}

// startupEstimate es la estimación del scheduler: q + n/b (§6.1).
func (m *sim) startupEstimate(srv *server, model int) float64 {
	return max(0, srv.loaderFree-m.now) + m.c.Hardware.LoadTime(m.c.Loader, m.tier(srv, model), m.c.ModelGB)
}

// dispatch asigna los pedidos pendientes, en orden de llegada: a una
// instancia de su modelo con lugar en el lote si hay, si no arrancando una
// nueva.
func (m *sim) dispatch() {
	if m.dispatching {
		return
	}
	m.dispatching = true
	for again := true; again; {
		pend := m.pending
		m.pending, m.requeued = nil, nil
		var kept []*job
		for _, j := range pend {
			if s := m.instance(j.req.Model); s != nil {
				m.join(s, j)
			} else if !m.start(j) {
				kept = append(kept, j)
			}
		}
		// Los desalojados vuelven adelante de la cola, y puede que ahora
		// haya lugar para ellos.
		again = len(m.requeued) > 0
		m.pending = append(m.requeued, kept...)
	}
	m.dispatching = false
}

// instance busca una GPU que ya tenga el modelo (o lo esté cargando) y
// lugar en su lote. Prefiere las que ya están corriendo.
func (m *sim) instance(model int) *slot {
	var warming *slot
	for _, srv := range m.servers {
		for _, s := range srv.slots {
			if s.model != model || len(s.jobs) >= m.c.Batch || s.reserved != nil || s.victim != nil {
				continue
			}
			switch s.state {
			case busy, idle:
				return s
			case loading:
				if warming == nil {
					warming = s
				}
			}
		}
	}
	return warming
}

func (m *sim) join(s *slot, j *job) {
	if s.state == loading {
		j.cold = true
		s.jobs = append(s.jobs, j)
		return
	}
	m.run(s, j)
}

// available es la GPU que se puede usar en srv para otro modelo: una libre,
// o si no la que tiene un modelo ocioso hace más tiempo (keep-alive es un
// caché: si hace falta la GPU, se desaloja).
func available(srv *server) *slot {
	var best *slot
	for _, s := range srv.slots {
		if s.state == free {
			return s
		}
		if s.state == idle && (best == nil || s.since < best.since) {
			best = s
		}
	}
	return best
}

// start arranca una instancia nueva para j. Devuelve false si no hay GPU.
func (m *sim) start(j *job) bool {
	model := j.req.Model
	var cands []*slot
	for _, srv := range m.servers {
		if s := available(srv); s != nil {
			cands = append(cands, s)
		}
	}
	if m.c.Policy == Random {
		if len(cands) == 0 {
			return false
		}
		m.load(cands[m.rng.IntN(len(cands))], j)
		return true
	}
	var best *slot
	bestT := math.Inf(1)
	for _, s := range cands {
		if t := m.startupEstimate(s.srv, model); t < bestT {
			best, bestT = s, t
		}
	}
	if (m.c.Policy == Migrate || m.c.Policy == Preempt) && !j.preempted {
		if src, dst, t := m.makeRoom(model, len(cands) > 0); src != nil && t < bestT {
			if m.c.Policy == Migrate {
				m.migrate(src, dst, j)
			} else {
				m.preempt(src, j)
			}
			return true
		}
	}
	if best == nil {
		return false
	}
	m.load(best, j)
	return true
}

// makeRoom busca un servidor sin GPU disponible que tenga el modelo en DRAM
// y una GPU cuyas inferencias se puedan mudar a otro servidor (o cortar, si
// la política es Preempt). Devuelve la GPU de origen, la de destino y el
// tiempo estimado hasta arrancar el modelo.
func (m *sim) makeRoom(model int, anyFree bool) (src, dst *slot, est float64) {
	est = math.Inf(1)
	for _, srv := range m.servers {
		if available(srv) != nil || !srv.dram.has(model) {
			continue
		}
		for _, s := range srv.slots {
			if s.state != busy || s.reserved != nil || s.model == model || anyPreempted(s) {
				continue
			}
			var to *slot
			move := 0.0 // desalojar libera la GPU ya
			if m.c.Policy == Migrate {
				if !anyFree {
					continue
				}
				move = math.Inf(1)
				for _, other := range m.servers {
					d := available(other)
					if other == srv || d == nil {
						continue
					}
					if t := m.startupEstimate(other, s.model) + m.resumeTime(s); t < move {
						to, move = d, t
					}
				}
				if to == nil {
					continue
				}
			}
			if t := move + m.startupEstimate(srv, model); t < est {
				src, dst, est = s, to, t
			}
		}
	}
	return src, dst, est
}

func anyPreempted(s *slot) bool {
	for _, j := range s.jobs {
		if j.preempted {
			return true
		}
	}
	return false
}

// resumeTime estima cuánto tarda el destino en recalcular el KV cache de las
// inferencias de s: a·(tokens de entrada + tokens generados) (§6.2).
func (m *sim) resumeTime(s *slot) float64 {
	tokens := 0
	for _, j := range s.jobs {
		tokens += j.req.Prompt + j.generated(m.now, m.c)
	}
	return float64(tokens) * m.c.RecomputeMs / 1000
}

// load trae el checkpoint del modelo de j a la GPU s, y después corre j.
func (m *sim) load(s *slot, j *job) {
	m.loadModel(s, j.req.Model)
	j.cold = true
	s.jobs = []*job{j}
}

func (m *sim) loadModel(s *slot, model int) {
	srv := s.srv
	tier := m.tier(srv, model)
	m.s.Loads[tier]++
	begin := max(m.now, srv.loaderFree)
	done := begin + m.c.Hardware.LoadTime(m.c.Loader, tier, m.c.ModelGB)
	srv.loaderFree = done
	// Lo que pasa por un nivel queda guardado ahí.
	if tier == Remote {
		srv.ssd.insert(model, m.c.ModelGB)
	}
	if tier == SSD {
		srv.ssd.touch(model)
	}
	srv.dram.insert(model, m.c.ModelGB)
	m.set(s, loading)
	s.gen++
	s.model, s.jobs, s.victim, s.resuming = model, nil, nil, false
	m.push(event{t: done, kind: evLoaded, s: s, gen: s.gen})
}

func (m *sim) loaded(s *slot) {
	if s.victim != nil {
		m.resume(s)
		return
	}
	waiting := s.jobs
	s.jobs = nil
	if len(waiting) == 0 {
		m.becomeIdle(s)
		return
	}
	for _, j := range waiting {
		m.run(s, j)
	}
}

func (m *sim) run(s *slot, j *job) {
	if s.state != busy {
		m.set(s, busy)
		s.gen++
	}
	s.jobs = append(s.jobs, j)
	j.slot = s
	j.start = m.now
	j.end = m.now + j.service
	j.gen++
	m.push(event{t: j.end, kind: evDone, j: j, gen: j.gen})
}

func (m *sim) finished(j *job) {
	s := j.slot
	for i, x := range s.jobs {
		if x == j {
			s.jobs = append(s.jobs[:i], s.jobs[i+1:]...)
			break
		}
	}
	m.s.Requests++
	if j.cold {
		m.s.Cold++
	}
	m.delays = append(m.delays, m.now-j.req.Arrival-j.service)
	if len(s.jobs) > 0 {
		m.dispatch() // quedó lugar en el lote
		return
	}
	if s.reserved != nil {
		// Terminaron antes de que la migración se completara: se cancela, y
		// la GPU pasa directo al pedido que la esperaba.
		m.cancelMigration(s)
		return
	}
	m.becomeIdle(s)
}

func (m *sim) becomeIdle(s *slot) {
	m.set(s, idle)
	s.gen++
	s.jobs = nil
	m.dispatch()
	if s.state == idle && !math.IsInf(m.c.KeepAlive, 1) {
		m.push(event{t: m.now + m.c.KeepAlive, kind: evExpire, s: s, gen: s.gen})
	}
}

// --- migración en vivo (§5) ---

// migrate muda las inferencias que corren en src a dst, y cuando src queda
// libre carga ahí el modelo de j, que tiene el checkpoint en DRAM.
func (m *sim) migrate(src, dst *slot, j *job) {
	m.s.Migrations++
	src.reserved = j
	j.cold = true
	m.loadModel(dst, src.model) // paso 1: cargar el modelo en el destino
	dst.victim = src
}

// resume: el destino recalcula el KV cache con los tokens que le manda el
// origen, en rondas. Mientras recalcula, el origen sigue generando; como
// recalcular un token es mucho más rápido que generarlo, cada ronda es más
// corta que la anterior. Cuando la diferencia es de un token por pedido, el
// origen para y le manda el último pedazo: esa ronda final es la única
// pausa.
func (m *sim) resume(dst *slot) {
	src := dst.victim
	total := m.resumeTime(src)
	round := total
	n := float64(len(src.jobs))
	for range 20 {
		// tokens que generó cada pedido en el origen durante la ronda
		gen := round * 1000 / m.c.DecodeMs
		if gen <= 1 {
			break
		}
		round = gen * n * m.c.RecomputeMs / 1000
		total += round
	}
	dst.gen++
	dst.resuming = true
	m.push(event{t: m.now + total, kind: evMigrated, s: dst, gen: dst.gen})
}

func (m *sim) migrated(dst *slot) {
	src := dst.victim
	dst.victim, dst.resuming = nil, false
	if src == nil || len(src.jobs) == 0 || src.reserved == nil {
		m.becomeIdle(dst)
		return
	}
	pause := float64(len(src.jobs)) * m.c.RecomputeMs / 1000 // la última ronda: un token por pedido
	// Las inferencias siguen en dst desde donde estaban.
	m.set(dst, busy)
	dst.gen++
	for _, j := range src.jobs {
		j.pause += pause
		m.pauses = append(m.pauses, pause)
		j.end += pause
		j.slot = dst
		j.gen++
		m.push(event{t: j.end, kind: evDone, j: j, gen: j.gen})
	}
	dst.jobs = src.jobs
	// El origen queda libre para el pedido que lo esperaba.
	next := src.reserved
	src.reserved = nil
	m.load(src, next)
	m.dispatch()
}

func (m *sim) cancelMigration(src *slot) {
	next := src.reserved
	src.reserved = nil
	m.load(src, next)
	for _, srv := range m.servers {
		for _, d := range srv.slots {
			if d.victim == src {
				d.victim = nil
				if d.resuming {
					// ya cargó el modelo: queda ocioso y caliente
					d.resuming = false
					m.becomeIdle(d)
				}
				// si todavía está cargando, loaded lo deja ocioso
			}
		}
	}
}

// preempt corta las inferencias de src, carga ahí el modelo de j y vuelve a
// encolar a las víctimas desde cero: pierden lo que habían generado.
func (m *sim) preempt(src *slot, j *job) {
	m.s.Preemptions++
	for _, v := range src.jobs {
		lost := m.now - v.start
		v.pause += lost
		m.pauses = append(m.pauses, lost)
		v.preempted = true
		v.gen++
		m.requeued = append(m.requeued, v)
	}
	src.jobs = nil
	m.load(src, j)
}

// --- cachés de checkpoints ---

// lru es un caché de checkpoints por tamaño; el último de order es el más
// reciente.
type lru struct {
	cap, used float64
	order     []int
	size      []float64
}

func (c *lru) has(model int) bool {
	for _, x := range c.order {
		if x == model {
			return true
		}
	}
	return false
}

func (c *lru) touch(model int) {
	for i, x := range c.order {
		if x == model {
			sz := c.size[i]
			c.order = append(append(c.order[:i:i], c.order[i+1:]...), model)
			c.size = append(append(c.size[:i:i], c.size[i+1:]...), sz)
			return
		}
	}
}

func (c *lru) insert(model int, gb float64) {
	if c.has(model) {
		c.touch(model)
		return
	}
	if gb > c.cap {
		return
	}
	for c.used+gb > c.cap {
		c.used -= c.size[0]
		c.order, c.size = c.order[1:], c.size[1:]
	}
	c.order = append(c.order, model)
	c.size = append(c.size, gb)
	c.used += gb
}

// --- la alternativa: una GPU por modelo ---

// Dedicated sirve la misma carga con una GPU por modelo, siempre cargado: no
// hay arranques en frío, sólo cola cuando llegan más de Batch pedidos juntos.
func Dedicated(c Config, reqs []Request) Summary {
	free := map[int][]float64{} // cuándo se libera cada lugar del lote de cada modelo
	var delays []float64
	end := 0.0
	for _, r := range reqs {
		service := (float64(r.Prompt)*c.PrefillMs + float64(r.Output)*c.DecodeMs) / 1000
		slots := free[r.Model]
		if slots == nil {
			slots = make([]float64, c.Batch)
			free[r.Model] = slots
		}
		k := 0
		for i := range slots {
			if slots[i] < slots[k] {
				k = i
			}
		}
		begin := max(r.Arrival, slots[k])
		slots[k] = begin + service
		delays = append(delays, begin-r.Arrival)
		end = max(end, begin+service)
	}
	s := Summary{Requests: len(reqs), GPUs: len(free)}
	s.DelayP50, s.DelayP99, s.DelayMean = percentile(delays, 0.5), percentile(delays, 0.99), mean(delays)
	if end > 0 && len(free) > 0 {
		s.Throughput = float64(len(reqs)) / end
	}
	s.Busy = busyFraction(c, reqs, len(free), end)
	s.IdleWarm = 1 - s.Busy
	return s
}

// busyFraction calcula qué fracción del tiempo de las GPUs dedicadas tiene al
// menos un pedido corriendo.
func busyFraction(c Config, reqs []Request, gpus int, end float64) float64 {
	type iv struct{ a, b float64 }
	by := map[int][]iv{}
	free := map[int][]float64{}
	for _, r := range reqs {
		service := (float64(r.Prompt)*c.PrefillMs + float64(r.Output)*c.DecodeMs) / 1000
		slots := free[r.Model]
		if slots == nil {
			slots = make([]float64, c.Batch)
			free[r.Model] = slots
		}
		k := 0
		for i := range slots {
			if slots[i] < slots[k] {
				k = i
			}
		}
		begin := max(r.Arrival, slots[k])
		slots[k] = begin + service
		by[r.Model] = append(by[r.Model], iv{begin, begin + service})
	}
	total := 0.0
	for _, ivs := range by {
		sort.Slice(ivs, func(i, j int) bool { return ivs[i].a < ivs[j].a })
		curA, curB := ivs[0].a, ivs[0].b
		for _, x := range ivs[1:] {
			if x.a > curB {
				total += curB - curA
				curA, curB = x.a, x.b
			} else {
				curB = max(curB, x.b)
			}
		}
		total += curB - curA
	}
	if end == 0 || gpus == 0 {
		return 0
	}
	return total / (end * float64(gpus))
}

// MigrationBytes compara lo que hay que mandar por la red para mudar una
// inferencia: los tokens (4 bytes cada uno) o el KV cache entero.
// kvPerToken es en bytes (por ejemplo 819.200 para un modelo de 13B en FP16:
// 40 capas · 2 · 5120 · 2 bytes).
func MigrationBytes(tokens int, kvPerToken float64) (tokenBytes, kvBytes float64) {
	return 4 * float64(tokens), kvPerToken * float64(tokens)
}
