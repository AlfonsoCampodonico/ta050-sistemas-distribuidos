// Package sched es un simulador de eventos discretos de un clúster de GPUs
// compartido, con los planificadores de la clase:
//
//   - FIFO con gang scheduling: el primero de la cola entra entero o nadie
//     pasa (bloqueo de cabeza de cola).
//   - FIFO + backfill: si el primero no entra, pasan los que entran.
//   - "De a un pod": sin gang; cada trabajo va juntando GPUs sueltas y las
//     retiene hasta tenerlas todas (retener y esperar: puede trabarse).
//   - Localidad: consolidada siempre, donde haya lugar, consolidada hasta
//     esperar un rato y después relajada (Philly), o consolidada sólo para los
//     trabajos que la necesitan (Tiresias).
//   - 2D-LAS (Tiresias): prioridad al que menos servicio recibió, medido en
//     GPUs × tiempo, con dos colas y preempción, sin conocer duraciones.
//   - SRTF: el que menos GPU·tiempo le falta. Necesita conocer las
//     duraciones: es la referencia inalcanzable.
//
// El tiempo es virtual: Run procesa una traza entera en milisegundos.
package sched

import (
	"math"
	"sort"

	"github.com/fiubaTA050/planificar-gpus/cluster"
	"github.com/fiubaTA050/planificar-gpus/trace"
)

// Order es en qué orden se atiende la cola.
type Order int

const (
	FIFO Order = iota // por llegada
	LAS               // 2D-LAS discretizado: menos GPU·tiempo recibido primero (Tiresias)
	SRTF              // menos GPU·tiempo restante primero (necesita la duración)
)

// Place es qué localidad se le exige a cada trabajo.
type Place int

const (
	Anywhere    Place = iota // intenta consolidar; si no puede, donde haya lugar
	Strict                   // consolidada o nada
	Relax                    // consolidada hasta esperar RelaxAfter, después donde haya lugar (Philly)
	Consolidate              // consolidada sólo si el trabajo es sensible (Tiresias)
)

// Policy es un planificador.
type Policy struct {
	Name        string
	Order       Order
	Backfill    bool    // si el primero no entra, probar con los siguientes
	Place       Place   //
	RelaxAfter  float64 // segundos de espera antes de relajar la localidad
	Threshold   float64 // LAS: GPU·segundos para pasar a la cola de baja prioridad
	PreemptCost float64 // segundos que pierde un trabajo cada vez que se lo desaloja (checkpoint y restauración)
	PodByPod    bool    // sin gang: GPUs de a una, retenidas hasta tener todas
}

// Result son las métricas de una corrida.
type Result struct {
	Policy                 string
	Jobs, Completed        int
	AvgJCT, P50JCT, P99JCT float64            // segundos
	AvgWait                float64            // segundos sin correr, entre la llegada y el final
	WaitBySize             map[string]float64 // espera promedio por clase de tamaño
	JCTBySize              map[string]float64
	FragWaitShare          map[string]float64 // de la espera, fracción con GPUs libres suficientes pero sin lugar
	Slowdown               float64            // trabajos de varias GPUs: tiempo corriendo / duración ideal
	Util                   float64            // GPUs asignadas / GPUs del clúster, promedio hasta la última llegada
	Useful                 float64            // lo mismo, pesado por la velocidad: GPUs que avanzan
	Fragmentation          float64            // fracción de las GPUs libres en servidores a medio usar, promedio hasta la última llegada
	Preemptions            int
	Makespan               float64
	Deadlock               bool
	DeadlockAt             float64
	HeldIdle               float64 // GPU·horas retenidas por trabajos que no podían arrancar
}

const eps = 1e-6

type job struct {
	trace.Job
	remaining float64 // segundos de ejecución ideal que faltan
	attained  float64 // GPU·segundos recibidos
	run       float64 // segundos corriendo
	wait      float64
	fragWait  float64
	since     float64 // desde cuándo está esperando
	place     cluster.Placement
	speed     float64
	held      cluster.Placement // PodByPod: GPUs retenidas
	end       float64
	done      bool
	started   float64 // cuándo arrancó por primera vez; +Inf si nunca
}

func (j *job) running() bool { return j.place != nil }

type sim struct {
	p       Policy
	t       cluster.Topology
	c       *cluster.Cluster
	jobs    []*job
	next    int    // próxima llegada
	active  []*job // llegados y sin terminar
	now     float64
	busy    float64 // ∫ GPUs asignadas dt
	useful  float64 // ∫ GPUs·velocidad dt
	frag    float64 // ∫ fragmentación dt, mientras hay GPUs libres
	fragT   float64
	held    float64 // ∫ GPUs retenidas sin correr dt
	window  float64 // las métricas de uso se miden hasta la última llegada: después el clúster se vacía
	preempt int
	dead    bool
}

// Run simula la traza con el planificador p en un clúster vacío.
func Run(t cluster.Topology, jobs []trace.Job, p Policy) Result {
	s := &sim{p: p, t: t, c: cluster.New(t)}
	for _, j := range jobs {
		if j.GPUs > t.GPUs() {
			continue
		}
		s.jobs = append(s.jobs, &job{Job: j, remaining: j.Duration, started: math.Inf(1)})
	}
	sort.SliceStable(s.jobs, func(a, b int) bool { return s.jobs[a].Arrival < s.jobs[b].Arrival })
	if len(s.jobs) > 0 {
		s.window = s.jobs[len(s.jobs)-1].Arrival
	}
	for {
		tn := s.nextEvent()
		if math.IsInf(tn, 1) {
			break
		}
		s.advance(tn)
		s.complete()
		for s.next < len(s.jobs) && s.jobs[s.next].Arrival <= s.now+eps {
			j := s.jobs[s.next]
			j.since = j.Arrival
			s.active = append(s.active, j)
			s.next++
		}
		s.schedule()
		if s.p.PodByPod && s.deadlocked() {
			s.dead = true
			break
		}
	}
	return s.result()
}

func (s *sim) nextEvent() float64 {
	tn := math.Inf(1)
	if s.next < len(s.jobs) {
		tn = s.jobs[s.next].Arrival
	}
	for _, j := range s.active {
		if j.running() {
			tn = math.Min(tn, s.now+math.Max(j.remaining, 0)/j.speed)
			if s.p.Order == LAS && j.attained < s.p.Threshold-eps {
				tn = math.Min(tn, s.now+(s.p.Threshold-j.attained)/float64(j.GPUs))
			}
		} else if s.p.Place == Relax && s.now-j.since < s.p.RelaxAfter-eps {
			tn = math.Min(tn, j.since+s.p.RelaxAfter)
		}
	}
	return tn
}

func (s *sim) advance(t float64) {
	dt := t - s.now
	if dt <= 0 {
		s.now = math.Max(s.now, t)
		return
	}
	free := s.c.Free()
	in := math.Max(0, math.Min(t, s.window)-s.now) // parte de dt dentro de la ventana de medición
	if free > 0 {
		s.frag += in * s.c.Fragmentation()
		s.fragT += in
	}
	for _, j := range s.active {
		if j.running() {
			j.remaining -= dt * j.speed
			j.attained += dt * float64(j.GPUs)
			j.run += dt
			s.busy += in * float64(j.GPUs)
			s.useful += in * float64(j.GPUs) * j.speed
			continue
		}
		j.wait += dt
		if free >= j.GPUs {
			j.fragWait += dt
		}
		s.held += dt * float64(j.held.GPUs())
	}
	s.now = t
}

func (s *sim) complete() {
	kept := s.active[:0]
	for _, j := range s.active {
		if j.running() && j.remaining <= eps {
			s.c.Release(j.place)
			j.place, j.done, j.end = nil, true, s.now
			continue
		}
		kept = append(kept, j)
	}
	s.active = kept
}

// key ordena por prioridad: menor primero.
func (s *sim) key(j *job) (float64, float64) {
	switch s.p.Order {
	case LAS:
		q := 0.0
		if j.attained >= s.p.Threshold-eps {
			q = 1
		}
		// Dentro de una cola, FIFO por la hora en que arrancó por primera vez
		// (Tiresias §3.2.3): ordenar por llegada haría que un trabajo que
		// todavía no arrancó desaloje a uno que ya está corriendo.
		if math.IsInf(j.started, 1) {
			return q, 1e15 + j.Arrival
		}
		return q, j.started
	case SRTF:
		return float64(j.GPUs) * j.remaining, j.Arrival
	}
	return 0, j.Arrival
}

func (s *sim) ordered() []*job {
	o := append([]*job(nil), s.active...)
	sort.SliceStable(o, func(a, b int) bool {
		ka1, ka2 := s.key(o[a])
		kb1, kb2 := s.key(o[b])
		if ka1 != kb1 {
			return ka1 < kb1
		}
		return ka2 < kb2
	})
	return o
}

func (s *sim) locality(j *job) cluster.Locality {
	switch s.p.Place {
	case Strict:
		return cluster.Packed
	case Relax:
		if s.now-j.since < s.p.RelaxAfter-eps {
			return cluster.Packed
		}
	case Consolidate:
		if j.Sensitive {
			return cluster.Packed
		}
	}
	return cluster.Anywhere
}

func (s *sim) schedule() {
	if s.p.PodByPod {
		s.podByPod()
		return
	}
	order := s.ordered()
	for i, j := range order {
		if j.running() {
			continue
		}
		loc := s.locality(j)
		if pl, ok := s.c.Find(j.GPUs, loc); ok {
			s.start(j, pl)
			continue
		}
		if s.p.Order != FIFO && s.preemptFor(j, order[i+1:], loc) {
			continue
		}
		if !s.p.Backfill {
			break
		}
	}
}

func (s *sim) start(j *job, pl cluster.Placement) {
	s.c.Allocate(pl)
	j.place = pl
	j.started = math.Min(j.started, s.now)
	j.speed = cluster.Speed(s.t, pl, j.Sensitive)
}

// preemptFor desaloja trabajos de una prioridad estrictamente menor (otra
// cola en LAS) hasta que j entra, empezando por los que ocupan menos GPUs:
// desalojar un trabajo de 32 GPUs para hacer lugar a uno de 1 es tirar
// trabajo. Si ni desalojándolos a todos entra, no desaloja a nadie.
func (s *sim) preemptFor(j *job, lower []*job, loc cluster.Locality) bool {
	kj, _ := s.key(j)
	var cand []*job
	for _, v := range lower {
		if kv, _ := s.key(v); v.running() && kv > kj {
			cand = append(cand, v)
		}
	}
	// Los más chicos primero; entre iguales, el de menor prioridad.
	sort.SliceStable(cand, func(a, b int) bool { return cand[a].GPUs < cand[b].GPUs })
	var victims []*job
	for _, v := range cand {
		s.c.Release(v.place)
		victims = append(victims, v)
		if pl, ok := s.c.Find(j.GPUs, loc); ok {
			for _, v := range victims {
				v.place = nil
				v.since = s.now
				v.remaining += s.p.PreemptCost
				s.preempt++
			}
			s.start(j, pl)
			return true
		}
	}
	for _, v := range victims {
		s.c.Allocate(v.place)
	}
	return false
}

// podByPod reparte las GPUs libres de a una entre los que esperan, en orden
// de llegada y por turnos, como un orquestador que ubica cada pod por su
// cuenta. Cada trabajo retiene lo que consiguió hasta tener todas.
func (s *sim) podByPod() {
	for {
		progress := false
		for _, j := range s.active {
			if j.running() || j.held.GPUs() == j.GPUs || s.c.Free() == 0 {
				continue
			}
			srv := -1
			for _, sl := range j.held {
				if s.c.FreeOn(sl.Server) > 0 {
					srv = sl.Server
					break
				}
			}
			if srv < 0 {
				for k := 0; k < s.t.Servers(); k++ {
					if s.c.FreeOn(k) > 0 && (srv < 0 || s.c.FreeOn(k) > s.c.FreeOn(srv)) {
						srv = k
					}
				}
			}
			s.c.Allocate(cluster.Placement{{Server: srv, GPUs: 1}})
			j.held = addGPU(j.held, srv)
			progress = true
		}
		if !progress {
			break
		}
	}
	for _, j := range s.active {
		if !j.running() && j.held.GPUs() == j.GPUs {
			j.place, j.held = j.held, nil
			j.started = math.Min(j.started, s.now)
			j.speed = cluster.Speed(s.t, j.place, j.Sensitive)
		}
	}
}

func addGPU(p cluster.Placement, srv int) cluster.Placement {
	for i := range p {
		if p[i].Server == srv {
			p[i].GPUs++
			return p
		}
	}
	return append(p, cluster.Slot{Server: srv, GPUs: 1})
}

// deadlocked: nadie corre, no queda ninguna GPU libre y hay trabajos
// esperando. Cada uno retiene lo que el otro necesita: nadie va a avanzar.
func (s *sim) deadlocked() bool {
	if s.c.Free() > 0 {
		return false
	}
	waiting := false
	for _, j := range s.active {
		if j.running() {
			return false
		}
		waiting = true
	}
	return waiting
}

func (s *sim) result() Result {
	r := Result{Policy: s.p.Name, Jobs: len(s.jobs), Preemptions: s.preempt, Deadlock: s.dead,
		WaitBySize: map[string]float64{}, JCTBySize: map[string]float64{}, FragWaitShare: map[string]float64{}}
	if s.dead {
		r.DeadlockAt = s.now
	}
	var jct []float64
	n := map[string]float64{}
	wait := map[string]float64{}
	first, last := math.Inf(1), 0.0
	slow, slowN := 0.0, 0.0
	for _, j := range s.jobs {
		if !j.done {
			continue
		}
		k := trace.Size(j.GPUs)
		jct = append(jct, j.end-j.Arrival)
		r.AvgJCT += j.end - j.Arrival
		r.AvgWait += j.wait
		n[k]++
		r.WaitBySize[k] += j.wait
		r.JCTBySize[k] += j.end - j.Arrival
		wait[k] += j.wait
		r.FragWaitShare[k] += j.fragWait
		first, last = math.Min(first, j.Arrival), math.Max(last, j.end)
		if j.GPUs > 1 {
			slow += j.run / j.Duration
			slowN++
		}
	}
	r.Completed = len(jct)
	if r.Completed == 0 {
		return r
	}
	r.AvgJCT /= float64(r.Completed)
	r.AvgWait /= float64(r.Completed)
	for k := range n {
		r.WaitBySize[k] /= n[k]
		r.JCTBySize[k] /= n[k]
		if wait[k] > 0 {
			r.FragWaitShare[k] /= wait[k]
		}
	}
	sort.Float64s(jct)
	r.P50JCT, r.P99JCT = trace.Percentile(jct, 0.5), trace.Percentile(jct, 0.99)
	if slowN > 0 {
		r.Slowdown = slow / slowN
	}
	r.Makespan = last - first
	if s.dead {
		r.Makespan = s.now - first
	}
	capacity := float64(s.t.GPUs()) * math.Min(s.now, s.window)
	r.Util, r.Useful = s.busy/capacity, s.useful/capacity
	if s.fragT > 0 {
		r.Fragmentation = s.frag / s.fragT
	}
	r.HeldIdle = s.held / 3600
	return r
}
