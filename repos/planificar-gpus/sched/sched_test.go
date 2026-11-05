package sched

import (
	"testing"

	"github.com/fiubaTA050/planificar-gpus/cluster"
	"github.com/fiubaTA050/planificar-gpus/trace"
)

var top = cluster.Default

func load(seed uint64) []trace.Job {
	c := trace.Default(top.GPUs())
	c.Seed = seed
	return trace.Generate(c)
}

// Tres cargas distintas: las afirmaciones de la clase no deberían depender
// de una semilla con suerte.
var seeds = []uint64{21, 1, 2}

func run(t *testing.T, jobs []trace.Job, p Policy) Result {
	t.Helper()
	r := Run(top, jobs, p)
	if !r.Deadlock && r.Completed != r.Jobs {
		t.Fatalf("%s: terminaron %d de %d", p.Name, r.Completed, r.Jobs)
	}
	return r
}

func byName(ps []Policy, name string) Policy {
	for _, p := range ps {
		if p.Name == name {
			return p
		}
	}
	panic(name)
}

// Sin gang, dos trabajos que necesitan todas las GPUs se quedan cada uno con
// la mitad y esperan para siempre: retener y esperar. Con gang terminan los dos.
func TestPodByPodDeadlocks(t *testing.T) {
	tiny := cluster.Topology{Racks: 1, ServersPerRack: 2, GPUsPerServer: 4}
	jobs := []trace.Job{{ID: 0, GPUs: 8, Duration: 3600}, {ID: 1, GPUs: 8, Duration: 3600}}
	if r := Run(tiny, jobs, Policy{PodByPod: true}); !r.Deadlock || r.Completed != 0 {
		t.Errorf("de a un pod: trabado %v, terminaron %d", r.Deadlock, r.Completed)
	}
	if r := Run(tiny, jobs, Policy{}); r.Deadlock || r.Completed != 2 {
		t.Errorf("gang: trabado %v, terminaron %d", r.Deadlock, r.Completed)
	}
}

// En el clúster entero, sin gang las GPUs retenidas esperando a las que
// faltan son un desperdicio enorme: todos esperan más que con FIFO y gang.
func TestPodByPodWastes(t *testing.T) {
	jobs := load(21)
	pods := run(t, jobs, Gang()[0])
	fifo := run(t, jobs, Gang()[1])
	if pods.AvgJCT < 2*fifo.AvgJCT || pods.HeldIdle < 1000 {
		t.Errorf("de a un pod: JCT %.1f h, %.0f GPU·h retenidas; FIFO con gang: %.1f h",
			pods.AvgJCT/3600, pods.HeldIdle, fifo.AvgJCT/3600)
	}
}

// Con FIFO estricto, un trabajo grande al frente bloquea a todos: los de 1
// GPU esperan horas con GPUs libres. Backfill lo arregla, y a cambio los
// grandes son los que esperan: los chicos les pasan por adelante.
func TestBackfillFixesHeadOfLineBlocking(t *testing.T) {
	for _, seed := range seeds {
		jobs := load(seed)
		fifo := run(t, jobs, Gang()[1])
		bf := run(t, jobs, Gang()[2])
		if bf.AvgJCT > fifo.AvgJCT/2 {
			t.Errorf("semilla %d: JCT FIFO %.1f h, backfill %.1f h", seed, fifo.AvgJCT/3600, bf.AvgJCT/3600)
		}
		if fifo.WaitBySize["1"] < 3600 || bf.WaitBySize["1"] > 120 {
			t.Errorf("semilla %d: espera de 1 GPU: FIFO %.0f s, backfill %.0f s", seed, fifo.WaitBySize["1"], bf.WaitBySize["1"])
		}
		if bf.WaitBySize[">8"] < 20*bf.WaitBySize["1"] {
			t.Errorf("semilla %d: con backfill los grandes deberían esperar mucho más: %.0f s vs %.0f s",
				seed, bf.WaitBySize[">8"], bf.WaitBySize["1"])
		}
	}
}

// Exigir localidad hace esperar más a los trabajos grandes, pero como corren
// horas, esperar conviene: repartirlos donde haya lugar los hace más lentos
// y el JCT promedio empeora (Jeon et al., primera recomendación).
func TestLocalityIsWorthWaitingFor(t *testing.T) {
	for _, seed := range seeds {
		jobs := load(seed)
		ps := Locality(600)
		strict := run(t, jobs, byName(ps, "consolidada siempre"))
		any := run(t, jobs, byName(ps, "donde haya lugar"))
		if any.Slowdown < 1.05 {
			t.Errorf("semilla %d: repartidos deberían correr más lento: %.2f×", seed, any.Slowdown)
		}
		if strict.AvgJCT >= any.AvgJCT {
			t.Errorf("semilla %d: JCT consolidada %.2f h, donde haya lugar %.2f h", seed, strict.AvgJCT/3600, any.AvgJCT/3600)
		}
	}
}

// Consolidar siempre hace que los grandes esperen con GPUs libres de sobra
// pero repartidas: espera por fragmentación, como en Philly.
func TestFragmentationDelay(t *testing.T) {
	jobs := load(21)
	ps := Locality(600)
	strict := run(t, jobs, byName(ps, "consolidada siempre"))
	any := run(t, jobs, byName(ps, "donde haya lugar"))
	if strict.FragWaitShare[">8"] < 0.25 || any.FragWaitShare[">8"] > 0.05 {
		t.Errorf("espera por fragmentación de los >8: consolidada %.0f%%, donde haya lugar %.0f%%",
			100*strict.FragWaitShare[">8"], 100*any.FragWaitShare[">8"])
	}
	if any.WaitBySize[">8"] >= strict.WaitBySize[">8"] {
		t.Errorf("relajar la localidad debería acortar la espera de los >8: %.0f vs %.0f s", any.WaitBySize[">8"], strict.WaitBySize[">8"])
	}
}

// Tiresias: consolidar sólo los trabajos sensibles a la localidad. Tienen el
// JCT de consolidar siempre (±2%), con menos espera para los grandes.
func TestConsolidateOnlyWhenItMatters(t *testing.T) {
	for _, seed := range seeds {
		jobs := load(seed)
		ps := Locality(600)
		strict := run(t, jobs, byName(ps, "consolidada siempre"))
		any := run(t, jobs, byName(ps, "donde haya lugar"))
		tir := run(t, jobs, byName(ps, "consolidar sólo si importa"))
		if tir.AvgJCT > 1.02*strict.AvgJCT || tir.AvgJCT >= any.AvgJCT {
			t.Errorf("semilla %d: JCT %.2f h; consolidada %.2f h, donde haya lugar %.2f h", seed, tir.AvgJCT/3600, strict.AvgJCT/3600, any.AvgJCT/3600)
		}
		if tir.WaitBySize[">8"] >= strict.WaitBySize[">8"] {
			t.Errorf("semilla %d: espera de los >8 %.0f s vs consolidada %.0f s", seed, tir.WaitBySize[">8"], strict.WaitBySize[">8"])
		}
	}
}

// 2D-LAS no sabe cuánto dura nada y anda casi como SRTF, que sí lo sabe:
// mucho mejor que FIFO y mejor que FIFO + backfill (Tiresias).
func TestLASWithoutKnowingDurations(t *testing.T) {
	for _, seed := range seeds {
		jobs := load(seed)
		ps := Duration(TiresiasPreempt)
		fifo := run(t, jobs, ps[0])
		bf := run(t, jobs, ps[1])
		las := run(t, jobs, ps[3])
		srtf := run(t, jobs, ps[4])
		if las.AvgJCT > fifo.AvgJCT/2 || las.AvgJCT >= bf.AvgJCT || las.AvgJCT > 1.05*srtf.AvgJCT {
			t.Errorf("semilla %d: JCT LAS %.2f h; FIFO %.2f, backfill %.2f, SRTF %.2f",
				seed, las.AvgJCT/3600, fifo.AvgJCT/3600, bf.AvgJCT/3600, srtf.AvgJCT/3600)
		}
		if las.Preemptions == 0 {
			t.Errorf("semilla %d: LAS necesita desalojar", seed)
		}
	}
}

// Los trabajos grandes nuevos tienen prioridad mientras recibieron poco
// servicio: LAS no los deja morir de hambre como el backfill.
func TestLASDoesNotStarveBigJobs(t *testing.T) {
	jobs := load(21)
	ps := Duration(TiresiasPreempt)
	bf := run(t, jobs, ps[1])
	las := run(t, jobs, ps[3])
	if las.WaitBySize[">8"] > bf.WaitBySize[">8"]/2 {
		t.Errorf("espera de los >8: LAS %.0f s, backfill %.0f s", las.WaitBySize[">8"], bf.WaitBySize[">8"])
	}
}

// Un umbral muy bajo para esta carga manda casi todo a la cola de baja
// prioridad enseguida: muchas más preempciones y peor JCT.
func TestThresholdMatters(t *testing.T) {
	jobs := load(21)
	ps := Duration(TiresiasPreempt)
	low := run(t, jobs, ps[2])
	good := run(t, jobs, ps[3])
	if low.Preemptions < 3*good.Preemptions || low.AvgJCT <= good.AvgJCT {
		t.Errorf("umbral 3.200: %d preempciones, %.2f h; 300.000: %d, %.2f h",
			low.Preemptions, low.AvgJCT/3600, good.Preemptions, good.AvgJCT/3600)
	}
}

// Cada preempción cuesta guardar y recargar un checkpoint: si es caro,
// LAS pierde lo que ganó.
func TestPreemptionCost(t *testing.T) {
	jobs := load(21)
	free := run(t, jobs, Duration(0)[3])
	costly := run(t, jobs, Duration(1800)[3])
	if costly.AvgJCT <= free.AvgJCT {
		t.Errorf("JCT con preempción gratis %.2f h, de 30 min %.2f h", free.AvgJCT/3600, costly.AvgJCT/3600)
	}
}

// El simulador no pierde ni inventa trabajo: con cualquier planificador, el
// tiempo corriendo de cada trabajo es al menos su duración ideal.
func TestConservation(t *testing.T) {
	jobs := load(21)[:300]
	for _, p := range append(append(Gang()[1:], Locality(600)...), Duration(TiresiasPreempt)...) {
		r := run(t, jobs, p)
		if r.Slowdown < 1-1e-9 {
			t.Errorf("%s: corrieron menos de lo que necesitaban (%.3f×)", p.Name, r.Slowdown)
		}
		if r.Util <= 0 || r.Util > 1 || r.Useful > r.Util+1e-9 {
			t.Errorf("%s: uso %.2f, útil %.2f", p.Name, r.Util, r.Useful)
		}
	}
}
