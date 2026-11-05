package sched

// Los planificadores de las demos, con los parámetros de la clase.

// Gang son los de la primera demo: qué pasa sin gang, con FIFO y con
// backfill. Los dos con gang consolidan sólo los trabajos que lo necesitan,
// como en la tercera demo.
func Gang() []Policy {
	return []Policy{
		{Name: "de a un pod (sin gang)", PodByPod: true},
		{Name: "FIFO, gang", Place: Consolidate},
		{Name: "FIFO + backfill, gang", Backfill: true, Place: Consolidate},
	}
}

// Locality son los de la segunda: cuánto se exige de localidad. Todos con
// FIFO + backfill.
func Locality(relax float64) []Policy {
	return []Policy{
		{Name: "consolidada siempre", Backfill: true, Place: Strict},
		{Name: "donde haya lugar", Backfill: true, Place: Anywhere},
		{Name: "relajar después de esperar", Backfill: true, Place: Relax, RelaxAfter: relax},
		{Name: "consolidar sólo si importa", Backfill: true, Place: Consolidate},
	}
}

// TiresiasThreshold es el umbral entre las dos colas del testbed de
// Tiresias: 3.200 GPU·segundos (Gu et al., NSDI 2019, §5.2).
const TiresiasThreshold = 3200

// TiresiasPreempt es lo que costó cada preempción en el testbed de
// Tiresias: 13.724 segundos en 221 preempciones, unos 62 s cada una
// (Gu et al., §5.2.4).
const TiresiasPreempt = 13724.0 / 221

// Threshold es el umbral entre las dos colas de 2D-LAS para la carga de las
// demos: 300.000 GPU·segundos, unas 80 GPU·horas. Tiresias usó 3.200 en su
// testbed, con trabajos más cortos; el umbral se elige mirando la carga.
const Threshold = 300000

// Duration son los de la tercera: sin conocer las duraciones. Todos con la
// misma ubicación (consolidar sólo si importa).
func Duration(preempt float64) []Policy {
	return []Policy{
		{Name: "FIFO", Place: Consolidate},
		{Name: "FIFO + backfill", Backfill: true, Place: Consolidate},
		{Name: "2D-LAS, umbral 3.200 GPU·s", Order: LAS, Backfill: true, Place: Consolidate, Threshold: TiresiasThreshold, PreemptCost: preempt},
		{Name: "2D-LAS, umbral 300.000 GPU·s", Order: LAS, Backfill: true, Place: Consolidate, Threshold: Threshold, PreemptCost: preempt},
		{Name: "SRTF (conoce las duraciones)", Order: SRTF, Backfill: true, Place: Consolidate, PreemptCost: preempt},
	}
}
