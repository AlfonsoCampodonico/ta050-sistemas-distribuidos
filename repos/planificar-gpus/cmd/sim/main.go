// sim compara planificadores de un clúster de GPUs compartido con la misma
// carga, en tiempo virtual:
//
//	go run ./cmd/sim carga        # la forma de la carga: muchos chicos, pocos enormes
//	go run ./cmd/sim gang         # sin gang, FIFO con gang, y backfill
//	go run ./cmd/sim localidad    # consolidar, repartir o relajar después de esperar
//	go run ./cmd/sim duraciones   # FIFO, 2D-LAS (Tiresias) y SRTF, con el costo de preempción
//	go run ./cmd/sim traba        # dos trabajos que se traban sin gang, paso a paso
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/fiubaTA050/planificar-gpus/cluster"
	"github.com/fiubaTA050/planificar-gpus/sched"
	"github.com/fiubaTA050/planificar-gpus/trace"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	args := os.Args[2:]
	switch os.Args[1] {
	case "carga":
		carga(args)
	case "gang":
		gang(args)
	case "localidad":
		localidad(args)
	case "duraciones":
		duraciones(args)
	case "traba":
		traba()
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "uso: sim {carga|gang|localidad|duraciones|traba} [flags]   (-h para ver los flags)")
	os.Exit(2)
}

// workload agrega los flags de la carga y devuelve cómo armarla.
func workload(fs *flag.FlagSet) func() (cluster.Topology, []trace.Job) {
	t := cluster.Default
	c := trace.Default(t.GPUs())
	jobs := fs.Int("trabajos", c.Jobs, "trabajos")
	load := fs.Float64("carga", c.Load, "GPU·tiempo pedido sobre el disponible")
	seed := fs.Uint64("semilla", c.Seed, "semilla de la carga")
	racks := fs.Int("racks", t.Racks, "racks de 8 servidores de 8 GPUs")
	return func() (cluster.Topology, []trace.Job) {
		t.Racks = *racks
		c.Jobs, c.Load, c.Seed, c.TotalGPUs = *jobs, *load, *seed, t.GPUs()
		js := trace.Generate(c)
		fmt.Printf("%d trabajos en %d GPUs (%d racks × %d servidores × %d GPUs), carga %.0f%%.\n\n",
			len(js), t.GPUs(), t.Racks, t.ServersPerRack, t.GPUsPerServer, 100*c.Load)
		return t, js
	}
}

func dur(s float64) string {
	switch {
	case s >= 86400*2:
		return fmt.Sprintf("%.1f d", s/86400)
	case s >= 3600:
		return fmt.Sprintf("%.1f h", s/3600)
	case s >= 60:
		return fmt.Sprintf("%.0f min", s/60)
	}
	return fmt.Sprintf("%.0f s", s)
}

// hrs muestra un tiempo en horas con dos decimales, para comparar promedios.
func hrs(s float64) string { return fmt.Sprintf("%.2f h", s/3600) }

func pct(x float64) string { return fmt.Sprintf("%.0f%%", 100*x) }

func carga(args []string) {
	fs := flag.NewFlagSet("carga", flag.ExitOnError)
	setup := workload(fs)
	fs.Parse(args)
	_, jobs := setup()
	s := trace.Describe(jobs)
	fmt.Printf("%-10s %12s %16s\n", "GPUs", "trabajos", "GPU·tiempo")
	for _, k := range trace.Sizes {
		fmt.Printf("%-10s %12s %16s\n", k, pct(s.ShareJobs[k]), pct(s.ShareGPUTime[k]))
	}
	fmt.Printf("\nDuración: mediana %s, p90 %s, p99 %s.\n", dur(s.MedianMin*60), dur(s.P90H*3600), dur(s.P99H*3600))
	fmt.Printf("El 1%% de los trabajos que más GPU·tiempo piden se lleva el %s del total.\n", pct(s.TopShare))
	fmt.Printf("Las llegadas abarcan %s.\n", dur(s.Span))
	fmt.Println("\nLa forma es la de Philly: más del 82% de 1 GPU (Weng et al., NSDI 2022), mediana de 26 min,")
	fmt.Println("cola de días (Jeon et al., ATC 2019). Los porcentajes de esta carga son ilustrativos.")
}

func header(cols ...string) {
	fmt.Printf("%-30s", "")
	for _, c := range cols {
		fmt.Printf(" %11s", c)
	}
	fmt.Println()
}

func line(name string, cols ...string) {
	fmt.Printf("%-30s", name)
	for _, c := range cols {
		fmt.Printf(" %11s", c)
	}
	fmt.Println()
}

func gang(args []string) {
	fs := flag.NewFlagSet("gang", flag.ExitOnError)
	setup := workload(fs)
	fs.Parse(args)
	t, jobs := setup()
	header("terminados", "JCT prom.", "espera 1", "espera 2-8", "espera >8", "uso GPUs", "retenidas")
	for _, p := range sched.Gang() {
		r := sched.Run(t, jobs, p)
		held := "-"
		if p.PodByPod {
			held = fmt.Sprintf("%.0f k GPU·h", r.HeldIdle/1000)
		}
		if r.Deadlock {
			line(p.Name, fmt.Sprintf("%d", r.Completed), "trabado", "", "", "", pct(r.Util), held)
			fmt.Printf("%-30s se trabó a las %s: ninguna GPU libre y nadie corriendo\n", "", dur(r.DeadlockAt))
			continue
		}
		line(p.Name, fmt.Sprintf("%d", r.Completed), hrs(r.AvgJCT), dur(r.WaitBySize["1"]), dur(r.WaitBySize["2-8"]),
			dur(r.WaitBySize[">8"]), pct(r.Util), held)
	}
	fmt.Println("\nEspera = tiempo sin correr entre la llegada y el final, promedio por cantidad de GPUs.")
	fmt.Println("Uso = GPUs asignadas a trabajos que corren, sobre el total. Retenidas = GPU·horas que un")
	fmt.Println("trabajo tuvo tomadas sin poder arrancar porque le faltaban las demás.")
	fmt.Println("Con FIFO, un trabajo grande al frente de la cola deja GPUs libres sin usar hasta que entra entero.")
}

func localidad(args []string) {
	fs := flag.NewFlagSet("localidad", flag.ExitOnError)
	relax := fs.Float64("relajar", 10, "minutos de espera antes de relajar la localidad")
	setup := workload(fs)
	fs.Parse(args)
	t, jobs := setup()
	header("JCT prom.", "JCT >8", "espera 2-8", "espera >8", "frag. >8", "más lento", "uso útil")
	for _, p := range sched.Locality(*relax * 60) {
		r := sched.Run(t, jobs, p)
		line(p.Name, hrs(r.AvgJCT), hrs(r.JCTBySize[">8"]), dur(r.WaitBySize["2-8"]), dur(r.WaitBySize[">8"]),
			pct(r.FragWaitShare[">8"]), fmt.Sprintf("%.2f×", r.Slowdown), pct(r.Useful))
	}
	fmt.Println("\nfrag. >8 = de la espera de los trabajos de más de 8 GPUs, cuánta fue con GPUs libres de sobra")
	fmt.Println("pero repartidas: espera por fragmentación (Jeon et al., tabla 2). Más lento = tiempo corriendo")
	fmt.Println("sobre el ideal, trabajos de varias GPUs. Uso útil = GPUs asignadas pesadas por su velocidad.")
	fmt.Println("La mitad de los trabajos de varias GPUs es sensible a la localidad (corre al 60% si se reparte).")
}

func duraciones(args []string) {
	fs := flag.NewFlagSet("duraciones", flag.ExitOnError)
	cost := fs.Float64("preempcion", sched.TiresiasPreempt, "segundos que cuesta cada preempción")
	setup := workload(fs)
	fs.Parse(args)
	t, jobs := setup()
	header("JCT prom.", "JCT p50", "espera 1", "espera 2-8", "espera >8", "preempc.")
	run := func(p sched.Policy) {
		r := sched.Run(t, jobs, p)
		line(p.Name, hrs(r.AvgJCT), dur(r.P50JCT), dur(r.WaitBySize["1"]), dur(r.WaitBySize["2-8"]), dur(r.WaitBySize[">8"]),
			fmt.Sprintf("%d", r.Preemptions))
	}
	for _, p := range sched.Duration(*cost) {
		run(p)
	}
	fmt.Println()
	for _, c := range []float64{0, *cost, 600, 1800} {
		p := sched.Duration(c)[3]
		p.Name = fmt.Sprintf("2D-LAS, preempción de %s", dur(c))
		run(p)
	}
	fmt.Printf("\nPreempción de %.0f s: lo que costó cada una en el testbed de Tiresias (13.724 s en 221).\n", *cost)
	fmt.Println("SRTF ordena por GPU·tiempo restante: necesita saber cuánto dura cada trabajo, y nadie lo sabe.")
}

// traba muestra el caso mínimo: 8 GPUs en 2 servidores y dos trabajos de 8.
func traba() {
	t := cluster.Topology{Racks: 1, ServersPerRack: 2, GPUsPerServer: 4}
	jobs := []trace.Job{{ID: 0, Arrival: 0, GPUs: 8, Duration: 3600}, {ID: 1, Arrival: 0, GPUs: 8, Duration: 3600}}
	fmt.Println("8 GPUs (2 servidores de 4) y dos trabajos de 8 GPUs que llegan juntos.")
	fmt.Println()
	for _, p := range []sched.Policy{{Name: "de a un pod (sin gang)", PodByPod: true}, {Name: "gang"}} {
		r := sched.Run(t, jobs, p)
		if r.Deadlock {
			fmt.Printf("%-24s trabado: cada uno retiene 4 GPUs y espera las 4 del otro. Terminaron %d de 2.\n", p.Name, r.Completed)
			continue
		}
		fmt.Printf("%-24s terminaron %d de 2; el último a las %s.\n", p.Name, r.Completed, dur(r.Makespan))
	}
	fmt.Println("\nLas cuatro condiciones de Coffman: exclusión mutua, retener y esperar, sin desalojo y espera")
	fmt.Println("circular. Gang scheduling rompe la segunda: se piden todas las GPUs juntas o ninguna.")
}
