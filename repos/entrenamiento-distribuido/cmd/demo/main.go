// demo es la herramienta que se usa en clase. Tiene tres subcomandos:
//
//	go run ./cmd/demo allreduce   # central vs ring: cuánto envía cada GPU
//	go run ./cmd/demo train       # paralelismo de datos, stragglers y fallas
//	go run ./cmd/demo pipeline    # burbuja de GPipe vs 1F1B
//
// Y para la Clase 17, entrenamiento a escala de frontera:
//
//	go run ./cmd/demo 4d          # dónde cae cada dimensión de paralelismo
//	go run ./cmd/demo moe         # cuántos nodos toca cada token con expertos
//	go run ./cmd/demo fallas      # fallas, checkpoints y tiempo útil a escala
//	go run ./cmd/demo diloco      # sincronizar cada H pasos en lugar de cada paso
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/fiubaTA050/entrenamiento-distribuido/collective"
	"github.com/fiubaTA050/entrenamiento-distribuido/pipeline"
	"github.com/fiubaTA050/entrenamiento-distribuido/train"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	args := os.Args[2:]
	switch os.Args[1] {
	case "allreduce":
		allreduce(args)
	case "train":
		trainCmd(args)
	case "pipeline":
		pipelineCmd(args)
	case "4d":
		layoutCmd(args)
	case "moe":
		moeCmd(args)
	case "fallas":
		fallasCmd(args)
	case "diloco":
		dilocoCmd(args)
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "uso: demo {allreduce|train|pipeline|4d|moe|fallas|diloco} [flags]   (-h para ver los flags)")
	os.Exit(2)
}

func allreduce(args []string) {
	fs := flag.NewFlagSet("allreduce", flag.ExitOnError)
	size := fs.Int("size", 100_000, "floats por gradiente (S); el resultado en múltiplos de S no depende de este valor")
	fs.Parse(args)

	fmt.Printf("All-reduce de un gradiente de %d floats.\n", *size)
	fmt.Println("Columna 'máx enviado': lo que envía el rank más cargado, en múltiplos de S.")
	fmt.Println()
	fmt.Printf("%4s │ %16s │ %16s │ %14s\n", "GPUs", "central (máx)", "ring (máx)", "ring teórico")
	fmt.Println("─────┼──────────────────┼──────────────────┼───────────────")
	for _, n := range []int{2, 4, 8, 16, 32} {
		vec := func() [][]float64 {
			vs := make([][]float64, n)
			for r := range vs {
				vs[r] = make([]float64, *size)
				for i := range vs[r] {
					vs[r][i] = float64(r)
				}
			}
			return vs
		}
		c := collective.NewComm(n)
		collective.Run(c, collective.Central, vec())
		central := float64(c.MaxSent()) / float64(*size)
		c = collective.NewComm(n)
		collective.Run(c, collective.Ring, vec())
		ring := float64(c.MaxSent()) / float64(*size)
		fmt.Printf("%4d │ %14.2f·S │ %14.2f·S │ %11.3f·S\n", n, central, ring, 2*float64(n-1)/float64(n))
	}
	fmt.Println()
	fmt.Println("Central crece como (n-1)·S: el rank 0 es el cuello de botella.")
	fmt.Println("Ring se queda en 2(n-1)/n·S < 2·S, sin importar cuántas GPUs haya.")
}

func trainCmd(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	workers := fs.Int("workers", 4, "grado de paralelismo de datos")
	steps := fs.Int("steps", 200, "pasos de entrenamiento")
	straggler := fs.Duration("straggler", 0, "demora extra del worker 0 en cada paso (ej: 2ms)")
	failAt := fs.Int("fail-at", 0, "paso en el que muere un worker (0 = nunca)")
	ckpt := fs.Int("checkpoint-every", 50, "cada cuántos pasos se guarda un checkpoint")
	fs.Parse(args)

	ds := train.NewDataset(4096, 32, 7)
	run := func(label string, cfg train.Config) train.Result {
		res, err := train.Run(ds, cfg)
		if err != nil {
			fmt.Println("error:", err)
			os.Exit(1)
		}
		fmt.Printf("%-28s loss=%.6f  pasos ejecutados=%4d  perdidos=%3d  tiempo=%v\n",
			label, res.Loss, res.StepsRun, res.StepsLost, res.WallTime.Round(time.Millisecond))
		return res
	}
	cfg := train.Config{Workers: 1, GlobalBatch: 64, Steps: *steps, LR: 0.05, AllReduce: collective.Ring}
	single := run("1 worker", cfg)

	cfg.Workers = *workers
	cfg.StragglerDelay = *straggler
	cfg.FailAtStep, cfg.CheckpointEvery = *failAt, *ckpt
	multi := run(fmt.Sprintf("%d workers (ring)", *workers), cfg)

	var diff float64
	for i := range single.Weights {
		diff = max(diff, abs(single.Weights[i]-multi.Weights[i]))
	}
	fmt.Printf("\nDiferencia máxima entre los pesos: %.2e (sólo redondeo: la suma se hace en otro orden)\n", diff)
	if *straggler > 0 {
		fmt.Printf("Con un straggler de %v por paso, el tiempo total es al menos %d × %v: cada all-reduce espera al más lento.\n",
			*straggler, *steps, *straggler)
	}
	if *failAt > 0 {
		fmt.Printf("La falla en el paso %d hizo volver a TODOS los workers al último checkpoint.\n", *failAt)
	}
}

func pipelineCmd(args []string) {
	fs := flag.NewFlagSet("pipeline", flag.ExitOnError)
	p := fs.Int("p", 4, "etapas del pipeline (GPUs)")
	m := fs.Int("m", 8, "microbatches por batch")
	fs.Parse(args)

	for _, s := range []struct {
		name  string
		sched pipeline.Schedule
	}{{"GPipe", pipeline.GPipe}, {"1F1B (PipeDream-Flush)", pipeline.OneFOneB}} {
		tl := pipeline.Simulate(s.sched, *p, *m, 1, 2)
		fmt.Printf("%s — p=%d, m=%d, tf=1, tb=2\n", s.name, *p, *m)
		fmt.Print(tl)
		fmt.Printf("tiempo=%d  ideal=%d  burbuja=%.1f%%  (fórmula (p-1)/m = %.1f%%)\n",
			tl.Makespan, tl.Ideal(), 100*tl.BubbleFraction(), 100*float64(*p-1)/float64(*m))
		fmt.Printf("activaciones en memoria (pico por GPU): %v\n\n", tl.PeakInFlight())
	}
	fmt.Println("Números = forward del microbatch, letras = backward (a = 1), puntos = GPU ociosa.")
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
