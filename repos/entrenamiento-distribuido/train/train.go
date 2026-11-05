// Package train entrena un modelo de juguete (regresión lineal) con
// paralelismo de datos: cada worker tiene una réplica completa del modelo y
// una porción del batch; después de calcular su gradiente local, todos hacen
// un all-reduce y aplican exactamente la misma actualización.
//
// El modelo es deliberadamente trivial. Lo que importa es la estructura del
// sistema distribuido, que es la misma que en un LLM de un billón de
// parámetros: réplicas, all-reduce sincrónico, barrera por paso, checkpoints.
package train

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/fiubaTA050/entrenamiento-distribuido/collective"
)

// Dataset: y = X·w* + ruido, generado de forma determinista.
type Dataset struct {
	X [][]float64
	Y []float64
}

// NewDataset genera n ejemplos de dimensión dim con una semilla fija.
func NewDataset(n, dim int, seed uint64) Dataset {
	rng := rand.New(rand.NewPCG(seed, 1))
	wStar := make([]float64, dim)
	for i := range wStar {
		wStar[i] = rng.NormFloat64()
	}
	d := Dataset{X: make([][]float64, n), Y: make([]float64, n)}
	for i := range n {
		x := make([]float64, dim)
		var y float64
		for j := range x {
			x[j] = rng.NormFloat64()
			y += x[j] * wStar[j]
		}
		d.X[i], d.Y[i] = x, y+0.01*rng.NormFloat64()
	}
	return d
}

// Config describe un entrenamiento.
type Config struct {
	Workers         int                  // grado de paralelismo de datos (d)
	GlobalBatch     int                  // B: ejemplos por paso, repartidos entre workers
	Steps           int                  // pasos de optimización
	LR              float64              // learning rate
	AllReduce       collective.AllReduce // Central o Ring
	CheckpointEvery int                  // 0 = sin checkpoints
	FailAtStep      int                  // si > 0, un worker "muere" en ese paso (una sola vez)
	StragglerDelay  time.Duration        // demora extra del worker 0 en cada paso
}

// Result resume un entrenamiento.
type Result struct {
	Weights      []float64
	Loss         float64
	StepsRun     int // pasos ejecutados, incluyendo los que hubo que rehacer
	StepsLost    int // pasos perdidos por fallas
	Restarts     int
	WallTime     time.Duration
	FloatsPerRun int64 // floats que envió el rank más cargado en todo el entrenamiento
}

var errWorkerDied = errors.New("un worker dejó de responder")

// Run entrena con la configuración dada.
func Run(ds Dataset, cfg Config) (Result, error) {
	if cfg.GlobalBatch%cfg.Workers != 0 {
		return Result{}, fmt.Errorf("el batch global (%d) tiene que ser múltiplo de la cantidad de workers (%d)",
			cfg.GlobalBatch, cfg.Workers)
	}
	dim := len(ds.X[0])
	comm := collective.NewComm(cfg.Workers)
	start := time.Now()

	// Estado del modelo. Todas las réplicas son idénticas, así que alcanza
	// con guardar una copia como checkpoint.
	w := make([]float64, dim)
	ckpt, ckptStep := append([]float64(nil), w...), 0
	failed := false
	var res Result

	for step := 0; step < cfg.Steps; step++ {
		if cfg.CheckpointEvery > 0 && step%cfg.CheckpointEvery == 0 {
			ckpt, ckptStep = append(ckpt[:0], w...), step
		}

		injectFailure := cfg.FailAtStep > 0 && step == cfg.FailAtStep && !failed
		next, err := dataParallelStep(ds, cfg, comm, w, step, injectFailure)
		res.StepsRun++
		if err != nil {
			// El entrenamiento es sincrónico: si un worker falla, el paso
			// no termina para NADIE. Volvemos todos al último checkpoint.
			failed = true
			res.Restarts++
			res.StepsLost += step - ckptStep + 1
			copy(w, ckpt)
			step = ckptStep - 1 // el for lo incrementa
			continue
		}
		w = next
	}

	res.Weights = w
	res.Loss = loss(ds, w)
	res.WallTime = time.Since(start)
	res.FloatsPerRun = comm.MaxSent()
	return res, nil
}

// dataParallelStep ejecuta un paso de SGD con d workers y devuelve los pesos
// nuevos (idénticos en todas las réplicas).
func dataParallelStep(ds Dataset, cfg Config, comm *collective.Comm, w []float64, step int, fail bool) ([]float64, error) {
	d := cfg.Workers
	local := cfg.GlobalBatch / d
	replicas := make([][]float64, d)
	grads := make([][]float64, d)

	// Fase de cómputo: cada worker procesa SU porción del batch.
	// El batch del paso depende sólo de step: es determinista, así que
	// re-ejecutar un paso después de una falla da el mismo resultado.
	var wg sync.WaitGroup
	for r := range d {
		replicas[r] = append([]float64(nil), w...)
		wg.Add(1)
		go func() {
			defer wg.Done()
			if r == 0 && cfg.StragglerDelay > 0 {
				time.Sleep(cfg.StragglerDelay)
			}
			lo := (step*cfg.GlobalBatch + r*local) % len(ds.X)
			grads[r] = gradient(ds, replicas[r], lo, local)
		}()
	}
	wg.Wait()

	if fail {
		return nil, errWorkerDied
	}

	// Fase de comunicación: all-reduce de los gradientes. Es una barrera:
	// nadie avanza hasta que llegaron los gradientes de todos.
	collective.Run(comm, cfg.AllReduce, grads)

	// Fase de actualización: cada réplica aplica la misma cuenta sobre el
	// mismo gradiente promedio, así que siguen siendo idénticas.
	for r := range d {
		for i := range replicas[r] {
			replicas[r][i] -= cfg.LR * grads[r][i] / float64(d)
		}
	}
	for r := 1; r < d; r++ {
		for i := range replicas[0] {
			if replicas[r][i] != replicas[0][i] {
				return nil, fmt.Errorf("las réplicas divergieron en el paso %d", step)
			}
		}
	}
	return replicas[0], nil
}

// gradient devuelve el gradiente del error cuadrático medio sobre count
// ejemplos a partir de lo (circular sobre el dataset).
func gradient(ds Dataset, w []float64, lo, count int) []float64 {
	g := make([]float64, len(w))
	for k := range count {
		i := (lo + k) % len(ds.X)
		x := ds.X[i]
		var pred float64
		for j := range w {
			pred += w[j] * x[j]
		}
		e := pred - ds.Y[i]
		for j := range w {
			g[j] += 2 * e * x[j] / float64(count)
		}
	}
	return g
}

func loss(ds Dataset, w []float64) float64 {
	var l float64
	for i, x := range ds.X {
		var pred float64
		for j := range w {
			pred += w[j] * x[j]
		}
		e := pred - ds.Y[i]
		l += e * e
	}
	return l / float64(len(ds.X))
}

// Gradient es el gradiente del error cuadrático medio sobre count ejemplos a
// partir de lo, para los demás paquetes.
func Gradient(ds Dataset, w []float64, lo, count int) []float64 { return gradient(ds, w, lo, count) }

// Loss es el error cuadrático medio sobre todo el dataset.
func Loss(ds Dataset, w []float64) float64 { return loss(ds, w) }
