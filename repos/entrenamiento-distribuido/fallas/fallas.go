// Package fallas simula un entrenamiento largo en un clúster grande donde las
// GPUs se rompen, y mide cuánto del tiempo se usa de verdad para entrenar.
//
// Con N GPUs, el clúster falla N veces más seguido que una GPU: si cada GPU
// falla en promedio una vez cada M horas, el clúster entero falla una vez
// cada M/N. Como el entrenamiento es sincrónico, una sola GPU rota detiene a
// todas y hace volver al último checkpoint. Llama 3 405B tuvo 419
// interrupciones inesperadas en 54 días con 16.384 GPUs (Grattafiori et al.,
// 2024, §3.3.4): una cada tres horas.
//
// Guardar checkpoints seguido pierde menos trabajo en cada falla pero frena
// más el entrenamiento. El intervalo que mejor balancea las dos cosas es,
// aproximadamente, sqrt(2·C·M) con C el costo de guardar y M el tiempo medio
// entre fallas del clúster (Young, 1974; Daly, 2006).
package fallas

import (
	"math"
	"math/rand/v2"
)

// LlamaGPUMTBF es el tiempo medio entre fallas inesperadas de una GPU, en
// horas, que sale de los números de Llama 3: 54 días × 24 h × 16.384 GPUs
// sobre 419 interrupciones. Unas 50.000 horas: casi seis años por GPU.
const LlamaGPUMTBF = 54 * 24 * 16384 / 419.0

// Config describe un entrenamiento.
type Config struct {
	GPUs            int
	GPUMTBF         float64 // horas entre fallas de una GPU, en promedio
	Work            float64 // horas de cómputo útil que necesita el entrenamiento
	CheckpointEvery float64 // horas de cómputo entre checkpoints
	CheckpointCost  float64 // horas que el entrenamiento queda frenado por cada checkpoint
	RestartCost     float64 // horas para detectar la falla, reemplazar el nodo y recargar
	Runs            int     // corridas a promediar
	Seed            uint64
}

// ClusterMTBF es el tiempo medio entre fallas del clúster, en horas.
func (c Config) ClusterMTBF() float64 { return c.GPUMTBF / float64(c.GPUs) }

// YoungDaly es el intervalo entre checkpoints que minimiza el tiempo perdido,
// en primera aproximación: sqrt(2·C·M).
func (c Config) YoungDaly() float64 { return math.Sqrt(2 * c.CheckpointCost * c.ClusterMTBF()) }

// Result es el promedio de las corridas.
type Result struct {
	Wall       float64 // horas de reloj
	Failures   float64
	Lost       float64 // horas de cómputo rehechas
	Checkpoint float64 // horas frenado guardando
	Restart    float64 // horas reiniciando
	Goodput    float64 // Work / Wall: fracción del tiempo que se entrenó de verdad
}

// Simulate corre el entrenamiento con fallas exponenciales (sin memoria).
func Simulate(c Config) Result {
	runs := max(1, c.Runs)
	rng := rand.New(rand.NewPCG(c.Seed, 3))
	lambda := 1 / c.ClusterMTBF()
	var sum Result
	for range runs {
		var r Result
		done := 0.0 // trabajo ya guardado en un checkpoint
		nextFailure := rng.ExpFloat64() / lambda
		now := 0.0
		for done < c.Work {
			segment := min(c.CheckpointEvery, c.Work-done)
			// Calcular el segmento y guardarlo, salvo que algo se rompa antes.
			end := now + segment + c.CheckpointCost
			if nextFailure >= end {
				now = end
				done += segment
				r.Checkpoint += c.CheckpointCost
				continue
			}
			// Falla: se pierde lo que se calculó desde el último checkpoint.
			r.Failures++
			r.Lost += min(nextFailure-now, segment)
			r.Checkpoint += max(0, nextFailure-now-segment)
			now = nextFailure + c.RestartCost
			r.Restart += c.RestartCost
			nextFailure = now + rng.ExpFloat64()/lambda
		}
		r.Wall = now
		sum.Wall += r.Wall
		sum.Failures += r.Failures
		sum.Lost += r.Lost
		sum.Checkpoint += r.Checkpoint
		sum.Restart += r.Restart
	}
	n := float64(runs)
	res := Result{Wall: sum.Wall / n, Failures: sum.Failures / n, Lost: sum.Lost / n,
		Checkpoint: sum.Checkpoint / n, Restart: sum.Restart / n}
	res.Goodput = c.Work / res.Wall
	return res
}
