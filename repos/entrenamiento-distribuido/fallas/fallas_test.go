package fallas

import (
	"math"
	"testing"
)

func llama(gpus int) Config {
	return Config{GPUs: gpus, GPUMTBF: LlamaGPUMTBF, Work: 54 * 24, CheckpointCost: 0.5 / 60, RestartCost: 3.0 / 60, Runs: 200, Seed: 1}
}

// Con los números de Llama 3, el clúster de 16.384 GPUs falla cada ~3 horas.
func TestClusterMTBF(t *testing.T) {
	if m := llama(16384).ClusterMTBF(); math.Abs(m-3.09) > 0.01 {
		t.Fatalf("MTBF del clúster = %.2f h", m)
	}
}

// El intervalo de Young/Daly le gana a checkpoints mucho más frecuentes o mucho
// más espaciados, y da más de 90% de tiempo útil como en Llama 3.
func TestYoungDalyIsNearOptimal(t *testing.T) {
	c := llama(16384)
	opt := c.YoungDaly()
	good := func(tau float64) float64 { c.CheckpointEvery = tau; return Simulate(c).Goodput }
	best := good(opt)
	if best < 0.9 {
		t.Fatalf("tiempo útil en el óptimo %.3f", best)
	}
	for _, tau := range []float64{opt / 4, opt * 4} {
		if g := good(tau); g >= best {
			t.Errorf("checkpoint cada %.2f h dio %.3f ≥ óptimo %.3f", tau, g, best)
		}
	}
}

// Más GPUs, más fallas, menos tiempo útil, aun en el óptimo.
func TestScaleHurts(t *testing.T) {
	prev := 1.0
	for _, n := range []int{4096, 16384, 65536, 131072} {
		c := llama(n)
		c.CheckpointEvery = c.YoungDaly()
		g := Simulate(c).Goodput
		if g >= prev {
			t.Errorf("%d GPUs: %.3f ≥ %.3f", n, g, prev)
		}
		prev = g
	}
}
