package pipeline

import (
	"math"
	"testing"
)

// §2.2.1 del paper: la burbuja es (p-1)/m, tanto para GPipe como para 1F1B.
func TestBubbleMatchesPaperFormula(t *testing.T) {
	for _, sched := range []Schedule{GPipe, OneFOneB} {
		for _, p := range []int{1, 2, 4, 8} {
			for _, m := range []int{1, 4, 8, 16, 32} {
				tl := Simulate(sched, p, m, 1, 2)
				want := float64(p-1) / float64(m)
				if math.Abs(tl.BubbleFraction()-want) > 1e-12 {
					t.Errorf("p=%d m=%d: burbuja %.4f, want (p-1)/m = %.4f",
						p, m, tl.BubbleFraction(), want)
				}
			}
		}
	}
}

// 1F1B tiene la misma burbuja que GPipe pero guarda a lo sumo p-s
// activaciones en la etapa s, en lugar de m.
func TestOneFOneBBoundsActivationMemory(t *testing.T) {
	const p, m = 4, 16
	g := Simulate(GPipe, p, m, 1, 2).PeakInFlight()
	f := Simulate(OneFOneB, p, m, 1, 2).PeakInFlight()
	for s := range p {
		if g[s] != m {
			t.Errorf("GPipe etapa %d: %d en vuelo, want %d", s, g[s], m)
		}
		if f[s] > p-s {
			t.Errorf("1F1B etapa %d: %d en vuelo, want <= %d", s, f[s], p-s)
		}
	}
}

func TestEveryOpRunsOnce(t *testing.T) {
	tl := Simulate(OneFOneB, 4, 8, 1, 2)
	for s, slots := range tl.Stages {
		if len(slots) != 16 {
			t.Fatalf("etapa %d ejecutó %d ops, want 16", s, len(slots))
		}
		for i := 1; i < len(slots); i++ {
			if slots[i].Start < slots[i-1].End {
				t.Fatalf("etapa %d: dos operaciones solapadas", s)
			}
		}
	}
}
