package train

import (
	"math"
	"testing"

	"github.com/fiubaTA050/entrenamiento-distribuido/collective"
)

func base() Config {
	return Config{Workers: 1, GlobalBatch: 64, Steps: 200, LR: 0.05, AllReduce: collective.Ring}
}

func approx(t *testing.T, a, b []float64, tol float64) {
	t.Helper()
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			t.Fatalf("pos %d: %v vs %v", i, a[i], b[i])
		}
	}
}

// Paralelismo de datos con semántica estricta: con el mismo batch global, d
// workers producen el mismo modelo que uno solo (salvo redondeo, porque la
// suma se hace en otro orden).
func TestDataParallelMatchesSingleWorker(t *testing.T) {
	ds := NewDataset(1024, 16, 7)
	single, err := Run(ds, base())
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range []int{2, 4, 8} {
		for _, op := range []collective.AllReduce{collective.Central, collective.Ring} {
			cfg := base()
			cfg.Workers, cfg.AllReduce = d, op
			got, err := Run(ds, cfg)
			if err != nil {
				t.Fatal(err)
			}
			approx(t, got.Weights, single.Weights, 1e-9)
		}
	}
}

func TestTrainingConverges(t *testing.T) {
	ds := NewDataset(1024, 16, 7)
	res, err := Run(ds, base())
	if err != nil {
		t.Fatal(err)
	}
	if res.Loss > 1e-3 {
		t.Fatalf("loss final %v, esperaba < 1e-3", res.Loss)
	}
}

// Una falla obliga a todos a volver al último checkpoint, pero como cada paso
// es determinista, el resultado final es el mismo que sin falla.
func TestRecoveryFromCheckpointIsExact(t *testing.T) {
	ds := NewDataset(1024, 16, 7)
	cfg := base()
	cfg.Workers, cfg.CheckpointEvery = 4, 50
	clean, _ := Run(ds, cfg)

	cfg.FailAtStep = 120
	faulty, err := Run(ds, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if faulty.Restarts != 1 || faulty.StepsLost != 21 { // checkpoint en 100, falla en 120
		t.Fatalf("restarts=%d lost=%d, want 1 y 21", faulty.Restarts, faulty.StepsLost)
	}
	if faulty.StepsRun != cfg.Steps+faulty.StepsLost {
		t.Fatalf("steps run %d", faulty.StepsRun)
	}
	approx(t, faulty.Weights, clean.Weights, 0)
}

func TestBatchMustDivide(t *testing.T) {
	cfg := base()
	cfg.Workers = 3
	if _, err := Run(NewDataset(10, 2, 1), cfg); err == nil {
		t.Fatal("esperaba error con batch 64 y 3 workers")
	}
}
