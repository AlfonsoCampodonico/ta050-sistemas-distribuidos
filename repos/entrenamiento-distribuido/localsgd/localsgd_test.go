package localsgd

import (
	"testing"

	"github.com/fiubaTA050/entrenamiento-distribuido/train"
)

// Sincronizar cada 10 pasos llega al mismo error con 10 veces menos rondas.
func TestInfrequentSyncConverges(t *testing.T) {
	ds := train.NewDataset(4096, 16, 3)
	c := Config{Workers: 4, TotalSteps: 300, Batch: 16, LR: 0.02, OuterLR: 1}
	c.InnerSteps = 1
	every := Run(ds, c)
	c.InnerSteps = 10
	local := Run(ds, c)
	if local.Syncs*10 != every.Syncs {
		t.Fatalf("rondas: %d vs %d", local.Syncs, every.Syncs)
	}
	if local.Loss > 2*every.Loss {
		t.Fatalf("loss cada 10 pasos %.2e vs cada paso %.2e", local.Loss, every.Loss)
	}
}
