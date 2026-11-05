package cluster

import "testing"

var small = Topology{Racks: 2, ServersPerRack: 2, GPUsPerServer: 8}

// Un trabajo chico va al servidor que menos lugar libre deja (best-fit):
// así los servidores vacíos quedan para los trabajos grandes.
func TestPackedIsBestFit(t *testing.T) {
	c := New(small)
	c.Allocate(Placement{{0, 6}})
	p, ok := c.Find(2, Packed)
	if !ok || len(p) != 1 || p[0].Server != 0 {
		t.Fatalf("2 GPUs: %v, %v; quería el servidor 0, que tiene justo 2 libres", p, ok)
	}
}

// Un trabajo de 16 GPUs consolidado necesita dos servidores vacíos del mismo
// rack. Con una GPU ocupada en cada rack no entra, aunque haya 30 libres:
// eso es fragmentación.
func TestFragmentationBlocksBigJobs(t *testing.T) {
	c := New(small)
	c.Allocate(Placement{{0, 1}, {2, 1}})
	if c.Free() != 30 {
		t.Fatalf("libres: %d", c.Free())
	}
	if _, ok := c.Find(16, Packed); ok {
		t.Fatal("16 GPUs consolidadas no deberían entrar")
	}
	p, ok := c.Find(16, Anywhere)
	if !ok || p.GPUs() != 16 {
		t.Fatalf("donde haya lugar: %v, %v", p, ok)
	}
	if p.Racks(small) != 2 {
		t.Errorf("debería cruzar racks: %v", p)
	}
	if f := c.Fragmentation(); f < 0.45 || f > 0.5 {
		t.Errorf("fragmentación %.2f: 14 de las 30 GPUs libres están en servidores a medio usar", f)
	}
}

// Anywhere prefiere un solo rack antes que cruzar.
func TestAnywherePrefersOneRack(t *testing.T) {
	c := New(small)
	c.Allocate(Placement{{0, 4}, {1, 4}, {2, 2}})
	p, ok := c.Find(12, Anywhere)
	if !ok || p.Racks(small) != 1 {
		t.Fatalf("12 GPUs: %v; entran en el rack 1 (14 libres)", p)
	}
}

func TestAllocateRelease(t *testing.T) {
	c := New(small)
	p, _ := c.Find(16, Packed)
	c.Allocate(p)
	if c.Free() != 16 || c.EmptyServers() != 2 {
		t.Fatalf("libres %d, vacíos %d", c.Free(), c.EmptyServers())
	}
	c.Release(p)
	if c.Free() != 32 || c.Fragmentation() != 0 {
		t.Fatalf("libres %d, fragmentación %.2f", c.Free(), c.Fragmentation())
	}
}

// Repartir a un trabajo sensible lo hace mucho más lento que a uno que no lo es.
func TestSpeed(t *testing.T) {
	ideal := Placement{{0, 8}, {1, 8}}
	spread := Placement{{0, 4}, {1, 4}, {2, 4}, {3, 4}}
	if Speed(small, ideal, true) != 1 {
		t.Error("consolidado debería correr a velocidad ideal")
	}
	if s, n := Speed(small, spread, true), Speed(small, spread, false); s >= n || s > 0.6 {
		t.Errorf("repartido: sensible %.2f, no sensible %.2f", s, n)
	}
}
