package kvcache

import "testing"

func TestPagedWastesLessThanOneBlockPerSequence(t *testing.T) {
	p := NewPaged(1024, 16)
	lengths := []int{1, 15, 16, 17, 100, 200}
	for id, n := range lengths {
		if !p.Admit(id, 1, 0) {
			t.Fatal("no entró")
		}
		for range n - 1 {
			if !p.Append(id) {
				t.Fatal("append falló")
			}
		}
	}
	st := p.Stats()
	waste := st.Allocated - st.Used
	if waste >= 16*len(lengths) {
		t.Fatalf("desperdicio %d slots, want < un bloque por secuencia (%d)", waste, 16*len(lengths))
	}
	for id, n := range lengths {
		if got, want := len(p.BlockTable(id)), (n+15)/16; got != want {
			t.Errorf("seq %d de %d tokens usa %d bloques, want %d", id, n, got, want)
		}
	}
}

func TestPagedRunsOutAndFrees(t *testing.T) {
	p := NewPaged(64, 16) // 4 bloques
	if !p.Admit(1, 48, 0) || !p.Admit(2, 16, 0) {
		t.Fatal("tenían que entrar")
	}
	if p.Admit(3, 1, 0) {
		t.Fatal("no queda ningún bloque")
	}
	if p.Append(2) {
		t.Fatal("el bloque de 2 está lleno y no hay otro")
	}
	p.Free(1)
	if p.FreeBlocks() != 3 || !p.Append(2) {
		t.Fatal("liberar tiene que devolver los 3 bloques")
	}
}

func TestCopyOnWrite(t *testing.T) {
	p := NewPaged(1024, 16)
	p.Admit(0, 40, 0) // 3 bloques: 16 + 16 + 8
	for child := 1; child <= 3; child++ {
		p.Fork(0, child)
	}
	if got := 64 - p.FreeBlocks(); got != 3 {
		t.Fatalf("4 secuencias que comparten el prompt usan %d bloques, want 3", got)
	}
	// Cada muestra escribe su primer token propio: se copia sólo el último
	// bloque, que está compartido y a medio llenar.
	for id := 0; id <= 3; id++ {
		if !p.Append(id) {
			t.Fatal("append falló")
		}
	}
	if p.Copies() != 3 {
		t.Fatalf("copias = %d, want 3 (la última muestra se queda con el original)", p.Copies())
	}
	if got := 64 - p.FreeBlocks(); got != 6 {
		t.Fatalf("bloques usados = %d, want 6 (2 compartidos + 4 propios)", got)
	}
	for id := 0; id <= 3; id++ {
		p.Free(id)
	}
	if p.FreeBlocks() != 64 {
		t.Fatalf("quedaron %d bloques sin liberar", 64-p.FreeBlocks())
	}
}

func TestContiguousReservesTheMaximum(t *testing.T) {
	c := NewContiguous(4096)
	if !c.Admit(1, 100, 2048) || !c.Admit(2, 100, 2048) {
		t.Fatal("tenían que entrar")
	}
	if c.Admit(3, 10, 2048) {
		t.Fatal("con dos reservas de 2048 no entra una tercera")
	}
	st := c.Stats()
	if st.Used != 200 || st.Allocated != 4096 {
		t.Fatalf("stats %+v", st)
	}
	if u := st.Utilization(); u > 0.05 {
		t.Fatalf("utilización %.3f, want < 5%%", u)
	}
}

func TestContiguousExternalFragmentation(t *testing.T) {
	c := NewContiguous(300)
	c.Admit(1, 100, 100)
	c.Admit(2, 100, 100)
	c.Admit(3, 100, 100)
	c.Free(1)
	c.Free(3)
	// Hay 200 libres, pero en dos huecos de 100: no entra una de 150.
	if c.Admit(4, 150, 150) {
		t.Fatal("no debería entrar: fragmentación externa")
	}
	if f := c.ExternalFragmentation(); f != 0.5 {
		t.Fatalf("fragmentación externa %.2f, want 0.50", f)
	}
	c.Free(2)
	if !c.Admit(4, 150, 150) {
		t.Fatal("al liberar el del medio los huecos se unen")
	}
}
