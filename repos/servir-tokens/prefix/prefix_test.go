package prefix

import "testing"

func chain(owner uint64, n int) []uint64 {
	hs := make([]uint64, n)
	prev := uint64(0)
	for i := range hs {
		prev = Hash(prev, owner<<32|uint64(i))
		hs[i] = prev
	}
	return hs
}

func TestMatchStopsAtFirstMiss(t *testing.T) {
	c := New(100)
	a := chain(1, 10)
	c.Insert(a[:6])
	if got := c.Match(a); got != 6 {
		t.Fatalf("match = %d, want 6", got)
	}
	// Otro prompt con el mismo bloque 0 pero distinto contenido no comparte
	// nada después: el hash encadenado arrastra todo lo anterior.
	b := chain(2, 10)
	if got := c.Peek(b); got != 0 {
		t.Fatalf("prompts distintos comparten %d bloques", got)
	}
}

func TestLRUKeepsWhatIsUsed(t *testing.T) {
	c := New(10)
	hot, cold := chain(1, 5), chain(2, 5)
	c.Insert(hot)
	c.Insert(cold)
	c.Match(hot) // hot pasa a ser el más reciente
	c.Insert(chain(3, 5))
	if c.Peek(hot) != 5 || c.Peek(cold) != 0 {
		t.Fatalf("hot %d, cold %d: tenía que desalojar cold", c.Peek(hot), c.Peek(cold))
	}
	if c.Len() != 10 {
		t.Fatalf("len = %d", c.Len())
	}
}
