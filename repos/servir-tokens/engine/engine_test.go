package engine

import (
	"testing"

	"github.com/fiubaTA050/servir-tokens/kvcache"
)

const maxLen = 2048

func cfg(mode Mode) Config {
	return Config{Mode: mode, MaxBatch: 32, MaxLen: maxLen, Cost: DefaultCost()}
}

func run(t *testing.T, c Config, kv kvcache.Allocator, reqs []Request) Summary {
	t.Helper()
	e := New(c, kv)
	sum, err := Run(e, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if sum.Requests != len(reqs) {
		t.Fatalf("terminaron %d de %d pedidos", sum.Requests, len(reqs))
	}
	if st := kv.Stats(); st.Allocated != 0 {
		t.Fatalf("quedó memoria asignada: %+v", st)
	}
	return sum
}

// Cada pedido recibe exactamente sus tokens, en orden, una sola vez,
// aunque lo desalojen y lo vuelvan a admitir.
func TestEveryRequestGetsAllItsTokens(t *testing.T) {
	reqs := Workload(200, 20, 1, maxLen)
	for _, kv := range []kvcache.Allocator{kvcache.NewPaged(4096, 16), kvcache.NewContiguous(64 * maxLen)} {
		for _, mode := range []Mode{Static, Continuous} {
			e := New(cfg(mode), kv)
			seen := map[int]int{}
			e.OnToken(func(tok Token) {
				if tok.Index != seen[tok.Request] {
					t.Fatalf("pedido %d: token %d, want %d", tok.Request, tok.Index, seen[tok.Request])
				}
				seen[tok.Request]++
			})
			if _, err := Run(e, reqs); err != nil {
				t.Fatal(err)
			}
			for _, r := range reqs {
				if seen[r.ID] != r.Output {
					t.Fatalf("%v: pedido %d recibió %d tokens, want %d", mode, r.ID, seen[r.ID], r.Output)
				}
			}
		}
	}
}

// Orca: con batching continuo las respuestas cortas no esperan a las largas.
func TestContinuousBatchingBeatsStatic(t *testing.T) {
	reqs := Workload(400, 10, 2, maxLen)
	static := run(t, cfg(Static), kvcache.NewPaged(1<<20, 16), reqs)
	cont := run(t, cfg(Continuous), kvcache.NewPaged(1<<20, 16), reqs)
	if cont.LatencyMean*2 > static.LatencyMean {
		t.Errorf("latencia continuo %.0f ms vs estático %.0f ms: esperaba al menos 2× mejor", cont.LatencyMean, static.LatencyMean)
	}
	if cont.TTFTMean*10 > static.TTFTMean {
		t.Errorf("TTFT continuo %.0f ms vs estático %.0f ms: esperaba al menos 10× mejor", cont.TTFTMean, static.TTFTMean)
	}
}

// vLLM: con la misma memoria, los bloques dejan correr muchas más secuencias
// a la vez que la reserva contigua del máximo.
func TestPagingFitsMoreSequences(t *testing.T) {
	const capacity = 16 * maxLen
	reqs := Workload(400, 40, 3, maxLen)
	c := cfg(Continuous)
	c.MaxBatch = 256
	contig := run(t, c, kvcache.NewContiguous(capacity), reqs)
	paged := run(t, c, kvcache.NewPaged(capacity, 16), reqs)
	if contig.Engine.MaxBatch > 16 {
		t.Fatalf("reservando 2048 por pedido no pueden correr más de 16, corrieron %d", contig.Engine.MaxBatch)
	}
	if paged.Engine.AvgBatch < 2*contig.Engine.AvgBatch {
		t.Errorf("lote promedio paginado %.1f vs contiguo %.1f: esperaba al menos 2×", paged.Engine.AvgBatch, contig.Engine.AvgBatch)
	}
	// El paper mide que los sistemas previos usan entre 20,4% y 38,2% de la
	// memoria del KV cache para tokens de verdad.
	if paged.Engine.AvgKVUtil < 0.9 || contig.Engine.AvgKVUtil > 0.4 {
		t.Errorf("utilización del KV: paginado %.2f, contiguo %.2f", paged.Engine.AvgKVUtil, contig.Engine.AvgKVUtil)
	}
	if paged.Throughput < 1.5*contig.Throughput {
		t.Errorf("throughput paginado %.0f vs contiguo %.0f tok/s", paged.Throughput, contig.Throughput)
	}
}

// Con poca memoria hay desalojos, y aun así todos terminan.
func TestPreemptionRecomputes(t *testing.T) {
	reqs := Workload(150, 50, 4, maxLen)
	c := cfg(Continuous)
	c.MaxBatch = 256
	sum := run(t, c, kvcache.NewPaged(2*maxLen, 16), reqs)
	if sum.Engine.Preemptions == 0 {
		t.Fatal("con 4096 slots y 50 pedidos/s tenía que haber desalojos")
	}
}

func TestCancelFreesMemory(t *testing.T) {
	kv := kvcache.NewPaged(4096, 16)
	e := New(cfg(Continuous), kv)
	e.Submit(Request{ID: 1, Prompt: 100, Output: 500})
	e.Submit(Request{ID: 2, Prompt: 100, Output: 500})
	for range 10 {
		e.Step()
	}
	if !e.Cancel(1) || e.Running() != 1 {
		t.Fatal("cancelar tenía que sacar al pedido del lote")
	}
	e.Cancel(2)
	if st := kv.Stats(); st.Allocated != 0 {
		t.Fatalf("quedaron %d slots asignados", st.Allocated)
	}
}
