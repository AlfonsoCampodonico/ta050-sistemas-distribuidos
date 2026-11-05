package cluster

import (
	"testing"

	"github.com/fiubaTA050/servir-tokens/engine"
)

var slo = SLO{TTFT: 2000, MaxGap: 100}

func setup(sessions int, rate float64) ([]engine.Request, Config) {
	ch := engine.DefaultChat()
	ch.Sessions, ch.Rate = sessions, rate
	c := Config{
		Engine:      engine.Config{Mode: engine.Continuous, MaxBatch: 128, MaxLen: ch.MaxLen, Cost: engine.DefaultCost(), PrefixBlock: ch.Block},
		KVTokens:    1 << 18,
		Block:       16,
		CacheBlocks: 1024,
	}
	return ch.Requests(), c
}

func run(t *testing.T, c Config, n int, r Router, reqs []engine.Request) Summary {
	t.Helper()
	s, err := Run(c, n, r, reqs, slo)
	if err != nil {
		t.Fatal(err)
	}
	if s.Requests != len(reqs) {
		t.Fatalf("%s: terminaron %d de %d", r.Name(), s.Requests, len(reqs))
	}
	return s
}

// El caché de prefijos evita la mayor parte del prefill en una carga de chat,
// y el router que lo conoce aprovecha más que el que reparte en ronda.
func TestPrefixAwareRoutingReusesMore(t *testing.T) {
	reqs, c := setup(600, 5)
	rr := run(t, c, 4, &RoundRobin{}, reqs)
	pa := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	if pa.HitRate < rr.HitRate+0.1 {
		t.Errorf("caché: consciente del prefijo %.2f vs ronda %.2f", pa.HitRate, rr.HitRate)
	}
	if pa.TTFTP99 > rr.TTFTP99 {
		t.Errorf("TTFT p99: consciente del prefijo %.0f vs ronda %.0f ms", pa.TTFTP99, rr.TTFTP99)
	}
}

// Sticky consigue el mismo caché pero no mira la carga: su cola es peor.
func TestStickyIgnoresLoad(t *testing.T) {
	reqs, c := setup(1000, 5)
	st := run(t, c, 4, Sticky{}, reqs)
	pa := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	if st.Imbalance < pa.Imbalance || st.TTFTP99 < 2*pa.TTFTP99 {
		t.Errorf("sticky: desbalance %.2f, p99 %.0f ms; consciente: %.2f, %.0f ms", st.Imbalance, st.TTFTP99, pa.Imbalance, pa.TTFTP99)
	}
}

// Un pool compartido entre réplicas sube el reuso: Mooncake.
func TestSharedPoolReusesMore(t *testing.T) {
	reqs, c := setup(600, 5)
	local := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	c.PoolBlocks = 1 << 16
	pooled := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	if pooled.HitRate < local.HitRate+0.05 {
		t.Errorf("caché con pool %.2f vs sólo local %.2f", pooled.HitRate, local.HitRate)
	}
}

// Prefill en pedazos (Sarathi-Serve): los decodes dejan de esperar a los
// prompts largos.
func TestChunkedPrefillBoundsTBT(t *testing.T) {
	reqs, c := setup(600, 5)
	whole := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	c.Engine.ChunkPrefill = 512
	chunked := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	if whole.TBTP99 < slo.MaxGap || chunked.TBTP99 > slo.MaxGap {
		t.Errorf("TBT p99: entero %.0f ms, en pedazos %.0f ms (SLO %.0f)", whole.TBTP99, chunked.TBTP99, slo.MaxGap)
	}
}

// Desagregado: cada pedido recibe todos sus tokens, uno de prefill y el
// resto de decode, y el decode no sufre los prefills.
func TestDisaggregatedDeliversEveryToken(t *testing.T) {
	reqs, c := setup(300, 3)
	s, err := RunDisaggregated(c, 2, 2, 0.003, reqs, slo)
	if err != nil {
		t.Fatal(err)
	}
	if s.Requests != len(reqs) {
		t.Fatalf("terminaron %d de %d", s.Requests, len(reqs))
	}
	colocated := run(t, c, 4, PrefixAware{Cost: c.Engine.Cost}, reqs)
	if s.TBTP99 >= colocated.TBTP99 {
		t.Errorf("TBT p99 desagregado %.0f vs colocado %.0f ms", s.TBTP99, colocated.TBTP99)
	}
}
