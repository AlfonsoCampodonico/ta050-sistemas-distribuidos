package flota

import (
	"math"
	"testing"
)

func run(t *testing.T, c Config, reqs []Request) Summary {
	t.Helper()
	s := Run(c, reqs)
	if s.Requests != len(reqs) {
		t.Fatalf("%s: terminaron %d de %d", c.Policy, s.Requests, len(reqs))
	}
	return s
}

func frac(s Summary, tier Tier) float64 {
	return float64(s.Loads[tier]) / float64(s.Loads[DRAM]+s.Loads[SSD]+s.Loads[Remote])
}

// La carga tiene la tasa pedida y es muy despareja: el modelo más popular
// recibe muchos más pedidos que el menos popular.
func TestWorkload(t *testing.T) {
	w := DefaultWorkload()
	reqs := w.Requests()
	want := w.Rate * w.Duration
	if got := float64(len(reqs)); got < 0.8*want || got > 1.2*want {
		t.Errorf("%0.f pedidos, se esperaban unos %.0f", got, want)
	}
	count := make([]int, w.Models)
	for _, r := range reqs {
		count[r.Model]++
	}
	if count[0] < 20*max(1, count[w.Models-1]) {
		t.Errorf("el más popular tiene %d pedidos y el menos %d", count[0], count[w.Models-1])
	}
}

// Cada nivel es más lento que el anterior, y el cargador optimizado (en
// pipeline, aprovechando todo el ancho de banda) es varias veces más rápido
// que leer nivel por nivel.
func TestLoadTimes(t *testing.T) {
	h := DefaultHardware()
	for _, l := range []Loader{Naive, Fast} {
		d, s, r := h.LoadTime(l, DRAM, 13), h.LoadTime(l, SSD, 13), h.LoadTime(l, Remote, 13)
		if !(d < s && s < r) {
			t.Errorf("%s: DRAM %.1f, SSD %.1f, remoto %.1f s", l.Name, d, s, r)
		}
	}
	if n, f := h.LoadTime(Naive, SSD, 13), h.LoadTime(Fast, SSD, 13); n < 3*f {
		t.Errorf("desde SSD: ingenuo %.1f s, optimizado %.1f s", n, f)
	}
	// Con pipeline manda el nivel más lento: desde remoto, la red.
	if got, want := h.LoadTime(Fast, Remote, 13), 13/h.Net; math.Abs(got-want) > 1e-9 {
		t.Errorf("remoto en pipeline: %.2f s, se esperaba %.2f", got, want)
	}
}

// Elegir el servidor que tiene el checkpoint más cerca de la GPU carga más
// desde DRAM, menos desde remoto, y baja la demora.
func TestLocalityBeatsRandom(t *testing.T) {
	reqs := DefaultWorkload().Requests()
	c := DefaultConfig()
	c.Policy = Random
	rnd := run(t, c, reqs)
	c.Policy = Locality
	loc := run(t, c, reqs)
	if frac(loc, DRAM) < frac(rnd, DRAM)+0.2 || frac(loc, Remote) > frac(rnd, Remote)/2 {
		t.Errorf("cargas desde DRAM: %.2f vs %.2f; desde remoto: %.2f vs %.2f",
			frac(loc, DRAM), frac(rnd, DRAM), frac(loc, Remote), frac(rnd, Remote))
	}
	if loc.DelayMean > 0.85*rnd.DelayMean {
		t.Errorf("demora media: localidad %.1f s, al azar %.1f s", loc.DelayMean, rnd.DelayMean)
	}
}

// El cargador importa más que todo lo demás.
func TestFastLoaderMatters(t *testing.T) {
	reqs := DefaultWorkload().Requests()
	c := DefaultConfig()
	c.Policy = Random
	fast := run(t, c, reqs)
	c.Loader = Naive
	naive := run(t, c, reqs)
	if naive.DelayMean < 2*fast.DelayMean {
		t.Errorf("demora media: ingenuo %.1f s, optimizado %.1f s", naive.DelayMean, fast.DelayMean)
	}
}

// Para hacerle lugar a un modelo, mudar la inferencia que estaba corriendo
// la deja parada milisegundos; cortarla y volver a empezar le hace perder
// todo lo que llevaba, y la demora total sube.
func TestMigrationBeatsPreemption(t *testing.T) {
	reqs := DefaultWorkload().Requests()
	c := DefaultConfig()
	c.GPUs = 2 // pocas GPUs: hace falta hacer lugar más seguido
	c.Policy = Migrate
	mig := run(t, c, reqs)
	c.Policy = Preempt
	pre := run(t, c, reqs)
	c.Policy = Locality
	loc := run(t, c, reqs)
	if mig.Migrations == 0 || pre.Preemptions == 0 {
		t.Fatalf("migraciones %d, desalojos %d", mig.Migrations, pre.Preemptions)
	}
	if mig.PauseMean > 0.1 || pre.PauseMean < 10*mig.PauseMean {
		t.Errorf("pausa media: migración %.3f s, desalojo %.3f s", mig.PauseMean, pre.PauseMean)
	}
	if pre.DelayMean < 1.2*mig.DelayMean || mig.DelayMean > 1.02*loc.DelayMean {
		t.Errorf("demora media: migración %.1f, desalojo %.1f, sólo localidad %.1f s", mig.DelayMean, pre.DelayMean, loc.DelayMean)
	}
}

// Mudar una inferencia mandando los tokens cuesta miles de veces menos red
// que mandar su KV cache (ServerlessLLM §5.2).
func TestMigrationBytes(t *testing.T) {
	tok, kv := MigrationBytes(2000, 40*2*5120*2)
	if kv < 1000*tok {
		t.Errorf("tokens %.0f bytes, KV cache %.0f bytes", tok, kv)
	}
}

// Keep-alive cambia arranques en frío por GPUs ociosas. Si cada arranque es
// una descarga, alargarlo baja mucho la demora; si el checkpoint suele estar
// en DRAM y se carga rápido, casi no cambia: se puede liberar la GPU antes.
func TestKeepAlive(t *testing.T) {
	reqs := DefaultWorkload().Requests()
	classic := DefaultConfig()
	classic.DRAMGB, classic.SSDGB, classic.Replicas = 0, 0, 0
	classic.Loader, classic.Policy = Naive, Random
	fast := DefaultConfig()
	at := func(c Config, k float64) Summary {
		c.KeepAlive = k
		return run(t, c, reqs)
	}
	for _, c := range []Config{classic, fast} {
		short, long := at(c, 0), at(c, 600)
		if long.Cold > short.Cold-0.15 || long.IdleWarm < short.IdleWarm+0.3 {
			t.Errorf("%s: frío %.2f → %.2f, GPU ociosa %.2f → %.2f", c.Loader.Name, short.Cold, long.Cold, short.IdleWarm, long.IdleWarm)
		}
	}
	cs, cl := at(classic, 0), at(classic, 600)
	fs, fl := at(fast, 0), at(fast, 600)
	if cs.DelayMean-cl.DelayMean < 5 || fs.DelayMean-fl.DelayMean > 1 {
		t.Errorf("demora media, keep-alive 0 → 600 s: clásico %.1f → %.1f, ServerlessLLM %.1f → %.1f",
			cs.DelayMean, cl.DelayMean, fs.DelayMean, fl.DelayMean)
	}
}

// Una GPU por modelo no tiene arranques en frío pero está ociosa casi
// siempre; la flota compartida usa cuatro veces menos GPUs, mucho más
// ocupadas (multiplexación estadística).
func TestSharedFleet(t *testing.T) {
	reqs := DefaultWorkload().Requests()
	c := DefaultConfig()
	d := Dedicated(c, reqs)
	s := run(t, c, reqs)
	if d.GPUs < 4*s.GPUs || s.Busy < 3*d.Busy {
		t.Errorf("dedicado: %d GPUs, %.0f%% ocupadas; flota: %d GPUs, %.0f%% ocupadas", d.GPUs, 100*d.Busy, s.GPUs, 100*s.Busy)
	}
}

// --- LoRA ---

func runLoRA(t *testing.T, c LoRAConfig, reqs []Request) LoRASummary {
	t.Helper()
	s := RunLoRA(c, reqs)
	if s.Requests != len(reqs) {
		t.Fatalf("%s: terminaron %d de %d", c.Mode, s.Requests, len(reqs))
	}
	return s
}

func capacity(t *testing.T, mode LoRAMode, adapters int) float64 {
	c := DefaultLoRAConfig()
	c.Mode = mode
	return runLoRA(t, c, LoRAWorkload(adapters, 40).Requests()).Throughput
}

// Con un solo adaptador, fusionarlo en los pesos es lo mejor: no cuesta nada
// por token. Con varios, gana el lote mixto (en S-LoRA, figura 9, desde tres;
// en este modelo de costo, desde dos).
func TestMergeOnlyWinsWithOneAdapter(t *testing.T) {
	if mg, mx := capacity(t, Merged, 1), capacity(t, Mixed, 1); mg <= mx {
		t.Errorf("un adaptador: fusionado %.1f, mixto %.1f pedidos/s", mg, mx)
	}
	for _, n := range []int{2, 5, 100} {
		if mg, mx := capacity(t, Merged, n), capacity(t, Mixed, n); mx <= mg {
			t.Errorf("%d adaptadores: fusionado %.1f, mixto %.1f pedidos/s", n, mg, mx)
		}
	}
}

// El lote mixto casi no pierde capacidad de 5 a 2.000 adaptadores: los que
// entran en un lote están acotados por el lote, no por cuántos hay. El
// fusionado se derrumba (S-LoRA, tabla 3).
func TestMixedBatchingScales(t *testing.T) {
	mx5, mx2000 := capacity(t, Mixed, 5), capacity(t, Mixed, 2000)
	if mx2000 < 0.85*mx5 {
		t.Errorf("lote mixto: %.1f pedidos/s con 5 adaptadores, %.1f con 2.000", mx5, mx2000)
	}
	mg1, mg2000 := capacity(t, Merged, 1), capacity(t, Merged, 2000)
	if mg2000 > 0.25*mg1 {
		t.Errorf("fusionado: %.1f pedidos/s con 1 adaptador, %.1f con 2.000", mg1, mg2000)
	}
	if mx2000 < 5*mg2000 {
		t.Errorf("2.000 adaptadores: mixto %.1f, fusionado %.1f pedidos/s", mx2000, mg2000)
	}
}

// Dejar los adaptadores en el pool unificado mientras no haga falta el lugar
// es un caché: los populares se traen una sola vez.
func TestAdapterCache(t *testing.T) {
	reqs := LoRAWorkload(200, 10).Requests()
	c := DefaultLoRAConfig()
	cached := runLoRA(t, c, reqs)
	c.Cache = false
	uncached := runLoRA(t, c, reqs)
	if cached.Loads > 200 || uncached.Loads < 3*cached.Loads {
		t.Errorf("cargas: con caché %d, sin caché %d", cached.Loads, uncached.Loads)
	}
	if cached.AdapterMem < uncached.AdapterMem {
		t.Errorf("memoria de adaptadores: con caché %.3f, sin caché %.3f", cached.AdapterMem, uncached.AdapterMem)
	}
}
