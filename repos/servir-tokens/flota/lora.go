package flota

// LoRA: muchos fine-tunes de un mismo modelo base en una GPU.
//
// Un adaptador LoRA cambia cada matriz W del modelo por W + A·B, con A y B
// de rango r chico. Hay dos formas de servir muchos:
//
//   - Fusionar (Merged): sumar A·B a los pesos y servir ese modelo. No cuesta
//     nada por token, pero un lote sólo puede tener pedidos de un adaptador,
//     y cambiar de adaptador es restar uno y sumar otro con la GPU parada.
//   - Lote mixto (Mixed, S-LoRA): dejar W sola, calcular x·W para todo el
//     lote junto y x·A·B aparte para cada pedido con su adaptador. Cuesta un
//     poco por pedido y por adaptador distinto en el lote, pero cualquier
//     mezcla de adaptadores entra en el mismo lote.
//
// Los adaptadores viven en la memoria del host y se traen a la GPU cuando un
// pedido los necesita. En la GPU comparten un único pool paginado con el KV
// cache (Unified Paging, S-LoRA §5.1): un adaptador de rango r ocupa unas
// páginas, un pedido de s tokens otras, y las dos cosas compiten por el
// mismo lugar.

import "math"

// LoRAMode es cómo se arma el lote.
type LoRAMode int

const (
	Merged LoRAMode = iota
	Mixed
)

func (m LoRAMode) String() string {
	return [...]string{"fusionado, un adaptador por lote", "lote mixto (S-LoRA)"}[m]
}

// LoRACost es el costo de una iteración, en milisegundos. Del orden de un
// modelo de 7B en una GPU de 80 GB: lo que importa es que leer los pesos
// (Base) domina, y que lo de LoRA es chico.
type LoRACost struct {
	Base            float64 // leer los pesos del modelo base, una vez por iteración
	PerSeq          float64 // por secuencia del lote
	PerPrefillToken float64 // por token de prompt
	PerLoRASeq      float64 // x·A·B de cada secuencia (lote mixto)
	PerAdapter      float64 // por adaptador distinto en el lote (lote mixto)
	Merge           float64 // cambiar el adaptador fusionado: restar uno y sumar otro
	LoadPerRank     float64 // traer un adaptador del host a la GPU, por unidad de rango
}

// DefaultLoRACost: un adaptador de rango 8 de un modelo de 7B pesa unos
// 17 MB y se trae en menos de un milisegundo.
func DefaultLoRACost() LoRACost {
	return LoRACost{Base: 8, PerSeq: 0.1, PerPrefillToken: 0.02, PerLoRASeq: 0.02, PerAdapter: 0.03, Merge: 10, LoadPerRank: 0.08}
}

// LoRAConfig describe el servidor.
type LoRAConfig struct {
	Mode         LoRAMode
	MaxBatch     int
	PoolTokens   int   // el pool unificado de la GPU, en lugares de un token de KV cache
	Ranks        []int // rango de cada adaptador, asignado en ronda (S-LoRA, tabla 1)
	PagesPerRank int   // lugares del pool por unidad de rango: A y B de q, k, v y o son 8 vectores por rango, contra 2 (K y V) por token
	Cache        bool  // dejar en la GPU los adaptadores que el lote ya no usa, y desalojarlos LRU cuando falte lugar
	MaxWaitMs    float64
	Cost         LoRACost
}

// DefaultLoRAConfig es un Llama-7B en una A100 de 80 GB, con adaptadores de
// rangos 64, 32, 16 y 8 (la configuración S2 de S-LoRA).
func DefaultLoRAConfig() LoRAConfig {
	return LoRAConfig{Mode: Mixed, MaxBatch: 128, PoolTokens: 120000, Ranks: []int{64, 32, 16, 8},
		PagesPerRank: 4, Cache: true, MaxWaitMs: 2000, Cost: DefaultLoRACost()}
}

// LoRAWorkload es la carga de S-LoRA (tabla 2, 7B @ A100 80G): llegadas de
// Poisson, popularidad de los adaptadores con ley de potencia de exponente 1,
// prompts y respuestas uniformes entre 8 y 512 tokens, cinco minutos.
func LoRAWorkload(adapters int, rate float64) Workload {
	return Workload{Models: adapters, Rate: rate, CV: 1, Skew: 1, Duration: 300,
		PromptMin: 8, PromptMax: 512, OutputMin: 8, OutputMax: 512, Seed: 3}
}

// LoRASummary es lo que mide el servidor.
type LoRASummary struct {
	Requests   int
	Throughput float64 // pedidos por segundo terminados
	TTFTP50    float64 // ms
	TTFTP99    float64
	AvgBatch   float64
	Switches   int     // cambios de adaptador fusionado
	Loads      int     // adaptadores traídos del host
	LoadMs     float64 // tiempo de GPU trayendo adaptadores
	AdapterMem float64 // fracción promedio del pool ocupada por adaptadores
	Resident   float64 // adaptadores en la GPU, en promedio
	Distinct   float64 // adaptadores distintos por lote, en promedio
}

type lseq struct {
	r         Request
	adapter   int
	generated int
	prefilled bool
}

// RunLoRA sirve los pedidos; Request.Model es el adaptador de cada uno.
func RunLoRA(c LoRAConfig, reqs []Request) LoRASummary {
	rank := func(a int) int { return c.Ranks[a%len(c.Ranks)] }
	pages := func(a int) int { return rank(a) * c.PagesPerRank }

	var (
		now        float64 // ms
		waiting    []Request
		running    []*lseq
		next       int
		poolUsed   int
		resident   = map[int]float64{} // adaptador → último uso
		cur        = -1                // adaptador fusionado
		draining   bool
		s          LoRASummary
		ttfts      []float64
		iters      int
		batchSum   int
		memSum     float64
		resSum     float64
		distSum    int
		lastFinish float64
	)
	inUse := func(a int) bool {
		for _, q := range running {
			if q.adapter == a {
				return true
			}
		}
		return false
	}
	// evict libera adaptadores que el lote no usa, del menos reciente al más,
	// hasta tener need lugares libres.
	evict := func(need int) bool {
		for poolUsed+need > c.PoolTokens {
			victim, oldest := -1, math.Inf(1)
			for a, t := range resident {
				if (t < oldest || (t == oldest && a < victim)) && !inUse(a) {
					victim, oldest = a, t
				}
			}
			if victim < 0 {
				return false
			}
			delete(resident, victim)
			poolUsed -= pages(victim)
		}
		return true
	}

	for s.Requests < len(reqs) {
		for next < len(reqs) && reqs[next].Arrival*1000 <= now {
			waiting = append(waiting, reqs[next])
			next++
		}
		step := 0.0
		admitted := 0

		switch c.Mode {
		case Mixed:
			for len(waiting) > 0 && len(running) < c.MaxBatch {
				r := waiting[0]
				need := r.Prompt + r.Output
				_, isRes := resident[r.Model]
				if !isRes {
					need += pages(r.Model)
				}
				if !evict(need) {
					break
				}
				if !isRes {
					step += c.Cost.LoadPerRank * float64(rank(r.Model))
					s.LoadMs += c.Cost.LoadPerRank * float64(rank(r.Model))
					s.Loads++
					poolUsed += pages(r.Model)
				}
				resident[r.Model] = now
				poolUsed += r.Prompt + r.Output
				running = append(running, &lseq{r: r, adapter: r.Model})
				waiting = waiting[1:]
				admitted++
			}
		case Merged:
			if len(running) == 0 {
				draining = false
				if len(waiting) > 0 && waiting[0].Model != cur {
					cur = waiting[0].Model
					step += c.Cost.Merge
					s.Switches++
					s.Loads++
					step += c.Cost.LoadPerRank * float64(rank(cur))
					s.LoadMs += c.Cost.LoadPerRank * float64(rank(cur))
				}
			} else if !draining {
				for _, r := range waiting {
					if r.Model != cur && now-r.Arrival*1000 > c.MaxWaitMs {
						draining = true // alguien de otro adaptador espera demasiado: no entra nadie más
						break
					}
				}
			}
			if !draining {
				kept := waiting[:0]
				for _, r := range waiting {
					if r.Model == cur && len(running) < c.MaxBatch && poolUsed+r.Prompt+r.Output <= c.PoolTokens {
						poolUsed += r.Prompt + r.Output
						running = append(running, &lseq{r: r, adapter: r.Model})
						admitted++
					} else {
						kept = append(kept, r)
					}
				}
				waiting = kept
			}
		}

		if len(running) == 0 {
			if next < len(reqs) {
				now = max(now+step, reqs[next].Arrival*1000)
			} else {
				now += step
			}
			continue
		}

		prefill := 0
		adapters := map[int]bool{}
		for _, q := range running {
			if !q.prefilled {
				prefill += q.r.Prompt
			}
			adapters[q.adapter] = true
		}
		n := float64(len(running))
		step += c.Cost.Base + c.Cost.PerSeq*n + c.Cost.PerPrefillToken*float64(prefill)
		if c.Mode == Mixed {
			step += c.Cost.PerLoRASeq*n + c.Cost.PerAdapter*float64(len(adapters))
		}
		now += step
		iters++
		batchSum += len(running)
		adMem := 0
		for a := range resident {
			adMem += pages(a)
		}
		memSum += float64(adMem) / float64(c.PoolTokens)
		resSum += float64(len(resident))
		distSum += len(adapters)

		kept := running[:0]
		for _, q := range running {
			if !q.prefilled {
				q.prefilled = true
				ttfts = append(ttfts, now-q.r.Arrival*1000)
			}
			q.generated++
			if c.Mode == Mixed {
				resident[q.adapter] = now
			}
			if q.generated < q.r.Output {
				kept = append(kept, q)
				continue
			}
			poolUsed -= q.r.Prompt + q.r.Output
			s.Requests++
			lastFinish = now
		}
		running = kept
		if c.Mode == Mixed && !c.Cache {
			for a := range resident {
				if !inUse(a) {
					delete(resident, a)
					poolUsed -= pages(a)
				}
			}
		}
	}
	if lastFinish > 0 {
		s.Throughput = float64(s.Requests) / (lastFinish / 1000)
	}
	s.TTFTP50, s.TTFTP99 = percentile(ttfts, 0.5), percentile(ttfts, 0.99)
	if iters > 0 {
		s.AvgBatch = float64(batchSum) / float64(iters)
		s.AdapterMem = memSum / float64(iters)
		s.Resident = resSum / float64(iters)
		s.Distinct = float64(distSum) / float64(iters)
	}
	return s
}

// MergedCopies cuántas copias completas del modelo, una por adaptador ya
// fusionado, entran en la memoria de una GPU: la alternativa ingenua.
func MergedCopies(gpuGB, modelGB float64) int { return int(gpuGB / modelGB) }
