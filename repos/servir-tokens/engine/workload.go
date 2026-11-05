package engine

import (
	"math"
	"math/rand/v2"
	"sort"

	"github.com/fiubaTA050/servir-tokens/prefix"
)

// Workload genera n pedidos con llegadas de Poisson (rate pedidos por segundo)
// y largos con cola pesada: la mayoría de las respuestas son cortas y unas
// pocas son muy largas, como en ShareGPT. Es determinista para cada seed.
func Workload(n int, rate float64, seed uint64, maxLen int) []Request {
	rng := rand.New(rand.NewPCG(seed, 7))
	logNormal := func(mu, sigma float64) int {
		return max(1, int(math.Exp(mu+sigma*rng.NormFloat64())))
	}
	reqs := make([]Request, n)
	t := 0.0
	for i := range reqs {
		t += rng.ExpFloat64() / rate * 1000
		prompt := min(logNormal(5.0, 0.8), maxLen/2)
		output := min(logNormal(4.8, 1.0), maxLen-prompt)
		reqs[i] = Request{ID: i, Arrival: t, Prompt: prompt, Output: output}
	}
	return reqs
}

// Chat describe una carga de conversaciones de varios turnos, la forma de la
// carga de un chatbot: cada turno manda la conversación entera otra vez, así
// que los prompts son largos, crecen turno a turno y casi todo es un prefijo
// que ya se vio. En la traza de Kimi el prompt promedio tiene 8.590 tokens y
// la respuesta 182 (Mooncake, §4).
type Chat struct {
	Sessions     int     // conversaciones
	Rate         float64 // conversaciones nuevas por segundo
	Turns        float64 // turnos por conversación, en promedio
	ThinkMs      float64 // tiempo entre turnos de una conversación, en promedio
	SystemPrompt int     // tokens del prompt de sistema, igual para todas las conversaciones de una app
	Apps         int     // cuántos prompts de sistema distintos hay
	UserTokens   float64 // tokens de cada mensaje del usuario, en promedio
	AnswerTokens float64 // tokens de cada respuesta, en promedio
	Block        int     // tokens por bloque de Request.Prefix
	MaxLen       int
	Seed         uint64
}

// DefaultChat es una carga parecida a la de un chatbot.
func DefaultChat() Chat {
	return Chat{Sessions: 300, Rate: 3, Turns: 5, ThinkMs: 15000, SystemPrompt: 2000, Apps: 4,
		UserTokens: 300, AnswerTokens: 180, Block: 64, MaxLen: 32768, Seed: 1}
}

// Requests genera los pedidos de la carga, ordenados por llegada.
func (c Chat) Requests() []Request {
	rng := rand.New(rand.NewPCG(c.Seed, 11))
	logNormal := func(mean float64) int {
		// sigma 0.6: cola pesada, pero no tanto como en Workload.
		const sigma = 0.6
		mu := math.Log(mean) - sigma*sigma/2
		return max(1, int(math.Exp(mu+sigma*rng.NormFloat64())))
	}
	var reqs []Request
	start := 0.0
	for s := range c.Sessions {
		start += rng.ExpFloat64() / c.Rate * 1000
		app := rng.IntN(max(1, c.Apps))
		turns := 1 + int(rng.ExpFloat64()*(c.Turns-1))
		history, t := c.SystemPrompt, start
		for range turns {
			prompt := history + logNormal(c.UserTokens)
			output := logNormal(c.AnswerTokens)
			if prompt+output > c.MaxLen {
				break
			}
			reqs = append(reqs, Request{Arrival: t, Prompt: prompt, Output: output, Session: s,
				Prefix: c.hashes(app, s, prompt)})
			history = prompt + output
			t += rng.ExpFloat64() * c.ThinkMs
		}
	}
	sort.SliceStable(reqs, func(i, j int) bool { return reqs[i].Arrival < reqs[j].Arrival })
	for i := range reqs {
		reqs[i].ID = i
	}
	return reqs
}

// hashes son los hashes encadenados de los bloques completos de un prompt.
// Un bloque que cae entero dentro del prompt de sistema es igual para toda la
// app; desde ahí, el contenido es de la conversación. Como cada turno agrega
// al final, el prompt de un turno es prefijo del siguiente.
func (c Chat) hashes(app, session, prompt int) []uint64 {
	n := prompt / c.Block
	hs := make([]uint64, n)
	prev := uint64(0)
	for i := range hs {
		owner := uint64(1)<<62 | uint64(app)<<32 // contenido de la app
		if (i+1)*c.Block > c.SystemPrompt {
			owner = uint64(session) << 32 // contenido de la conversación
		}
		prev = prefix.Hash(prev, owner|uint64(i))
		hs[i] = prev
	}
	return hs
}

// Run corre todos los pedidos hasta terminar y devuelve el resumen.
func Run(e *Engine, reqs []Request) (Summary, error) {
	for _, r := range reqs {
		if err := e.Submit(r); err != nil {
			return Summary{}, err
		}
	}
	for {
		if _, idle := e.Step(); idle {
			break
		}
	}
	return Summarize(e.Results(), e.Stats()), nil
}

// Summary son las métricas que importan a quien usa el servicio y a quien lo paga.
type Summary struct {
	Requests    int
	Throughput  float64 // tokens generados por segundo
	TTFTMean    float64 // tiempo al primer token visible, ms
	TTFTP99     float64
	LatencyMean float64 // tiempo hasta la respuesta completa, ms
	TPOTMean    float64 // tiempo entre tokens una vez que arrancó, ms
	Engine      Stats
}

// Summarize calcula las métricas de un conjunto de pedidos terminados.
func Summarize(rs []Result, st Stats) Summary {
	s := Summary{Requests: len(rs), Engine: st}
	if len(rs) == 0 {
		return s
	}
	var tokens int
	first, last := math.Inf(1), 0.0
	ttft := make([]float64, 0, len(rs))
	var tpotSum float64
	var tpotN int
	for _, r := range rs {
		tokens += r.Output
		first = min(first, r.Arrival)
		last = max(last, r.Done)
		ttft = append(ttft, r.FirstToken-r.Arrival)
		s.LatencyMean += r.Done - r.Arrival
		if r.Output > 1 && r.Done > r.FirstToken {
			tpotSum += (r.Done - r.FirstToken) / float64(r.Output-1)
			tpotN++
		}
	}
	n := float64(len(rs))
	s.Throughput = float64(tokens) / ((last - first) / 1000)
	s.LatencyMean /= n
	sort.Float64s(ttft)
	for _, v := range ttft {
		s.TTFTMean += v
	}
	s.TTFTMean /= n
	s.TTFTP99 = ttft[min(len(ttft)-1, int(0.99*n))]
	if tpotN > 0 {
		s.TPOTMean = tpotSum / float64(tpotN)
	}
	return s
}
