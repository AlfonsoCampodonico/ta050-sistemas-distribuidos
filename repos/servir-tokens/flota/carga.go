package flota

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Request es un pedido a uno de los modelos de la flota.
type Request struct {
	ID      int
	Model   int     // a qué modelo (o adaptador) va
	Arrival float64 // segundos
	Prompt  int     // tokens de entrada
	Output  int     // tokens de salida
}

// Workload describe la carga de una plataforma con muchos modelos: pocos
// modelos populares y una cola larga de modelos que casi nadie usa, cada uno
// con llegadas en ráfagas. Es la forma de carga de las evaluaciones de
// ServerlessLLM y S-LoRA: llegadas Gamma con un coeficiente de variación CV
// (CV = 1 es Poisson; más alto es más en ráfagas) y tasas por modelo que
// siguen una ley de potencia.
type Workload struct {
	Models    int     // cuántos modelos (o adaptadores) distintos
	Rate      float64 // pedidos por segundo, sumando todos los modelos
	CV        float64 // coeficiente de variación de los tiempos entre llegadas
	Skew      float64 // exponente de la ley de potencia: el modelo i recibe ∝ 1/(i+1)^Skew
	Duration  float64 // segundos de carga
	PromptMin int
	PromptMax int
	OutputMin int
	OutputMax int
	Seed      uint64
}

// DefaultWorkload es la carga de las demos de arranque en frío: 64 modelos
// durante una hora, en ráfagas con CV = 4 (ServerlessLLM usa 8, §7.1) y
// largos uniformes entre 8 y 512 tokens (S-LoRA, tabla 2).
func DefaultWorkload() Workload {
	return Workload{Models: 64, Rate: 1, CV: 4, Skew: 1, Duration: 3600,
		PromptMin: 8, PromptMax: 512, OutputMin: 8, OutputMax: 512, Seed: 1}
}

// Weights devuelve la fracción del tráfico que recibe cada modelo.
func (w Workload) Weights() []float64 {
	ws := make([]float64, w.Models)
	total := 0.0
	for i := range ws {
		ws[i] = 1 / math.Pow(float64(i+1), w.Skew)
		total += ws[i]
	}
	for i := range ws {
		ws[i] /= total
	}
	return ws
}

// Requests genera los pedidos, ordenados por llegada. Es determinista para
// cada Seed.
func (w Workload) Requests() []Request {
	rng := rand.New(rand.NewPCG(w.Seed, 17))
	shape := 1 / (w.CV * w.CV)
	var reqs []Request
	for m, frac := range w.Weights() {
		rate := w.Rate * frac
		scale := 1 / (rate * shape) // media de cada intervalo: shape·scale = 1/rate
		// Arranca en un punto al azar del primer intervalo, para que no
		// lleguen todos los modelos juntos en t = 0.
		t := rng.Float64() * gamma(rng, shape) * scale
		for t < w.Duration {
			reqs = append(reqs, Request{
				Model:   m,
				Arrival: t,
				Prompt:  w.PromptMin + rng.IntN(w.PromptMax-w.PromptMin+1),
				Output:  w.OutputMin + rng.IntN(w.OutputMax-w.OutputMin+1),
			})
			t += gamma(rng, shape) * scale
		}
	}
	sort.SliceStable(reqs, func(i, j int) bool { return reqs[i].Arrival < reqs[j].Arrival })
	for i := range reqs {
		reqs[i].ID = i
	}
	return reqs
}

// gamma devuelve una muestra de una Gamma(shape, 1) (Marsaglia y Tsang, 2000).
func gamma(rng *rand.Rand, shape float64) float64 {
	if shape < 1 {
		// Gamma(k) = Gamma(k+1) · U^(1/k)
		return gamma(rng, shape+1) * math.Pow(1-rng.Float64(), 1/shape)
	}
	d := shape - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := 1 - rng.Float64()
		if math.Log(u) < 0.5*x*x+d-d*v+d*math.Log(v) {
			return d * v
		}
	}
}

func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[min(len(s)-1, int(p*float64(len(s))))]
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	t := 0.0
	for _, x := range xs {
		t += x
	}
	return t / float64(len(xs))
}
