// Package tarea es el problema de juguete que aprende el bucle de RL: un
// "modelo" que decide cuánto pensar antes de contestar. Pensar más (una
// respuesta más larga) acierta más seguido, pero la respuesta tiene un tope de
// tokens: si lo pasa, se corta y el verificador le da 0. La recompensa es 1 o
// 0, como en el RL con recompensas verificables de DeepSeek-R1.
//
// La política es una lognormal sobre el largo: log(largo) ~ N(Mu, Sigma²). Es
// la cola larga de los rollouts, y aprender la corre hacia la derecha: las
// respuestas se alargan durante el entrenamiento, como en R1-Zero. Se entrena
// con GRPO: la ventaja de cada respuesta se mide contra las otras respuestas
// del mismo grupo, sin crítico.
//
// El óptimo está en el medio: pensar bastante sin chocar contra el tope. Por
// eso entrenar con muestras viejas se nota: un gradiente calculado con la
// política de hace varios pasos sigue empujando cuando ya se pasó del óptimo.
package tarea

import (
	"math"
	"math/rand"
)

// Tarea define la recompensa y el tope.
type Tarea struct {
	Escala   float64 // P(acierto | largo L) = 1 − exp(−L/Escala)
	MaxLargo int     // las respuestas se cortan ahí, y una respuesta cortada vale 0
	Sigma    float64 // dispersión de la política: el ancho de la cola larga
}

// Default: con 300 tokens se acierta 19% de las veces, con 2.000 el 75%, con
// 6.000 el 98%; el tope es 8.192.
func Default() Tarea { return Tarea{Escala: 1400, MaxLargo: 8192, Sigma: 0.6} }

// Politica: log(largo) ~ N(Mu, Sigma²).
type Politica struct{ Mu float64 }

// Inicial: una política que piensa poco (mediana de 300 tokens).
func Inicial() Politica { return Politica{Mu: math.Log(300)} }

// Mediana del largo, en tokens.
func (p Politica) Mediana() float64 { return math.Exp(p.Mu) }

// Respuesta es una muestra del rollout.
type Respuesta struct {
	X          float64 // log del largo que eligió la política (antes del corte)
	Largo      int     // tokens que hubo que generar
	Recompensa float64
	MuBehav    float64 // la política que la generó
}

// Muestrear genera una respuesta con la política p.
func (t Tarea) Muestrear(rng *rand.Rand, p Politica) Respuesta {
	x := p.Mu + t.Sigma*rng.NormFloat64()
	l := max(int(math.Round(math.Exp(x))), 1)
	r := Respuesta{X: x, Largo: l, MuBehav: p.Mu}
	if l > t.MaxLargo {
		r.Largo = t.MaxLargo // cortada: se generó hasta el tope y no hay respuesta
	} else if rng.Float64() < 1-math.Exp(-float64(l)/t.Escala) {
		r.Recompensa = 1
	}
	return r
}

// Recompensa esperada de una política (integral numérica, sin muestrear).
func (t Tarea) Recompensa(p Politica) float64 {
	const n = 400
	s, w := 0.0, 0.0
	for i := range n {
		z := -5 + 10*(float64(i)+0.5)/n
		d := math.Exp(-z * z / 2)
		l := math.Exp(p.Mu + t.Sigma*z)
		w += d
		if l <= float64(t.MaxLargo) {
			s += d * (1 - math.Exp(-l/t.Escala))
		}
	}
	return s / w
}

// Cortadas es la fracción de respuestas que pasan el tope.
func (t Tarea) Cortadas(p Politica) float64 {
	z := (math.Log(float64(t.MaxLargo)) - p.Mu) / t.Sigma
	return 0.5 * math.Erfc(z/math.Sqrt2)
}

// LargoMedio esperado de lo que se genera (contando el corte en el tope).
func (t Tarea) LargoMedio(p Politica) float64 {
	const n = 400
	s, w := 0.0, 0.0
	for i := range n {
		z := -5 + 10*(float64(i)+0.5)/n
		d := math.Exp(-z * z / 2)
		s += d * math.Min(math.Exp(p.Mu+t.Sigma*z), float64(t.MaxLargo))
		w += d
	}
	return s / w
}

// Optimo devuelve la mejor política (búsqueda en una grilla) y su recompensa.
func (t Tarea) Optimo() (Politica, float64) {
	best, br := Politica{}, -1.0
	for mu := math.Log(100); mu < math.Log(float64(t.MaxLargo)); mu += 0.005 {
		if r := t.Recompensa(Politica{mu}); r > br {
			best, br = Politica{mu}, r
		}
	}
	return best, br
}

// Ventajas de GRPO: la recompensa de cada respuesta menos la media de su
// grupo, dividido por el desvío. Si todo el grupo sacó lo mismo, no hay señal.
func Ventajas(r []float64) []float64 {
	n := float64(len(r))
	m := 0.0
	for _, x := range r {
		m += x
	}
	m /= n
	v := 0.0
	for _, x := range r {
		v += (x - m) * (x - m)
	}
	sd := math.Sqrt(v / n)
	out := make([]float64, len(r))
	if sd < 1e-9 {
		return out
	}
	for i, x := range r {
		out[i] = (x - m) / sd
	}
	return out
}

// Actualizar da un paso de gradiente de política con GRPO sobre grupos de
// respuestas: Mu += lr · promedio(peso · ventaja · ∇log π(x)), con
// ∇log π(x) = (x − Mu)/Sigma².
//
// Sin corrección (peso 1), una muestra que generó una política vieja se usa
// como si fuera de la actual: el gradiente que sale es, en promedio, el de la
// política vieja. Con corrección, cada muestra pesa π_actual(x)/π_behav(x),
// truncado en maxPeso: es la idea del objetivo desacoplado de AReaL.
func (p *Politica) Actualizar(t Tarea, grupos [][]Respuesta, lr float64, corregir bool, maxPeso float64) {
	s2 := t.Sigma * t.Sigma
	g, n := 0.0, 0
	for _, gr := range grupos {
		r := make([]float64, len(gr))
		for i, x := range gr {
			r[i] = x.Recompensa
		}
		A := Ventajas(r)
		for i, x := range gr {
			n++
			w := 1.0
			if corregir {
				lw := (-(x.X-p.Mu)*(x.X-p.Mu) + (x.X-x.MuBehav)*(x.X-x.MuBehav)) / (2 * s2)
				w = math.Min(math.Exp(lw), maxPeso)
			}
			g += w * A[i] * (x.X - p.Mu) / s2
		}
	}
	if n > 0 {
		p.Mu += lr * g / float64(n)
	}
}
