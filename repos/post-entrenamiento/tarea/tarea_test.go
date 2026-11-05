package tarea

import (
	"math"
	"math/rand"
	"testing"
)

// GRPO: las ventajas de un grupo suman cero, y un grupo parejo no da señal.
func TestVentajas(t *testing.T) {
	a := Ventajas([]float64{1, 0, 0, 1, 1, 0, 0, 0})
	s := 0.0
	for _, x := range a {
		s += x
	}
	if math.Abs(s) > 1e-9 || a[0] <= 0 || a[1] >= 0 {
		t.Errorf("ventajas %v", a)
	}
	for _, x := range Ventajas([]float64{1, 1, 1, 1}) {
		if x != 0 {
			t.Fatal("un grupo donde todos aciertan no debería dar señal")
		}
	}
}

// El óptimo está en el medio: pensar bastante sin chocar contra el tope.
func TestOptimoInterior(t *testing.T) {
	ta := Default()
	p, r := ta.Optimo()
	if p.Mediana() < 1000 || p.Mediana() > float64(ta.MaxLargo) {
		t.Errorf("mediana óptima %.0f", p.Mediana())
	}
	if ta.Cortadas(p) <= 0 || ta.Cortadas(p) > 0.2 {
		t.Errorf("cortadas en el óptimo: %.3f", ta.Cortadas(p))
	}
	if r <= ta.Recompensa(Inicial())+0.3 {
		t.Errorf("recompensa óptima %.3f vs inicial %.3f", r, ta.Recompensa(Inicial()))
	}
}

// Con muestras de la política actual, GRPO aprende: la recompensa sube cerca
// del óptimo y las respuestas se alargan (como R1-Zero).
func TestAprendeYAlarga(t *testing.T) {
	ta := Default()
	rng := rand.New(rand.NewSource(1))
	p := Inicial()
	l0 := ta.LargoMedio(p)
	for range 40 {
		var gs [][]Respuesta
		for range 32 {
			var g []Respuesta
			for range 8 {
				g = append(g, ta.Muestrear(rng, p))
			}
			gs = append(gs, g)
		}
		p.Actualizar(ta, gs, 0.6, false, 2)
	}
	_, best := ta.Optimo()
	if r := ta.Recompensa(p); r < best-0.03 {
		t.Errorf("recompensa %.3f, óptimo %.3f", r, best)
	}
	if ta.LargoMedio(p) < 5*l0 {
		t.Errorf("largo medio %.0f → %.0f: debería crecer", l0, ta.LargoMedio(p))
	}
}

// La corrección por importancia baja el peso de lo que la política actual ya
// no generaría.
func TestCorreccion(t *testing.T) {
	ta := Default()
	vieja := Respuesta{X: math.Log(300), MuBehav: math.Log(300), Recompensa: 1}
	otra := Respuesta{X: math.Log(400), MuBehav: math.Log(300)}
	p := Politica{Mu: math.Log(3000)}
	q := p
	p.Actualizar(ta, [][]Respuesta{{vieja, otra}}, 1, false, 2)
	q.Actualizar(ta, [][]Respuesta{{vieja, otra}}, 1, true, 2)
	if math.Abs(q.Mu-math.Log(3000)) >= math.Abs(p.Mu-math.Log(3000)) {
		t.Errorf("con corrección el paso debería ser más chico: %.4f vs %.4f", q.Mu, p.Mu)
	}
}
