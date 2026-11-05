package bucle

import (
	"math/rand"
	"testing"

	"github.com/fiubaTA050/post-entrenamiento/tarea"
)

var ta = tarea.Default()

// promedio de la recompensa final sobre varias semillas
func final(c Config, n int) (r, tiempo float64) {
	for s := 1; s <= n; s++ {
		c.Semilla = int64(s)
		x := Correr(c, ta)
		r += x.Final() / float64(n)
		tiempo += x.Tiempo / float64(n)
	}
	return
}

func separado(eta int) Config {
	c := Default()
	c.Ubicacion, c.Eta, c.Interrumpible, c.Corregir = Separado, eta, true, true
	return c
}

// Un paso sincrónico dura lo que la respuesta más larga, y la mayor parte del
// GPU·tiempo de generación se pierde esperándola.
func TestSincronicoEsperaALaMasLarga(t *testing.T) {
	c := Default()
	rng := rand.New(rand.NewSource(1))
	p, _ := ta.Optimo()
	var ls []int
	maxL := 0
	for range c.Problemas * c.Grupo {
		l := ta.Muestrear(rng, p).Largo
		ls = append(ls, l)
		maxL = max(maxL, l)
	}
	fin, util := Generar(c, ls)
	ultima, primera := 0.0, 1e18
	for _, f := range fin {
		ultima, primera = max(ultima, f), min(primera, f)
	}
	if ultima < float64(maxL)*c.Base/1000 {
		t.Errorf("la última GPU terminó a los %.0f s, antes de poder generar %d tokens", ultima, maxL)
	}
	if util > 0.5 {
		t.Errorf("útil %.0f%%: la cola larga debería dejar ociosa a la mayor parte", 100*util)
	}
}

// Con η = 0 el sistema asincrónico es sincrónico (AReaL §5.1): nada viejo.
func TestEtaCeroEsSincronico(t *testing.T) {
	r := Correr(separado(0), ta)
	if r.StaleMax != 0 {
		t.Errorf("staleness máxima %d con η = 0", r.StaleMax)
	}
}

// Más η, más staleness, más throughput y menos GPU ociosa.
func TestEtaCambiaThroughputYStaleness(t *testing.T) {
	r0, r1, r4 := Correr(separado(0), ta), Correr(separado(1), ta), Correr(separado(4), ta)
	if !(r0.Stale < r1.Stale && r1.Stale < r4.Stale) || r1.Stale > 1 {
		t.Errorf("staleness: %.2f %.2f %.2f", r0.Stale, r1.Stale, r4.Stale)
	}
	if !(r0.MuestrasPorSeg() < r1.MuestrasPorSeg() && r1.MuestrasPorSeg() < r4.MuestrasPorSeg()) {
		t.Errorf("throughput: %.2f %.2f %.2f", r0.MuestrasPorSeg(), r1.MuestrasPorSeg(), r4.MuestrasPorSeg())
	}
	if r4.Ocioso > r0.Ocioso/2 {
		t.Errorf("ociosa: sincrónico %.0f%%, η = 4 %.0f%%", 100*r0.Ocioso, 100*r4.Ocioso)
	}
}

// Con las mismas 8 GPUs, el asincrónico termina los mismos pasos en mucho
// menos tiempo que el colocado sincrónico, y llega a la misma recompensa.
func TestAsincronicoMasRapidoMismaRecompensa(t *testing.T) {
	rs, ts := final(Default(), 4)
	ra, tas := final(separado(4), 4)
	if tas > ts/2 {
		t.Errorf("tiempo: colocado %.0f s, asincrónico %.0f s", ts, tas)
	}
	if ra < rs-0.02 {
		t.Errorf("recompensa: colocado %.3f, asincrónico %.3f", rs, ra)
	}
}

// Sin corrección, entrenar con muestras de varias versiones atrás arruina el
// aprendizaje; con corrección por importancia, se recupera.
func TestStalenessSinCorreccion(t *testing.T) {
	sync, _ := final(separado(0), 6)
	c := separado(8)
	c.Corregir = false
	sin, _ := final(c, 6)
	con, _ := final(separado(8), 6)
	if sin > sync-0.2 {
		t.Errorf("sin corrección η = 8: %.3f vs sincrónico %.3f", sin, sync)
	}
	if con < sync-0.02 {
		t.Errorf("con corrección η = 8: %.3f vs sincrónico %.3f", con, sync)
	}
	c.Eta = 1
	if uno, _ := final(c, 6); uno < sync-0.02 {
		t.Errorf("η = 1 sin corrección debería alcanzar: %.3f vs %.3f", uno, sync)
	}
}

// Rollouts parciales (Kimi): el paso no espera a la cola, a cambio de algo de staleness.
func TestParciales(t *testing.T) {
	c := Default()
	s := Correr(c, ta)
	c.Parciales = true
	p := Correr(c, ta)
	if p.Tiempo > 0.75*s.Tiempo {
		t.Errorf("parciales %.0f s vs sincrónico %.0f s", p.Tiempo, s.Tiempo)
	}
	if p.Stale == 0 || p.Final() < s.Final()-0.05 {
		t.Errorf("parciales: staleness %.2f, recompensa %.3f vs %.3f", p.Stale, p.Final(), s.Final())
	}
}

// Rollouts interrumpibles (AReaL): no hay que esperar a que un worker
// termine lo que tiene para cargar los pesos nuevos.
func TestInterrumpible(t *testing.T) {
	c := separado(4)
	c.Interrumpible = false
	_, drena := final(c, 3)
	_, inter := final(separado(4), 3)
	if inter >= drena {
		t.Errorf("interrumpible %.0f s vs esperar %.0f s", inter, drena)
	}
}

// Las respuestas se alargan mientras aprende, y con ellas el rollout.
func TestRespuestasSeAlargan(t *testing.T) {
	r := Correr(Default(), ta)
	if r.Curva[len(r.Curva)-1].Largo < 5*r.Curva[0].Largo {
		t.Errorf("largo %.0f → %.0f", r.Curva[0].Largo, r.Curva[len(r.Curva)-1].Largo)
	}
}

// Todo lo generado y entrenado cierra: cada paso consume B·G respuestas.
func TestCuentas(t *testing.T) {
	for _, c := range []Config{Default(), separado(2), separado(-1)} {
		r := Correr(c, ta)
		if r.Muestras != c.Pasos*c.Problemas*c.Grupo || len(r.Curva) != c.Pasos+1 {
			t.Errorf("muestras %d, curva %d", r.Muestras, len(r.Curva))
		}
		if r.Generados < r.Tokens || r.Ocioso < 0 || r.Ocioso > 1 {
			t.Errorf("generados %d, entrenados %d, ociosa %.2f", r.Generados, r.Tokens, r.Ocioso)
		}
	}
}
