// Package bucle simula, en tiempo virtual, el bucle de post-entrenamiento con
// RL de un modelo de lenguaje: generar respuestas (una carga de serving, Clase
// 18), puntuarlas con un verificador y actualizar los pesos (una carga de
// entrenamiento, Clase 16). Lo que se simula es lo que importa para la
// materia: quién espera a quién, cuánto cuesta llevar los pesos nuevos a
// donde se genera, y con qué pesos se generó cada muestra que se entrena.
//
// Cada GPU de rollout es un motor con batching continuo: una iteración de
// decode cuesta Base + PorSeq·secuencias milisegundos y produce un token para
// cada secuencia del lote. Entrenar cuesta Entrenar ms·GPU por token.
//
// Dos ubicaciones:
//
//   - Colocado (verl / HybridFlow, Kimi k1.5): las mismas GPUs generan y
//     después entrenan, por turnos. Entre una cosa y la otra, la transición
//     de layout. Con rollouts parciales (Kimi), la generación termina cuando
//     hay un lote completo y lo que quedó a medias sigue en la vuelta
//     siguiente.
//   - Separado (OpenRLHF, AReaL): unas GPUs generan y otras entrenan; después
//     de cada paso, el trainer manda los pesos a las de rollout. Con Eta = 0 es
//     sincrónico; con Eta > 0 los rollouts siguen generando con pesos de hasta
//     Eta versiones atrás (AReaL, ecuación 3); con Eta < 0, sin límite.
package bucle

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/fiubaTA050/post-entrenamiento/pesos"
	"github.com/fiubaTA050/post-entrenamiento/tarea"
)

// Ubicacion de los dos trabajos.
type Ubicacion int

const (
	Colocado Ubicacion = iota
	Separado
)

// Config de una corrida.
type Config struct {
	Ubicacion Ubicacion
	GPUs      int // en total
	Rollout   int // Separado: cuántas generan; el resto entrena
	Slots     int // secuencias por GPU de rollout (el lote de la Clase 18)

	Base, PorSeq float64 // ms por iteración de decode
	Prefill      float64 // ms por token para recalcular KV cache al retomar una secuencia
	Entrenar     float64 // ms·GPU por token entrenado

	PesosGB    float64 // tamaño de los pesos
	RedGBs     float64 // GB/s entre nodos
	Anillo     bool    // Separado: broadcast en anillo/pipeline en vez de ingenuo
	Transicion float64 // Colocado: segundos para pasar al layout de generación

	Eta           int  // Separado: staleness máxima (0 sincrónico, <0 sin límite)
	Interrumpible bool // Separado asincrónico: los workers cambian de pesos sin terminar lo que tienen (AReaL)
	Parciales     bool // Colocado: rollouts parciales (Kimi k1.5)

	Problemas, Grupo int // B problemas por paso, G respuestas por problema
	Pasos            int
	LR               float64
	Corregir         bool    // corrección por importancia de las muestras viejas
	MaxPeso          float64 // tope del peso de importancia
	Semilla          int64
	Politica         tarea.Politica // de dónde arranca; cero: tarea.Inicial()
}

// Default: 8 GPUs, 32 problemas × 8 respuestas por paso. Los costos son
// ilustrativos; lo que importa es la forma.
func Default() Config {
	return Config{
		GPUs: 8, Rollout: 6, Slots: 64,
		Base: 20, PorSeq: 0.2, Prefill: 0.05, Entrenar: 0.2,
		PesosGB: 14, RedGBs: 25, Anillo: true, Transicion: 2,
		Problemas: 32, Grupo: 8, Pasos: 40, LR: 0.6, MaxPeso: 2, Semilla: 1,
	}
}

// Punto de la curva de aprendizaje, después de cada paso.
type Punto struct {
	Paso       int
	T          float64 // segundos
	Recompensa float64 // esperada, de la política después del paso
	Largo      float64 // largo medio esperado de las respuestas
}

// Resultado de una corrida.
type Resultado struct {
	Tiempo     float64 // segundos hasta el último paso
	Muestras   int     // respuestas entrenadas
	Tokens     int     // tokens entrenados
	Generados  int     // tokens generados en total (incluye los que no llegaron a entrenarse)
	Ocioso     float64 // fracción de GPU·tiempo sin trabajo útil
	OciosoGen  float64 // Separado: de las GPUs de rollout
	OciosoEntr float64 // Separado: de las GPUs de entrenamiento
	FracGen    float64 // Colocado: fracción del tiempo generando
	FracEntr   float64 // Colocado: entrenando
	FracSync   float64 // fracción del tiempo moviendo pesos o recalculando KV cache
	Stale      float64 // staleness promedio de lo que se entrenó (versiones)
	StaleMax   int
	Curva      []Punto
}

// MuestrasPorSeg entrenadas.
func (r Resultado) MuestrasPorSeg() float64 { return float64(r.Muestras) / r.Tiempo }

// TokensPorSeg entrenados.
func (r Resultado) TokensPorSeg() float64 { return float64(r.Tokens) / r.Tiempo }

// Final es la recompensa esperada al terminar.
func (r Resultado) Final() float64 { return r.Curva[len(r.Curva)-1].Recompensa }

// TiempoHasta devuelve los segundos hasta que la recompensa llegó a x, o +Inf.
func (r Resultado) TiempoHasta(x float64) float64 {
	for _, p := range r.Curva {
		if p.Recompensa >= x {
			return p.T
		}
	}
	return math.Inf(1)
}

// PasoHasta devuelve el primer paso en que la recompensa llegó a x, o -1.
func (r Resultado) PasoHasta(x float64) int {
	for _, p := range r.Curva {
		if p.Recompensa >= x {
			return p.Paso
		}
	}
	return -1
}

// ---------------------------------------------------------------------------

type muestra struct {
	g      *grupo
	r      tarea.Respuesta
	hechos int
	ver    int // versión de los pesos con que empezó
}

type grupo struct {
	idx, ver int
	ms       []*muestra
	faltan   int
}

type worker struct {
	id      int
	activas []*muestra
	ver     int
	t0      float64 // inicio del segmento de iteraciones en curso
	costo   float64 // ms por iteración del segmento
	epoca   int
	pausa   float64 // no decodifica hasta este momento (cargar pesos, recalcular KV)
	drenar  bool    // tiene pesos nuevos pendientes: termina lo que tiene y no empieza nada
	util    float64 // ms·GPU de decode útil (pesado por ocupación del lote)
}

type evento struct {
	t     float64
	seq   int
	tipo  int
	w     *worker
	epoca int
}

const (
	evWorker = iota
	evPublicar
	evReanudar
)

type cola []evento

func (c cola) Len() int { return len(c) }
func (c cola) Less(i, j int) bool {
	if c[i].t != c[j].t {
		return c[i].t < c[j].t
	}
	return c[i].seq < c[j].seq
}
func (c cola) Swap(i, j int) { c[i], c[j] = c[j], c[i] }
func (c *cola) Push(x any)   { *c = append(*c, x.(evento)) }
func (c *cola) Pop() any     { o := *c; e := o[len(o)-1]; *c = o[:len(o)-1]; return e }

type sim struct {
	c   Config
	t   tarea.Tarea
	rng *rand.Rand

	now  float64
	ev   cola
	seq  int
	ws   []*worker
	pend []*muestra // muestras de grupos ya empezados que esperan un lugar

	pol      tarea.Politica
	pols     []tarea.Politica // por versión publicada
	ver      int              // última versión publicada
	empez    int              // grupos empezados
	consum   int              // grupos consumidos
	listos   []*grupo
	ocupado  bool    // el trainer (o, colocado, todo el clúster) está entrenando
	generar  bool    // Colocado: fase de generación
	syncMS   float64 // tiempo de reloj moviendo pesos (o recalculando)
	entrMS   float64 // GPU·ms entrenando
	entrReal float64 // tiempo de reloj entrenando (colocado)

	res      Resultado
	staleSum float64
}

// Correr simula Pasos actualizaciones con la configuración c sobre la tarea t.
func Correr(c Config, t tarea.Tarea) Resultado {
	if c.Ubicacion == Colocado {
		c.Rollout = c.GPUs
	}
	if c.Ubicacion == Separado && (c.Rollout <= 0 || c.Rollout >= c.GPUs) {
		panic(fmt.Sprintf("bucle: Rollout=%d con %d GPUs", c.Rollout, c.GPUs))
	}
	s := &sim{c: c, t: t, rng: rand.New(rand.NewSource(c.Semilla)), pol: c.Politica, generar: true}
	if s.pol.Mu == 0 {
		s.pol = tarea.Inicial()
	}
	s.pols = []tarea.Politica{s.pol}
	s.res.Curva = []Punto{{0, 0, t.Recompensa(s.pol), t.LargoMedio(s.pol)}}
	for i := range c.Rollout {
		s.ws = append(s.ws, &worker{id: i})
	}
	for _, w := range s.ws {
		s.llenar(w)
	}
	for s.ev.Len() > 0 && s.ver < c.Pasos {
		e := heap.Pop(&s.ev).(evento)
		s.now = e.t
		switch e.tipo {
		case evWorker:
			if e.epoca == e.w.epoca {
				s.llenar(e.w)
			}
		case evPublicar:
			s.publicar()
		case evReanudar:
			s.reanudar()
		}
	}
	if s.ver < c.Pasos {
		panic("bucle: la simulación se trabó")
	}
	s.cerrar()
	return s.res
}

func (s *sim) programar(t float64, tipo int, w *worker) {
	s.seq++
	e := evento{t: t, seq: s.seq, tipo: tipo, w: w}
	if w != nil {
		w.epoca++
		e.epoca = w.epoca
	}
	heap.Push(&s.ev, e)
}

func (s *sim) costo(n int) float64 { return s.c.Base + s.c.PorSeq*float64(n) }

// avanzar aplica las iteraciones completas que el worker hizo hasta ahora.
func (s *sim) avanzar(w *worker) {
	if len(w.activas) == 0 || s.now <= w.t0 {
		w.t0 = math.Max(w.t0, s.now)
		return
	}
	it := int((s.now-w.t0)/w.costo + 1e-9)
	for _, m := range w.activas {
		it = min(it, m.r.Largo-m.hechos)
	}
	for _, m := range w.activas {
		m.hechos += it
	}
	s.res.Generados += it * len(w.activas)
	w.util += float64(it) * w.costo * float64(len(w.activas)) / float64(s.c.Slots)
	w.t0 = s.now
	// las que terminaron dejan su lugar
	quedan := w.activas[:0]
	for _, m := range w.activas {
		if m.hechos >= m.r.Largo {
			m.g.faltan--
			if m.g.faltan == 0 {
				s.listos = append(s.listos, m.g)
			}
		} else {
			quedan = append(quedan, m)
		}
	}
	w.activas = quedan
}

// puedeEmpezar dice si el controlador deja empezar un grupo nuevo.
func (s *sim) puedeEmpezar(w *worker) bool {
	B := s.c.Problemas
	if s.c.Ubicacion == Colocado {
		if !s.generar {
			return false
		}
		if s.c.Parciales {
			return s.empez-s.consum < 2*B
		}
		return s.empez/B <= s.ver
	}
	if w.ver != s.ver || w.drenar {
		return false
	}
	if s.c.Eta < 0 {
		return true
	}
	return s.empez/B <= s.ver+s.c.Eta // AReaL, ecuación 3
}

func (s *sim) empezarGrupo() {
	g := &grupo{idx: s.empez, ver: s.ver, faltan: s.c.Grupo}
	s.empez++
	for range s.c.Grupo {
		m := &muestra{g: g, r: s.t.Muestrear(s.rng, s.pols[s.ver]), ver: s.ver}
		g.ms = append(g.ms, m)
		s.pend = append(s.pend, m)
	}
}

// llenar avanza el worker hasta ahora, mete secuencias nuevas si puede y
// programa su próximo evento.
func (s *sim) llenar(w *worker) {
	if s.now < w.pausa {
		s.programar(w.pausa, evWorker, w)
		return
	}
	pausado := s.c.Ubicacion == Colocado && !s.generar
	if !pausado {
		w.t0 = math.Max(w.t0, w.pausa)
		s.avanzar(w)
	}
	if w.drenar && len(w.activas) == 0 {
		w.drenar = false
		w.ver = s.ver
	}
	otros := false
	for !pausado && len(w.activas) < s.c.Slots {
		if len(s.pend) == 0 {
			if !s.puedeEmpezar(w) {
				break
			}
			s.empezarGrupo()
			otros = true
		}
		w.activas = append(w.activas, s.pend[0])
		s.pend = s.pend[1:]
	}
	s.intentarEntrenar()
	if !pausado && len(w.activas) > 0 {
		w.costo = s.costo(len(w.activas))
		w.t0 = s.now
		k := math.MaxInt
		for _, m := range w.activas {
			k = min(k, m.r.Largo-m.hechos)
		}
		s.programar(s.now+float64(k)*w.costo, evWorker, w)
	} else {
		w.epoca++ // ocioso: lo despierta otro evento
	}
	if otros && len(s.pend) > 0 {
		for _, o := range s.ws {
			if o != w && len(o.activas) < s.c.Slots && s.now >= o.pausa {
				s.llenar(o)
			}
		}
	}
}

// intentarEntrenar arranca un paso si hay un lote completo y el trainer está libre.
func (s *sim) intentarEntrenar() {
	B := s.c.Problemas
	if s.ocupado || len(s.listos) < B || s.ver >= s.c.Pasos {
		return
	}
	if s.c.Ubicacion == Colocado && !s.c.Parciales {
		// sincrónico: espera a que termine la última respuesta del lote
		for _, w := range s.ws {
			if len(w.activas) > 0 {
				return
			}
		}
	}
	// los B grupos más viejos (AReaL prioriza las trayectorias viejas)
	sort.Slice(s.listos, func(i, j int) bool { return s.listos[i].idx < s.listos[j].idx })
	lote := s.listos[:B]
	s.listos = append([]*grupo(nil), s.listos[B:]...)
	s.consum += B
	grupos := make([][]tarea.Respuesta, 0, B)
	tokens := 0
	for _, g := range lote {
		resp := make([]tarea.Respuesta, 0, len(g.ms))
		st := s.ver - g.ver
		for _, m := range g.ms {
			resp = append(resp, m.r)
			tokens += m.r.Largo
			s.staleSum += float64(st)
			s.res.StaleMax = max(s.res.StaleMax, st)
		}
		grupos = append(grupos, resp)
	}
	s.res.Muestras += B * s.c.Grupo
	s.res.Tokens += tokens
	s.pol.Actualizar(s.t, grupos, s.c.LR, s.c.Corregir, s.c.MaxPeso)
	s.ocupado = true

	gpusEntr := s.c.GPUs - s.c.Rollout
	if s.c.Ubicacion == Colocado {
		gpusEntr = s.c.GPUs
		s.generar = false
		for _, w := range s.ws {
			s.avanzar(w)
		}
	}
	entr := float64(tokens) * s.c.Entrenar / float64(gpusEntr)
	s.entrMS += entr * float64(gpusEntr)
	var sync float64
	if s.c.Ubicacion == Colocado {
		sync = s.c.Transicion * 1000
		s.entrReal += entr
	} else {
		sync = pesos.Broadcast(s.c.PesosGB, s.c.RedGBs, s.c.Rollout, s.c.Anillo) * 1000
	}
	s.syncMS += sync
	s.programar(s.now+entr+sync, evPublicar, nil)
}

// publicar: hay pesos nuevos donde se genera.
func (s *sim) publicar() {
	s.ver++
	s.pols = append(s.pols, s.pol)
	s.res.Curva = append(s.res.Curva, Punto{s.ver, s.now / 1000, s.t.Recompensa(s.pol), s.t.LargoMedio(s.pol)})
	s.ocupado = false
	if s.ver >= s.c.Pasos {
		return
	}
	if s.c.Ubicacion == Colocado {
		// Kimi: las respuestas a medias se retoman; hay que recalcular su KV cache.
		pausa := 0.0
		for _, w := range s.ws {
			n := 0
			for _, m := range w.activas {
				n += m.hechos
			}
			pausa = math.Max(pausa, float64(n)*s.c.Prefill)
		}
		s.syncMS += pausa
		s.programar(s.now+pausa, evReanudar, nil)
		return
	}
	for _, w := range s.ws {
		if s.now >= w.pausa {
			s.avanzar(w)
		}
		switch {
		case len(w.activas) == 0:
			w.ver = s.ver
		case s.c.Interrumpible:
			// AReaL: se interrumpe, se descarta el KV cache y se recalcula con los pesos nuevos
			n := 0
			for _, m := range w.activas {
				n += m.hechos
			}
			w.ver = s.ver
			w.pausa = math.Max(w.pausa, s.now) + float64(n)*s.c.Prefill
		default:
			w.drenar = true
		}
	}
	for _, w := range s.ws {
		s.llenar(w)
	}
	s.intentarEntrenar()
}

func (s *sim) reanudar() {
	s.generar = true
	for _, w := range s.ws {
		w.ver = s.ver
		w.t0 = s.now
	}
	for _, w := range s.ws {
		s.llenar(w)
	}
}

func (s *sim) cerrar() {
	T := s.now // ms
	r := &s.res
	r.Tiempo = T / 1000
	gen := 0.0
	for _, w := range s.ws {
		gen += w.util
	}
	total := float64(s.c.GPUs) * T
	r.Ocioso = 1 - (gen+s.entrMS)/total
	if s.c.Ubicacion == Separado {
		r.OciosoGen = 1 - gen/(float64(s.c.Rollout)*T)
		r.OciosoEntr = 1 - s.entrMS/(float64(s.c.GPUs-s.c.Rollout)*T)
		r.FracSync = s.syncMS / T
	} else {
		r.FracEntr = s.entrReal / T
		r.FracSync = s.syncMS / T
		r.FracGen = 1 - r.FracEntr - r.FracSync
	}
	if r.Muestras > 0 {
		r.Stale = s.staleSum / float64(r.Muestras)
	}
}

// Generar simula la fase de generación de un paso sincrónico con los largos
// dados, repartidos en ronda entre las GPUs de rollout (cada una con un lote
// continuo de hasta Slots secuencias). Devuelve cuándo termina cada GPU, en
// segundos, y la fracción de GPU·tiempo útil hasta que termina la última.
func Generar(c Config, largos []int) (fin []float64, util float64) {
	n := c.Rollout
	if c.Ubicacion == Colocado {
		n = c.GPUs
	}
	por := make([][]int, n)
	for i, l := range largos {
		por[i%n] = append(por[i%n], l)
	}
	usado := 0.0
	for _, ls := range por {
		sort.Ints(ls)
		t := 0.0
		// las primeras Slots entran ya; cada vez que termina una entra la siguiente
		activas := append([]int(nil), ls[:min(len(ls), c.Slots)]...)
		cola := ls[min(len(ls), c.Slots):]
		hechos := make([]int, len(activas))
		for len(activas) > 0 {
			k := math.MaxInt
			for i, l := range activas {
				k = min(k, l-hechos[i])
			}
			costo := c.Base + c.PorSeq*float64(len(activas))
			t += float64(k) * costo
			usado += float64(k) * costo * float64(len(activas)) / float64(c.Slots)
			var na []int
			var nh []int
			for i, l := range activas {
				if hechos[i]+k < l {
					na = append(na, l)
					nh = append(nh, hechos[i]+k)
				} else if len(cola) > 0 {
					na = append(na, cola[0])
					nh = append(nh, 0)
					cola = cola[1:]
				}
			}
			activas, hechos = na, nh
		}
		fin = append(fin, t/1000)
	}
	m := 0.0
	for _, f := range fin {
		m = math.Max(m, f)
	}
	return fin, usado / 1000 / (m * float64(n))
}
