// Package pipeline simula schedules de paralelismo de pipeline: el modelo se
// parte en p etapas (una por GPU), el batch en m microbatches, y cada etapa
// ejecuta forwards (F) y backwards (B) en un orden fijo. El simulador calcula
// cuándo arranca cada operación respetando dos restricciones:
//
//   - una etapa ejecuta una sola operación a la vez, en el orden del schedule;
//   - F(s,i) necesita F(s-1,i) y B(s,i) necesita B(s+1,i): los datos fluyen
//     hacia adelante en el forward y hacia atrás en el backward.
//
// Con eso se puede medir la burbuja (tiempo ocioso) y la memoria de
// activaciones de cada schedule, y comparar con las fórmulas del paper de
// Narayanan et al. (SC 2021), §2.2.
package pipeline

import (
	"fmt"
	"strings"
)

// Kind distingue forward de backward.
type Kind byte

const (
	Forward  Kind = 'F'
	Backward Kind = 'B'
)

// Op es una operación de una etapa sobre un microbatch.
type Op struct {
	Kind       Kind
	Microbatch int // 0..m-1
}

// Schedule devuelve, para cada etapa, el orden en que ejecuta sus operaciones.
type Schedule func(p, m int) [][]Op

// GPipe: todos los forwards primero, después todos los backwards.
func GPipe(p, m int) [][]Op {
	order := make([][]Op, p)
	for s := range p {
		for i := range m {
			order[s] = append(order[s], Op{Forward, i})
		}
		for i := range m {
			order[s] = append(order[s], Op{Backward, i})
		}
	}
	return order
}

// OneFOneB (PipeDream-Flush): cada etapa hace un warm-up de p-s-1 forwards y
// después alterna un forward y un backward. Limita los microbatches en vuelo
// a la profundidad del pipeline en lugar de m.
func OneFOneB(p, m int) [][]Op {
	order := make([][]Op, p)
	for s := range p {
		warmup := min(p-s-1, m)
		f, b := 0, 0
		for ; f < warmup; f++ {
			order[s] = append(order[s], Op{Forward, f})
		}
		for f < m {
			order[s] = append(order[s], Op{Forward, f})
			f++
			order[s] = append(order[s], Op{Backward, b})
			b++
		}
		for ; b < m; b++ {
			order[s] = append(order[s], Op{Backward, b})
		}
	}
	return order
}

// Slot es una operación ya ubicada en el tiempo.
type Slot struct {
	Op
	Start, End int
}

// Timeline es el resultado de simular un schedule.
type Timeline struct {
	P, M, TF, TB int
	Stages       [][]Slot
	Makespan     int
}

// Simulate ubica cada operación en el tiempo. tf y tb son las duraciones de
// un forward y un backward de un microbatch en una etapa.
func Simulate(sched Schedule, p, m, tf, tb int) Timeline {
	order := sched(p, m)
	doneF := make([][]int, p) // doneF[s][i] = fin de F(s,i), -1 si no corrió
	doneB := make([][]int, p)
	for s := range p {
		doneF[s] = filled(m, -1)
		doneB[s] = filled(m, -1)
	}
	next := make([]int, p) // próxima operación de cada etapa
	free := make([]int, p) // instante en que la etapa queda libre
	stages := make([][]Slot, p)

	for remaining := p * 2 * m; remaining > 0; {
		progressed := false
		for s := range p {
			if next[s] == len(order[s]) {
				continue
			}
			op := order[s][next[s]]
			ready, ok := 0, true
			switch op.Kind {
			case Forward:
				if s > 0 {
					ready, ok = doneF[s-1][op.Microbatch], doneF[s-1][op.Microbatch] >= 0
				}
			case Backward:
				if s < p-1 {
					ready, ok = doneB[s+1][op.Microbatch], doneB[s+1][op.Microbatch] >= 0
				} else {
					ready, ok = doneF[s][op.Microbatch], doneF[s][op.Microbatch] >= 0
				}
			}
			if !ok {
				continue
			}
			start := max(ready, free[s])
			dur := tf
			if op.Kind == Backward {
				dur = tb
			}
			end := start + dur
			stages[s] = append(stages[s], Slot{op, start, end})
			if op.Kind == Forward {
				doneF[s][op.Microbatch] = end
			} else {
				doneB[s][op.Microbatch] = end
			}
			free[s], next[s] = end, next[s]+1
			remaining--
			progressed = true
		}
		if !progressed {
			panic("pipeline: el schedule tiene un deadlock")
		}
	}

	t := Timeline{P: p, M: m, TF: tf, TB: tb, Stages: stages}
	for s := range p {
		t.Makespan = max(t.Makespan, free[s])
	}
	return t
}

// Ideal es el tiempo sin burbuja: m·(tf+tb).
func (t Timeline) Ideal() int { return t.M * (t.TF + t.TB) }

// BubbleFraction es el tiempo ocioso sobre el ideal. El paper predice
// (p-1)/m para GPipe y para 1F1B.
func (t Timeline) BubbleFraction() float64 {
	return float64(t.Makespan-t.Ideal()) / float64(t.Ideal())
}

// PeakInFlight devuelve, por etapa, la máxima cantidad de microbatches con
// el forward hecho y el backward pendiente: son las activaciones que la GPU
// tiene que guardar en memoria.
func (t Timeline) PeakInFlight() []int {
	peaks := make([]int, t.P)
	for s, slots := range t.Stages {
		inFlight := 0
		for _, sl := range slots {
			if sl.Kind == Forward {
				inFlight++
			} else {
				inFlight--
			}
			peaks[s] = max(peaks[s], inFlight)
		}
	}
	return peaks
}

// String dibuja el timeline: una fila por etapa, una columna por unidad de
// tiempo. Los forwards se muestran con el número de microbatch (1-based) y
// los backwards con una letra (a = microbatch 1). Los puntos son burbuja.
func (t Timeline) String() string {
	var b strings.Builder
	for s, slots := range t.Stages {
		row := []byte(strings.Repeat(".", t.Makespan))
		for _, sl := range slots {
			var c byte
			if sl.Kind == Forward {
				c = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[sl.Microbatch%35]
			} else {
				c = "abcdefghijklmnopqrstuvwxyz"[sl.Microbatch%26]
			}
			for x := sl.Start; x < sl.End; x++ {
				row[x] = c
			}
		}
		fmt.Fprintf(&b, "GPU %d │%s│\n", s, row)
	}
	return b.String()
}

func filled(n, v int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = v
	}
	return s
}
