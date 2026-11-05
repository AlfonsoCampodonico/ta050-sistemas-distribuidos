package kvcache

import "sort"

// Contiguous reserva un tramo contiguo de slots por secuencia, del tamaño de
// su longitud máxima, con first-fit. Sufre los tres desperdicios que describe
// el paper de vLLM (§3.1):
//
//   - reserva: slots apartados para tokens que todavía no se generaron;
//   - fragmentación interna: slots reservados que nunca se van a usar, porque
//     la respuesta terminó antes del máximo;
//   - fragmentación externa: huecos libres entre tramos, demasiado chicos para
//     una secuencia nueva aunque en total sobre memoria.
type Contiguous struct {
	capacity int
	free     []span // ordenados por inicio, sin solaparse
	seqs     map[int]*cseq
}

type span struct{ start, len int }

type cseq struct {
	span
	used int
}

// NewContiguous crea un administrador con capacity slots.
func NewContiguous(capacity int) *Contiguous {
	return &Contiguous{
		capacity: capacity,
		free:     []span{{0, capacity}},
		seqs:     map[int]*cseq{},
	}
}

func (c *Contiguous) Admit(id, tokens, maxTokens int) bool {
	need := max(tokens, maxTokens)
	for i, f := range c.free {
		if f.len < need {
			continue
		}
		c.seqs[id] = &cseq{span: span{f.start, need}, used: tokens}
		if f.len == need {
			c.free = append(c.free[:i], c.free[i+1:]...)
		} else {
			c.free[i] = span{f.start + need, f.len - need}
		}
		return true
	}
	return false
}

func (c *Contiguous) Append(id int) bool {
	s := c.seqs[id]
	if s.used == s.len {
		return false
	}
	s.used++
	return true
}

func (c *Contiguous) Free(id int) {
	s, ok := c.seqs[id]
	if !ok {
		return
	}
	delete(c.seqs, id)
	c.free = append(c.free, s.span)
	sort.Slice(c.free, func(i, j int) bool { return c.free[i].start < c.free[j].start })
	merged := c.free[:1]
	for _, f := range c.free[1:] {
		last := &merged[len(merged)-1]
		if last.start+last.len == f.start {
			last.len += f.len
		} else {
			merged = append(merged, f)
		}
	}
	c.free = merged
}

func (c *Contiguous) Stats() Stats {
	st := Stats{Capacity: c.capacity}
	for _, s := range c.seqs {
		st.Allocated += s.len
		st.Used += s.used
	}
	return st
}

// ExternalFragmentation es la fracción de la memoria libre que no está en el
// hueco más grande: memoria que existe pero no sirve para una secuencia larga.
func (c *Contiguous) ExternalFragmentation() float64 {
	total, largest := 0, 0
	for _, f := range c.free {
		total += f.len
		largest = max(largest, f.len)
	}
	if total == 0 {
		return 0
	}
	return 1 - float64(largest)/float64(total)
}
