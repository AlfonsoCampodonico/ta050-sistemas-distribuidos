// Package collective implementa operaciones colectivas (all-reduce) entre
// N "ranks" simulados. Cada rank es una goroutine y cada enlace dirigido
// entre dos ranks es un canal. No hay red real: lo que nos interesa es
// contar cuántos datos envía cada rank, que es lo que determina el costo
// en un clúster de verdad.
package collective

import "sync/atomic"

// Comm simula la red entre n ranks. links[from][to] es el canal por el que
// from le envía a to. El canal tiene buffer para que un rank pueda enviar
// antes de recibir sin trabarse (como un send no bloqueante de NCCL/MPI).
type Comm struct {
	n     int
	links [][]chan []float64
	sent  []atomic.Int64 // floats enviados por cada rank
}

// NewComm crea una red totalmente conectada entre n ranks.
func NewComm(n int) *Comm {
	if n < 1 {
		panic("collective: se necesita al menos un rank")
	}
	c := &Comm{n: n, links: make([][]chan []float64, n), sent: make([]atomic.Int64, n)}
	for from := range n {
		c.links[from] = make([]chan []float64, n)
		for to := range n {
			c.links[from][to] = make(chan []float64, 4)
		}
	}
	return c
}

// Size devuelve la cantidad de ranks.
func (c *Comm) Size() int { return c.n }

// Send envía una copia de data de from a to. Copiamos porque en una red real
// el receptor recibe bytes, no un puntero a la memoria del emisor.
func (c *Comm) Send(from, to int, data []float64) {
	cp := make([]float64, len(data))
	copy(cp, data)
	c.sent[from].Add(int64(len(data)))
	c.links[from][to] <- cp
}

// Recv bloquea hasta recibir lo que from le envió a to.
func (c *Comm) Recv(from, to int) []float64 {
	return <-c.links[from][to]
}

// SentBy devuelve cuántos floats envió el rank desde el último Reset.
func (c *Comm) SentBy(rank int) int64 { return c.sent[rank].Load() }

// MaxSent devuelve lo que envió el rank que más envió: en un clúster, ese
// rank es el cuello de botella de la operación.
func (c *Comm) MaxSent() int64 {
	var m int64
	for r := range c.n {
		m = max(m, c.SentBy(r))
	}
	return m
}

// Reset pone en cero los contadores.
func (c *Comm) Reset() {
	for r := range c.n {
		c.sent[r].Store(0)
	}
}
