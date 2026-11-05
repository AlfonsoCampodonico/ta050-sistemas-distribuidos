package collective

import "sync"

// AllReduce es una operación colectiva: la llaman TODOS los ranks a la vez,
// cada uno con su vector local, y al terminar cada uno tiene en data la suma
// elemento a elemento de los vectores de todos.
type AllReduce func(c *Comm, rank int, data []float64)

// Central es el enfoque "parameter server" ingenuo: todos le envían su
// vector al rank 0, que suma y le devuelve el resultado a cada uno.
//
// Es correcto pero el rank 0 envía y recibe (n-1)·S floats: su enlace es el
// cuello de botella y el costo crece linealmente con la cantidad de ranks.
func Central(c *Comm, rank int, data []float64) {
	n := c.Size()
	if rank == 0 {
		for r := 1; r < n; r++ {
			add(data, c.Recv(r, 0))
		}
		for r := 1; r < n; r++ {
			c.Send(0, r, data)
		}
		return
	}
	c.Send(rank, 0, data)
	copy(data, c.Recv(0, rank))
}

// Ring implementa ring all-reduce (Patarasuk y Yuan, 2009): los ranks forman
// un anillo y cada uno sólo habla con su vecino derecho. El vector se parte
// en n pedazos y la operación tiene dos fases de n-1 pasos cada una:
//
//  1. reduce-scatter: en cada paso, cada rank le pasa un pedazo a su vecino,
//     que lo suma al suyo. Al final, cada rank tiene UN pedazo completamente
//     reducido.
//  2. all-gather: cada rank hace circular su pedazo terminado hasta que todos
//     tienen todos los pedazos.
//
// Cada rank envía 2·(n-1)/n·S floats: casi independiente de n. Ese es el
// óptimo en ancho de banda para all-reduce.
func Ring(c *Comm, rank int, data []float64) {
	n := c.Size()
	if n == 1 {
		return
	}
	right, left := (rank+1)%n, (rank-1+n)%n
	chunk := func(i int) []float64 {
		i = ((i % n) + n) % n
		lo, hi := i*len(data)/n, (i+1)*len(data)/n
		return data[lo:hi]
	}

	// Fase 1: reduce-scatter.
	for s := range n - 1 {
		c.Send(rank, right, chunk(rank-s))
		add(chunk(rank-s-1), c.Recv(left, rank))
	}
	// Ahora el rank tiene completo el pedazo rank+1.

	// Fase 2: all-gather.
	for s := range n - 1 {
		c.Send(rank, right, chunk(rank+1-s))
		copy(chunk(rank-s), c.Recv(left, rank))
	}
}

// Run ejecuta op en todos los ranks en paralelo (una goroutine por rank) y
// espera a que terminen. vectors[r] es el vector local del rank r.
func Run(c *Comm, op AllReduce, vectors [][]float64) {
	var wg sync.WaitGroup
	for r := range c.Size() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			op(c, r, vectors[r])
		}()
	}
	wg.Wait()
}

func add(dst, src []float64) {
	for i := range dst {
		dst[i] += src[i]
	}
}
