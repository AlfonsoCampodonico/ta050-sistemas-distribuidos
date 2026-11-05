// Package cluster modela un clúster de GPUs compartido: racks de servidores,
// cada servidor con la misma cantidad de GPUs, y cómo se elige en qué GPUs
// corre un trabajo.
//
// Lo que importa para la clase es la topología: las GPUs de un mismo
// servidor se hablan por NVLink, las de servidores distintos por la red, y
// las de racks distintos atraviesan más saltos. Un trabajo distribuido hace
// un all-reduce por paso (Clase 16), así que repartirlo en más servidores lo
// hace más lento. Pero exigir que entre en pocos servidores hace que espere:
// con el clúster a dos tercios, en Philly menos del 4,5% de los servidores
// estaba completamente vacío (Jeon et al., ATC 2019, §3.1.1).
package cluster

import (
	"fmt"
	"sort"
)

// Topology es la forma del clúster.
type Topology struct {
	Racks          int
	ServersPerRack int
	GPUsPerServer  int
}

// Default es el clúster de las demos: 4 racks de 8 servidores de 8 GPUs, 256 GPUs.
var Default = Topology{Racks: 4, ServersPerRack: 8, GPUsPerServer: 8}

func (t Topology) Servers() int { return t.Racks * t.ServersPerRack }
func (t Topology) GPUs() int    { return t.Servers() * t.GPUsPerServer }
func (t Topology) Rack(server int) int {
	return server / t.ServersPerRack
}

// Slot son g GPUs de un servidor.
type Slot struct{ Server, GPUs int }

// Placement es dónde corre un trabajo: un slot por servidor que usa.
type Placement []Slot

// GPUs es el total de GPUs de la ubicación.
func (p Placement) GPUs() int {
	n := 0
	for _, s := range p {
		n += s.GPUs
	}
	return n
}

// Servers es en cuántos servidores distintos está repartido.
func (p Placement) Servers() int { return len(p) }

// Racks es en cuántos racks distintos está repartido.
func (p Placement) Racks(t Topology) int {
	seen := map[int]bool{}
	for _, s := range p {
		seen[t.Rack(s.Server)] = true
	}
	return len(seen)
}

// MinServers es la menor cantidad de servidores en la que entran n GPUs.
func (t Topology) MinServers(n int) int { return (n + t.GPUsPerServer - 1) / t.GPUsPerServer }

// Locality es cuánto se le exige a una ubicación.
type Locality int

const (
	// Packed exige la menor cantidad posible de servidores, todos en un rack:
	// un servidor si el trabajo entra en uno; si no, servidores vacíos.
	Packed Locality = iota
	// Anywhere intenta Packed, después un solo rack, y si no, donde haya lugar.
	Anywhere
)

func (l Locality) String() string {
	if l == Packed {
		return "consolidada"
	}
	return "donde haya lugar"
}

// Cluster lleva cuántas GPUs libres tiene cada servidor.
type Cluster struct {
	T    Topology
	free []int
}

// New arma un clúster vacío.
func New(t Topology) *Cluster {
	c := &Cluster{T: t, free: make([]int, t.Servers())}
	for i := range c.free {
		c.free[i] = t.GPUsPerServer
	}
	return c
}

// FreeOn es cuántas GPUs libres tiene el servidor s.
func (c *Cluster) FreeOn(s int) int { return c.free[s] }

// Free es el total de GPUs libres.
func (c *Cluster) Free() int {
	n := 0
	for _, f := range c.free {
		n += f
	}
	return n
}

// EmptyServers es cuántos servidores tienen todas sus GPUs libres.
func (c *Cluster) EmptyServers() int {
	n := 0
	for _, f := range c.free {
		if f == c.T.GPUsPerServer {
			n++
		}
	}
	return n
}

// Fragmentation es la fracción de las GPUs libres que está en servidores a
// medio usar: GPUs que sirven para un trabajo chico pero no para uno que
// necesita un servidor entero. 0 si no hay GPUs libres.
func (c *Cluster) Fragmentation() float64 {
	free, loose := 0, 0
	for _, f := range c.free {
		free += f
		if f < c.T.GPUsPerServer {
			loose += f
		}
	}
	if free == 0 {
		return 0
	}
	return float64(loose) / float64(free)
}

// Allocate ocupa las GPUs de p. Entra en pánico si no están libres: eso es un
// error del planificador, no del trabajo.
func (c *Cluster) Allocate(p Placement) {
	for _, s := range p {
		if c.free[s.Server] < s.GPUs {
			panic(fmt.Sprintf("servidor %d: pide %d GPUs, hay %d", s.Server, s.GPUs, c.free[s.Server]))
		}
		c.free[s.Server] -= s.GPUs
	}
}

// Release libera las GPUs de p.
func (c *Cluster) Release(p Placement) {
	for _, s := range p {
		c.free[s.Server] += s.GPUs
		if c.free[s.Server] > c.T.GPUsPerServer {
			panic(fmt.Sprintf("servidor %d: más GPUs libres que las que tiene", s.Server))
		}
	}
}

// Find busca dónde ubicar n GPUs con la localidad pedida, sin ocuparlas.
// Todas las búsquedas son best-fit: entre los lugares que sirven, el que deja
// menos libre, para no romper servidores vacíos si no hace falta.
func (c *Cluster) Find(n int, l Locality) (Placement, bool) {
	if p, ok := c.packed(n); ok {
		return p, true
	}
	if l == Packed {
		return nil, false
	}
	if p, ok := c.oneRack(n); ok {
		return p, true
	}
	return c.spread(n, 0, c.T.Servers())
}

// packed: un servidor si n entra en uno; si no, servidores vacíos de un rack.
func (c *Cluster) packed(n int) (Placement, bool) {
	g := c.T.GPUsPerServer
	if n <= g {
		best := -1
		for s, f := range c.free {
			if f >= n && (best < 0 || f < c.free[best]) {
				best = s
			}
		}
		if best < 0 {
			return nil, false
		}
		return Placement{{best, n}}, true
	}
	k := c.T.MinServers(n)
	bestRack, bestEmpty := -1, 0
	for r := 0; r < c.T.Racks; r++ {
		empty := 0
		for s := r * c.T.ServersPerRack; s < (r+1)*c.T.ServersPerRack; s++ {
			if c.free[s] == g {
				empty++
			}
		}
		if empty >= k && (bestRack < 0 || empty < bestEmpty) {
			bestRack, bestEmpty = r, empty
		}
	}
	if bestRack < 0 {
		return nil, false
	}
	var p Placement
	left := n
	for s := bestRack * c.T.ServersPerRack; left > 0; s++ {
		if c.free[s] == g {
			take := min(g, left)
			p = append(p, Slot{s, take})
			left -= take
		}
	}
	return p, true
}

// oneRack: todo en un rack, en la menor cantidad de servidores de ese rack.
func (c *Cluster) oneRack(n int) (Placement, bool) {
	best, bestFree := -1, 0
	for r := 0; r < c.T.Racks; r++ {
		f := 0
		for s := r * c.T.ServersPerRack; s < (r+1)*c.T.ServersPerRack; s++ {
			f += c.free[s]
		}
		if f >= n && (best < 0 || f < bestFree) {
			best, bestFree = r, f
		}
	}
	if best < 0 {
		return nil, false
	}
	return c.spread(n, best*c.T.ServersPerRack, (best+1)*c.T.ServersPerRack)
}

// spread toma GPUs de los servidores [from, to), primero de los que tienen
// más libres, para tocar la menor cantidad de servidores.
func (c *Cluster) spread(n, from, to int) (Placement, bool) {
	idx := make([]int, 0, to-from)
	total := 0
	for s := from; s < to; s++ {
		if c.free[s] > 0 {
			idx = append(idx, s)
			total += c.free[s]
		}
	}
	if total < n {
		return nil, false
	}
	sort.SliceStable(idx, func(i, j int) bool { return c.free[idx[i]] > c.free[idx[j]] })
	var p Placement
	for _, s := range idx {
		if n == 0 {
			break
		}
		take := min(c.free[s], n)
		p = append(p, Slot{s, take})
		n -= take
	}
	return p, true
}

// Speed es a qué fracción de su velocidad ideal corre un trabajo en esa
// ubicación. Ideal es la menor cantidad de servidores, en un rack. Los
// trabajos sensibles (modelos con un tensor enorme, como VGG o AlexNet:
// Gu et al., NSDI 2019, §2.2) pierden mucho al repartirse; el resto, poco.
// Los números son ilustrativos: Tiresias mide hasta 1,67× entre ubicar bien
// y ubicar al azar, de ahí el 0,6.
func Speed(t Topology, p Placement, sensitive bool) float64 {
	n := p.GPUs()
	switch {
	case p.Servers() <= t.MinServers(n) && p.Racks(t) == 1:
		return 1
	case p.Racks(t) == 1:
		if sensitive {
			return 0.6
		}
		return 0.95
	default:
		if sensitive {
			return 0.5
		}
		return 0.9
	}
}
