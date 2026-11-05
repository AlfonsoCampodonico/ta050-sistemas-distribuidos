// Package layout ubica las cuatro dimensiones de paralelismo de un
// entrenamiento grande sobre el hardware.
//
// Cada GPU tiene una coordenada por dimensión: tensores (TP), contexto (CP),
// pipeline (PP) y datos (DP). El rank de la GPU se arma con la primera
// dimensión variando más rápido, así que la primera queda entre GPUs vecinas
// —el mismo servidor, unidas por NVLink— y la última queda lo más lejos
// posible. Llama 3 usa el orden [TP, CP, PP, DP] (Grattafiori et al., 2024,
// §3.3.2): adentro lo que más se comunica y afuera lo que tolera más latencia.
package layout

import "fmt"

// Dim es una dimensión de paralelismo.
type Dim struct {
	Name string
	Size int
	// Cuándo se comunica, para mostrar.
	When string
}

// Topology describe el clúster: GPUs por servidor y servidores por pod (el
// grupo con ancho de banda completo entre sí; en el clúster de Llama 3, 3.072
// GPUs, con sobresuscripción 1:7 hacia arriba).
type Topology struct {
	GPUsPerNode int
	NodesPerPod int
}

// Llama3 es la red del clúster de Llama 3 (§3.3.1).
var Llama3 = Topology{GPUsPerNode: 8, NodesPerPod: 384}

// Scope es hasta dónde llega un grupo de comunicación.
type Scope int

const (
	SameNode Scope = iota // NVLink
	SamePod               // la red, sin sobresuscripción
	CrossPod              // la red, pasando por la capa sobresuscripta
)

func (s Scope) String() string {
	return [...]string{"dentro de un servidor", "dentro de un pod", "entre pods"}[s]
}

// Group describe los grupos de una dimensión.
type Group struct {
	Dim
	Stride int   // distancia en ranks entre dos miembros consecutivos
	Scope  Scope // hasta dónde llega el grupo más disperso
	Nodes  int   // servidores que toca un grupo
}

// Place devuelve dónde cae cada dimensión, en el orden dado (la primera
// varía más rápido).
func Place(t Topology, dims []Dim) ([]Group, error) {
	total := 1
	for _, d := range dims {
		if d.Size <= 0 {
			return nil, fmt.Errorf("dimensión %s de tamaño %d", d.Name, d.Size)
		}
		total *= d.Size
	}
	groups := make([]Group, len(dims))
	stride := 1
	for i, d := range dims {
		g := Group{Dim: d, Stride: stride}
		// El grupo de la GPU 0: ranks 0, stride, 2·stride, …
		nodes, pods := map[int]bool{}, map[int]bool{}
		for k := range d.Size {
			rank := k * stride
			node := rank / t.GPUsPerNode
			nodes[node] = true
			pods[node/t.NodesPerPod] = true
		}
		g.Nodes = len(nodes)
		switch {
		case len(nodes) == 1:
			g.Scope = SameNode
		case len(pods) == 1:
			g.Scope = SamePod
		default:
			g.Scope = CrossPod
		}
		groups[i] = g
		stride *= d.Size
	}
	_ = total
	return groups, nil
}
