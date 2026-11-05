// Package moe simula el ruteo de tokens en una capa mixture-of-experts con
// paralelismo de expertos (EP): los expertos están repartidos entre GPUs de
// varios nodos, y cada token tiene que viajar a las GPUs de los expertos que
// eligió. Esa comunicación es un all-to-all, y la parte que cruza entre nodos
// va por InfiniBand, unas 3,2 veces más lento que NVLink adentro del nodo
// (DeepSeek-V3, §3.2.2).
//
// DeepSeek-V3 tiene 256 expertos ruteados por capa y cada token activa 8
// (§4.2). Con ruteo libre, un token toca muchos nodos. Con "node-limited
// routing" cada token va a lo sumo a M = 4 nodos: primero se eligen los
// nodos con mejor puntaje y después los expertos dentro de ellos (§2.1.2).
// DeepSeek además manda cada token una sola vez por nodo por la red y lo
// reparte adentro del nodo por NVLink.
package moe

import (
	"math/rand/v2"
	"sort"
)

// Config describe la capa y el ruteo.
type Config struct {
	Experts  int // expertos ruteados por capa (256 en DeepSeek-V3)
	TopK     int // expertos por token (8)
	Nodes    int // nodos entre los que se reparten los expertos (8: EP de 64 GPUs)
	MaxNodes int // a cuántos nodos puede ir cada token; 0 = sin límite
	Tokens   int // tokens a simular
	Seed     uint64
}

// Result es el promedio por token.
type Result struct {
	Nodes       float64 // nodos distintos que toca cada token
	RemoteNodes float64 // de esos, los que no son el nodo donde está el token
	MaxNodes    int
}

// Route simula el ruteo con afinidades al azar (expertos balanceados).
func Route(c Config) Result {
	rng := rand.New(rand.NewPCG(c.Seed, 5))
	perNode := c.Experts / c.Nodes
	score := make([]float64, c.Experts)
	idx := make([]int, c.Experts)
	var res Result
	for range c.Tokens {
		for e := range score {
			score[e] = rng.Float64()
		}
		allowed := allNodes(c.Nodes)
		if c.MaxNodes > 0 && c.MaxNodes < c.Nodes {
			allowed = topNodes(score, c.Nodes, perNode, c.TopK/c.MaxNodes, c.MaxNodes)
		}
		// Top-k de los expertos de los nodos permitidos.
		n := 0
		for _, node := range allowed {
			for e := node * perNode; e < (node+1)*perNode; e++ {
				idx[n] = e
				n++
			}
		}
		cand := idx[:n]
		sort.Slice(cand, func(a, b int) bool { return score[cand[a]] > score[cand[b]] })
		touched := map[int]bool{}
		for _, e := range cand[:c.TopK] {
			touched[e/perNode] = true
		}
		home := rng.IntN(c.Nodes)
		res.Nodes += float64(len(touched))
		res.RemoteNodes += float64(len(touched))
		if touched[home] {
			res.RemoteNodes--
		}
		res.MaxNodes = max(res.MaxNodes, len(touched))
	}
	res.Nodes /= float64(c.Tokens)
	res.RemoteNodes /= float64(c.Tokens)
	return res
}

func allNodes(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// topNodes elige los m nodos cuya suma de sus mejores k afinidades es mayor.
func topNodes(score []float64, nodes, perNode, k, m int) []int {
	type ns struct {
		node int
		s    float64
	}
	all := make([]ns, nodes)
	buf := make([]float64, perNode)
	for node := range nodes {
		copy(buf, score[node*perNode:(node+1)*perNode])
		sort.Sort(sort.Reverse(sort.Float64Slice(buf)))
		var s float64
		for _, v := range buf[:k] {
			s += v
		}
		all[node] = ns{node, s}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	out := make([]int, m)
	for i := range out {
		out[i] = all[i].node
	}
	return out
}
