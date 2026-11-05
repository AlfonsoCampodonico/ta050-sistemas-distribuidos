package moe

import "testing"

func TestNodeLimitedRoutingCapsTraffic(t *testing.T) {
	free := Route(Config{Experts: 256, TopK: 8, Nodes: 8, Tokens: 5000, Seed: 1})
	lim := Route(Config{Experts: 256, TopK: 8, Nodes: 8, MaxNodes: 4, Tokens: 5000, Seed: 1})
	if lim.MaxNodes > 4 {
		t.Fatalf("con límite 4 un token tocó %d nodos", lim.MaxNodes)
	}
	// Al azar, 8 expertos de 256 en 8 nodos caen en unos 5,3 nodos distintos.
	if free.Nodes < 5 || free.Nodes > 5.6 || lim.Nodes >= free.Nodes {
		t.Fatalf("nodos por token: libre %.2f, limitado %.2f", free.Nodes, lim.Nodes)
	}
}
