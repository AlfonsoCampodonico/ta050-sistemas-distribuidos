package layout

import "testing"

// El orden de Llama 3 deja los tensores en un servidor y los datos entre pods;
// al revés, los tensores —lo que más se comunica— cruzarían la red.
func TestOrderDecidesLocality(t *testing.T) {
	good, _ := Place(Llama3, []Dim{{Name: "TP", Size: 8}, {Name: "CP", Size: 1}, {Name: "PP", Size: 16}, {Name: "DP", Size: 128}})
	if good[0].Scope != SameNode || good[2].Scope != SamePod || good[3].Scope != CrossPod {
		t.Fatalf("[TP,CP,PP,DP]: %v %v %v", good[0].Scope, good[2].Scope, good[3].Scope)
	}
	bad, _ := Place(Llama3, []Dim{{Name: "DP", Size: 128}, {Name: "PP", Size: 16}, {Name: "CP", Size: 1}, {Name: "TP", Size: 8}})
	if bad[3].Scope != CrossPod {
		t.Fatalf("TP afuera tenía que cruzar pods, quedó %v", bad[3].Scope)
	}
}

// Con CP = 16 y DP = 8, el grupo de pipeline de la GPU 0 (0..1920) entra en un
// pod, pero el de la GPU 2048 (2048..3968) cruza el borde en 3072: hay que
// reportar el peor grupo, no el primero.
func TestScopeIsWorstGroup(t *testing.T) {
	g, _ := Place(Llama3, []Dim{{Name: "TP", Size: 8}, {Name: "CP", Size: 16}, {Name: "PP", Size: 16}, {Name: "DP", Size: 8}})
	if g[2].Scope != CrossPod {
		t.Fatalf("PP tenía que cruzar pods, quedó %v", g[2].Scope)
	}
}
