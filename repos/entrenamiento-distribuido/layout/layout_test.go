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
