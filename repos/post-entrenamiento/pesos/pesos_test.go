package pesos

import "testing"

// Ingenuo, el trainer manda una copia por GPU; en anillo, casi no depende de n.
func TestBroadcast(t *testing.T) {
	if r := Broadcast(140, 25, 48, false) / Broadcast(140, 25, 6, false); r != 8 {
		t.Errorf("ingenuo debería escalar con n: %.2f", r)
	}
	if Broadcast(140, 25, 48, true) > 1.5*Broadcast(140, 25, 6, true) {
		t.Error("en anillo no debería depender casi de n")
	}
	if Broadcast(140, 25, 6, true) >= Broadcast(140, 25, 6, false) {
		t.Error("el anillo debería ganar")
	}
}

// Tabla 2 de HybridFlow: el 3D-HybridEngine mueve menos que las otras dos, y
// nada si el layout de generación es igual al de entrenamiento.
func TestTransicion(t *testing.T) {
	entr := Layout{P: 1, T: 8}
	hf := VolumenHybridFlow(entr, Layout{P: 1, T: 2})
	if !(hf < VolumenHybridFlowV(entr) && VolumenHybridFlowV(entr) < VolumenDSChat(entr, 2)) {
		t.Errorf("volúmenes: %.3f %.3f %.3f", hf, VolumenHybridFlowV(entr), VolumenDSChat(entr, 2))
	}
	if VolumenHybridFlow(entr, entr) != 0 {
		t.Error("mismo layout: no hay nada que mover")
	}
}
