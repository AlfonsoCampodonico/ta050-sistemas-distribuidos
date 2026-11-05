package trace

import (
	"reflect"
	"testing"
)

// La carga tiene la forma de Philly: la mayoría de los trabajos usa una GPU,
// la mediana dura cerca de media hora, y unos pocos trabajos grandes y
// largos se llevan la mayor parte del GPU·tiempo.
func TestShape(t *testing.T) {
	s := Describe(Generate(Default(256)))
	if s.ShareJobs["1"] < 0.65 {
		t.Errorf("trabajos de 1 GPU: %.0f%%", 100*s.ShareJobs["1"])
	}
	if s.ShareGPUTime["1"] > 0.2 {
		t.Errorf("los de 1 GPU no deberían pedir casi nada del GPU·tiempo: %.0f%%", 100*s.ShareGPUTime["1"])
	}
	if s.MedianMin < 15 || s.MedianMin > 45 {
		t.Errorf("mediana %.0f min", s.MedianMin)
	}
	if s.P99H < 24 {
		t.Errorf("cola pesada: el p99 debería pasar el día, es %.1f h", s.P99H)
	}
	if s.TopShare < 0.3 {
		t.Errorf("el 1%% más grande se lleva %.0f%% del GPU·tiempo", 100*s.TopShare)
	}
}

func TestDeterministic(t *testing.T) {
	if !reflect.DeepEqual(Generate(Default(256)), Generate(Default(256))) {
		t.Fatal("la misma semilla debería dar la misma carga")
	}
}
