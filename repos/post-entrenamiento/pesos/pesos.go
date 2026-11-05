// Package pesos calcula cuánto cuesta llevar los pesos recién entrenados a
// donde se genera. Hay dos casos:
//
//   - Pools separados: el trainer manda una copia completa a cada GPU de
//     rollout. Ingenuo, el trainer la manda n veces; en anillo (o en pipeline,
//     partida en pedazos), cada GPU reenvía a la siguiente mientras recibe y
//     el tiempo casi no depende de n. Es el broadcast de la Clase 16.
//   - Colocado: las mismas GPUs pasan del layout de entrenamiento (p, t, d) al
//     de generación (pg, tg, dg). El volumen que se mueve depende de cómo se
//     haga; las fórmulas son las de la tabla 2 de HybridFlow (§5.4).
package pesos

// Broadcast devuelve los segundos para mandar gb gigabytes a n GPUs con
// enlaces de gbs GB/s.
func Broadcast(gb, gbs float64, n int, anillo bool) float64 {
	if n <= 0 {
		return 0
	}
	if anillo {
		// En pipeline con muchos pedazos: gb/gbs más el llenado del pipeline,
		// que con pedazos chicos es despreciable. Contamos un 1% por salto.
		return gb / gbs * (1 + 0.01*float64(n-1))
	}
	return float64(n) * gb / gbs
}

// Layout es una forma de partir el modelo: pipeline × tensores.
type Layout struct{ P, T int }

// Volumen de la transición del layout de entrenamiento (p, t, d) al de
// generación, como fracción del tamaño del modelo M (tabla 2 de HybridFlow).
//
//   - DS-Chat: all-gather de todo el modelo entre las p·t·d GPUs: (tpd−1)/(tpd).
//   - HybridFlow-V: all-gather dentro de cada grupo de p·t: (tp−1)/(tp).
//   - HybridFlow (3D-HybridEngine): cada GPU sólo junta lo que le falta para su
//     pedazo de generación, sin copias de más: (tp − tg·pg)/(tg·pg·tp).
func VolumenDSChat(entr Layout, d int) float64 {
	n := float64(entr.T * entr.P * d)
	return (n - 1) / n
}

func VolumenHybridFlowV(entr Layout) float64 {
	n := float64(entr.T * entr.P)
	return (n - 1) / n
}

func VolumenHybridFlow(entr, gen Layout) float64 {
	tp := float64(entr.T * entr.P)
	g := float64(gen.T * gen.P)
	return (tp - g) / (g * tp)
}
