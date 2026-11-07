// Package localsgd entrena con sincronización infrecuente, la idea de
// DiLoCo (Douillard et al., 2023): en lugar de hacer un all-reduce de los
// gradientes en cada paso, cada réplica da H pasos por su cuenta y recién
// ahí se promedian los cambios. Se comunica H veces menos.
//
// Cada ronda externa, cada réplica calcula cuánto se movió desde los pesos
// compartidos (su "pseudo-gradiente"), se promedian esos cambios y se
// aplican con un optimizador externo con momento de Nesterov. Con H = 1 es
// paralelismo de datos común sólo porque el optimizador interno es SGD y el
// externo, con Momentum = 0 (como en dilocoCmd), un promedio simple: con
// AdamW adentro, como en el paper, ya no sería equivalente.
//
// El modelo es la regresión lineal del paquete train: es convexa, así que
// este experimento es optimista. DiLoCo muestra que con LLMs reales también
// funciona, con 8 réplicas y 500 veces menos comunicación.
package localsgd

import (
	"github.com/fiubaTA050/entrenamiento-distribuido/train"
)

// Config describe un entrenamiento.
type Config struct {
	Workers    int
	InnerSteps int     // H: pasos locales entre sincronizaciones
	TotalSteps int     // pasos locales de cada réplica en total
	Batch      int     // ejemplos por paso, por réplica
	LR         float64 // del optimizador interno (SGD)
	OuterLR    float64 // del optimizador externo
	Momentum   float64 // Nesterov del optimizador externo; 0 = promedio simple
}

// Result resume el entrenamiento.
type Result struct {
	Loss   float64
	Syncs  int // rondas de comunicación
	Floats int // floats que envió cada réplica (un all-reduce en anillo manda ≈ 2·tamaño)
}

// Run entrena. Cada réplica ve su propio pedazo del dataset.
func Run(ds train.Dataset, c Config) Result {
	dim := len(ds.X[0])
	global := make([]float64, dim)
	velocity := make([]float64, dim)
	shard := len(ds.X) / c.Workers
	local := make([][]float64, c.Workers)
	cursor := make([]int, c.Workers)
	var res Result
	for step := 0; step < c.TotalSteps; step += c.InnerSteps {
		delta := make([]float64, dim)
		for w := range c.Workers {
			local[w] = append(local[w][:0], global...)
			for range min(c.InnerSteps, c.TotalSteps-step) {
				lo := w*shard + cursor[w]%shard
				g := train.Gradient(ds, local[w], lo, c.Batch)
				for j := range g {
					local[w][j] -= c.LR * g[j]
				}
				cursor[w] += c.Batch
			}
			for j := range delta {
				delta[j] += (global[j] - local[w][j]) / float64(c.Workers)
			}
		}
		// Optimizador externo: Nesterov sobre el pseudo-gradiente promedio.
		for j := range global {
			velocity[j] = c.Momentum*velocity[j] + delta[j]
			global[j] -= c.OuterLR * (delta[j] + c.Momentum*velocity[j])
		}
		res.Syncs++
		res.Floats += 2 * dim
	}
	res.Loss = train.Loss(ds, global)
	return res
}
