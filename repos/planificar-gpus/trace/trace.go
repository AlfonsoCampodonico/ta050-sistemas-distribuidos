// Package trace genera una carga sintética con la forma de las trazas
// publicadas de clústeres de entrenamiento: muchísimos trabajos chicos, unos
// pocos enormes, y duraciones con cola pesada.
//
// Lo que se sabe de Philly (Microsoft, 2017): más del 82% de los trabajos
// usaba una sola GPU (Weng et al., NSDI 2022, §2), la duración mediana era
// de 26 minutos (Weng et al., §3.2), el 0,5% tardaba más de una semana y los
// trabajos con más GPUs tienden a durar más (Jeon et al., ATC 2019, §2.4).
// Las proporciones de abajo son ilustrativas, con esa forma.
package trace

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Job es un trabajo de entrenamiento tal como llega al planificador.
type Job struct {
	ID        int
	Arrival   float64 // segundos
	GPUs      int
	Duration  float64 // segundos de ejecución en una ubicación ideal; el planificador no la conoce
	Sensitive bool    // pierde mucho si se reparte en más servidores de los necesarios
}

// GPUTime es el servicio que necesita: GPUs × duración, en GPU·segundos.
func (j Job) GPUTime() float64 { return float64(j.GPUs) * j.Duration }

// Size es la clase de tamaño de un trabajo, como en las tablas de Philly.
func Size(gpus int) string {
	switch {
	case gpus == 1:
		return "1"
	case gpus <= 8:
		return "2-8"
	default:
		return ">8"
	}
}

// Sizes son las clases en orden.
var Sizes = []string{"1", "2-8", ">8"}

// Config describe la carga.
type Config struct {
	Jobs      int
	TotalGPUs int     // GPUs del clúster, para calcular la tasa de llegada
	Load      float64 // GPU·tiempo pedido sobre GPU·tiempo disponible, en promedio
	Median    float64 // duración mediana de un trabajo de 1 GPU, en minutos
	Sigma     float64 // dispersión del lognormal: más alto, cola más pesada
	MaxDays   float64 // tope de duración
	Seed      uint64
}

// Default es la carga de las demos: 2.000 trabajos en 256 GPUs, al 90%.
func Default(totalGPUs int) Config {
	return Config{Jobs: 2000, TotalGPUs: totalGPUs, Load: 0.9, Median: 26, Sigma: 2, MaxDays: 5, Seed: 21}
}

// sizes y su probabilidad: el 70% de 1 GPU, unos pocos de 16 y 32.
var sizes = []struct {
	gpus int
	p    float64
}{{1, 0.70}, {2, 0.06}, {4, 0.08}, {8, 0.10}, {16, 0.04}, {32, 0.02}}

// Generate arma la carga. Es determinística dada la semilla.
func Generate(c Config) []Job {
	r := rand.New(rand.NewPCG(c.Seed, 0x7a05))
	jobs := make([]Job, c.Jobs)
	total := 0.0
	for i := range jobs {
		u, g := r.Float64(), sizes[len(sizes)-1].gpus
		for _, s := range sizes {
			if u < s.p {
				g = s.gpus
				break
			}
			u -= s.p
		}
		// Los trabajos más grandes duran más: la mediana crece con log2(GPUs).
		med := c.Median * 60 * (1 + 0.5*math.Log2(float64(g)))
		d := math.Min(med*math.Exp(c.Sigma*r.NormFloat64()), c.MaxDays*86400)
		d = math.Max(d, 60)
		jobs[i] = Job{ID: i, GPUs: g, Duration: d, Sensitive: g > 1 && r.Float64() < 0.5}
		total += jobs[i].GPUTime()
	}
	// Llegadas de Poisson con la tasa que da la carga pedida.
	rate := c.Load * float64(c.TotalGPUs) / (total / float64(c.Jobs))
	t := 0.0
	for i := range jobs {
		t += r.ExpFloat64() / rate
		jobs[i].Arrival = t
	}
	return jobs
}

// Stats resume la forma de la carga.
type Stats struct {
	Jobs                    int
	ShareJobs, ShareGPUTime map[string]float64 // por clase de tamaño
	MedianMin, P90H, P99H   float64            // duración
	TopShare                float64            // fracción del GPU·tiempo del 1% de trabajos más grandes
	Span                    float64            // segundos entre la primera y la última llegada
}

// Describe calcula las estadísticas de una carga.
func Describe(jobs []Job) Stats {
	s := Stats{Jobs: len(jobs), ShareJobs: map[string]float64{}, ShareGPUTime: map[string]float64{}}
	d := make([]float64, len(jobs))
	w := make([]float64, len(jobs))
	total := 0.0
	for i, j := range jobs {
		s.ShareJobs[Size(j.GPUs)]++
		s.ShareGPUTime[Size(j.GPUs)] += j.GPUTime()
		d[i], w[i] = j.Duration, j.GPUTime()
		total += j.GPUTime()
	}
	for k := range s.ShareJobs {
		s.ShareJobs[k] /= float64(len(jobs))
		s.ShareGPUTime[k] /= total
	}
	sort.Float64s(d)
	sort.Sort(sort.Reverse(sort.Float64Slice(w)))
	s.MedianMin = Percentile(d, 0.5) / 60
	s.P90H = Percentile(d, 0.9) / 3600
	s.P99H = Percentile(d, 0.99) / 3600
	top := 0.0
	for _, x := range w[:max(1, len(w)/100)] {
		top += x
	}
	s.TopShare = top / total
	if len(jobs) > 0 {
		s.Span = jobs[len(jobs)-1].Arrival - jobs[0].Arrival
	}
	return s
}

// Percentile de una lista ordenada.
func Percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	i := int(p * float64(len(sorted)))
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}
