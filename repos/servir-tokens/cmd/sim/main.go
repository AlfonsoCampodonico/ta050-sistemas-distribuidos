// sim compara políticas de batching y de memoria con el mismo motor y la
// misma carga, en tiempo virtual:
//
//	go run ./cmd/sim batching   # estático (FasterTransformer) vs continuo (Orca)
//	go run ./cmd/sim memoria    # reserva contigua vs bloques (PagedAttention)
//	go run ./cmd/sim cow        # muestreo paralelo con y sin copy-on-write
//
// Y para la Clase 19, con varias réplicas y una carga de chat de varios turnos:
//
//	go run ./cmd/sim router        # cómo elige réplica el router, y el caché de prefijos
//	go run ./cmd/sim desagregado   # prefill y decode juntos, en pedazos o separados
//	go run ./cmd/sim especulativo  # cuánto acelera la decodificación especulativa
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/fiubaTA050/servir-tokens/engine"
	"github.com/fiubaTA050/servir-tokens/kvcache"
)

const maxLen = 2048

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	args := os.Args[2:]
	switch os.Args[1] {
	case "batching":
		batching(args)
	case "memoria":
		memoria(args)
	case "cow":
		cow(args)
	case "router":
		router(args)
	case "desagregado":
		desagregado(args)
	case "especulativo":
		especulativo(args)
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "uso: sim {batching|memoria|cow|router|desagregado|especulativo} [flags]   (-h para ver los flags)")
	os.Exit(2)
}

func header() {
	fmt.Printf("%-24s %9s %9s %9s %9s %8s %8s %7s\n", "", "tok/s", "TTFT", "TTFT p99", "latencia", "lote", "uso KV", "desaloj")
	fmt.Printf("%-24s %9s %9s %9s %9s %8s %8s %7s\n", "", "", "ms", "ms", "ms", "prom.", "", "")
}

func row(name string, s engine.Summary) {
	fmt.Printf("%-24s %9.0f %9.0f %9.0f %9.0f %8.1f %7.0f%% %7d\n",
		name, s.Throughput, s.TTFTMean, s.TTFTP99, s.LatencyMean, s.Engine.AvgBatch, 100*s.Engine.AvgKVUtil, s.Engine.Preemptions)
}

func simulate(c engine.Config, kv kvcache.Allocator, reqs []engine.Request) engine.Summary {
	s, err := engine.Run(engine.New(c, kv), reqs)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return s
}

func batching(args []string) {
	fs := flag.NewFlagSet("batching", flag.ExitOnError)
	n := fs.Int("n", 400, "pedidos")
	rate := fs.Float64("rate", 4, "pedidos por segundo")
	batch := fs.Int("batch", 32, "secuencias por lote")
	fs.Parse(args)

	reqs := engine.Workload(*n, *rate, 1, maxLen)
	fmt.Printf("%d pedidos a %.0f/s, lote de hasta %d, memoria de sobra (paginada).\n\n", *n, *rate, *batch)
	header()
	for _, mode := range []engine.Mode{engine.Static, engine.Continuous} {
		c := engine.Config{Mode: mode, MaxBatch: *batch, MaxLen: maxLen, Cost: engine.DefaultCost()}
		row("batching "+mode.String(), simulate(c, kvcache.NewPaged(1<<22, 16), reqs))
	}
	fmt.Println("\nEstático: el lote corre hasta que termina la respuesta más larga; nadie ve nada hasta el final.")
	fmt.Println("Continuo: después de cada iteración salen las terminadas y entran las que esperan.")
}

func memoria(args []string) {
	fs := flag.NewFlagSet("memoria", flag.ExitOnError)
	n := fs.Int("n", 400, "pedidos")
	rate := fs.Float64("rate", 40, "pedidos por segundo")
	slots := fs.Int("kv", 16*maxLen, "slots de KV cache (tokens) en la GPU")
	block := fs.Int("block", 16, "tokens por bloque")
	fs.Parse(args)

	reqs := engine.Workload(*n, *rate, 3, maxLen)
	fmt.Printf("%d pedidos a %.0f/s, KV cache de %d tokens, batching continuo.\n\n", *n, *rate, *slots)
	header()
	c := engine.Config{Mode: engine.Continuous, MaxBatch: 256, MaxLen: maxLen, Cost: engine.DefaultCost()}
	c.Reserve = engine.ReserveMax
	row("contiguo, reserva máx", simulate(c, kvcache.NewContiguous(*slots), reqs))
	c.Reserve = engine.ReserveExact
	row("contiguo, oráculo", simulate(c, kvcache.NewContiguous(*slots), reqs))
	row(fmt.Sprintf("paginado, bloque %d", *block), simulate(c, kvcache.NewPaged(*slots, *block), reqs))
	fmt.Println("\nReserva máx = Orca (Max): reserva 2048 tokens por pedido. Oráculo = Orca (Oracle): sabe")
	fmt.Println("de antemano el largo de cada respuesta, imposible en la práctica. Paginado = vLLM.")
	fmt.Println("Uso KV = fracción de la memoria asignada que guarda tokens de verdad.")
}

func cow(args []string) {
	fs := flag.NewFlagSet("cow", flag.ExitOnError)
	prompt := fs.Int("prompt", 500, "tokens del prompt")
	output := fs.Int("output", 100, "tokens de cada muestra")
	samples := fs.Int("n", 4, "muestras del mismo prompt")
	fs.Parse(args)

	used := func(share bool) (blocks, copies int) {
		p := kvcache.NewPaged(1<<20, 16)
		p.Admit(0, *prompt, 0)
		for i := 1; i < *samples; i++ {
			if share {
				p.Fork(0, i)
			} else {
				p.Admit(i, *prompt, 0)
			}
		}
		for range *output {
			for i := range *samples {
				p.Append(i)
			}
		}
		return p.Stats().Allocated / 16, p.Copies()
	}
	without, _ := used(false)
	with, copies := used(true)
	fmt.Printf("%d muestras de un prompt de %d tokens, %d tokens cada una, bloques de 16.\n\n", *samples, *prompt, *output)
	fmt.Printf("sin compartir:          %4d bloques\n", without)
	fmt.Printf("con copy-on-write:      %4d bloques (%d copiados)\n", with, copies)
	fmt.Printf("ahorro:                 %4.0f%%\n", 100*(1-float64(with)/float64(without)))
}
