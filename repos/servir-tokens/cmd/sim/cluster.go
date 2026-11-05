package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/fiubaTA050/servir-tokens/cluster"
	"github.com/fiubaTA050/servir-tokens/engine"
)

// chat es la carga de las demos de la Clase 19: conversaciones de varios
// turnos, cada turno con la conversación entera como prompt.
func chat(fs *flag.FlagSet) func() (engine.Chat, cluster.Config, cluster.SLO) {
	sessions := fs.Int("sesiones", 1000, "conversaciones")
	rate := fs.Float64("rate", 5, "conversaciones nuevas por segundo")
	cache := fs.Int("cache", 1024, "bloques de 64 tokens en el caché de prefijos de cada réplica")
	batch := fs.Int("batch", 128, "secuencias por lote")
	ttft := fs.Float64("slo-ttft", 2000, "SLO: ms hasta el primer token")
	gap := fs.Float64("slo-tbt", 100, "SLO: ms entre tokens, como máximo")
	return func() (engine.Chat, cluster.Config, cluster.SLO) {
		ch := engine.DefaultChat()
		ch.Sessions, ch.Rate = *sessions, *rate
		c := cluster.Config{
			Engine:      engine.Config{Mode: engine.Continuous, MaxBatch: *batch, MaxLen: ch.MaxLen, Cost: engine.DefaultCost(), PrefixBlock: ch.Block},
			KVTokens:    1 << 18,
			Block:       16,
			CacheBlocks: *cache,
		}
		return ch, c, cluster.SLO{TTFT: *ttft, MaxGap: *gap}
	}
}

func describe(ch engine.Chat, reqs []engine.Request, replicas int) {
	prompt, output := 0, 0
	for _, r := range reqs {
		prompt += r.Prompt
		output += r.Output
	}
	fmt.Printf("%d conversaciones (%d pedidos), prompt promedio %d tokens, respuesta %d, %d réplicas.\n",
		ch.Sessions, len(reqs), prompt/len(reqs), output/len(reqs), replicas)
}

func must(s cluster.Summary, err error) cluster.Summary {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return s
}

func router(args []string) {
	fs := flag.NewFlagSet("router", flag.ExitOnError)
	n := fs.Int("replicas", 4, "réplicas")
	setup := chat(fs)
	fs.Parse(args)
	ch, c, slo := setup()
	reqs := ch.Requests()
	describe(ch, reqs, *n)
	fmt.Printf("Caché de prefijos de %d bloques (%d tokens) por réplica.\n\n", c.CacheBlocks, c.CacheBlocks*ch.Block)
	fmt.Printf("%-34s %7s %12s %10s %10s %10s\n", "", "caché", "prefill", "TTFT p50", "TTFT p99", "desbalance")
	fmt.Printf("%-34s %7s %12s %10s %10s %10s\n", "", "", "M tokens", "ms", "ms", "")
	row := func(name string, s cluster.Summary) {
		fmt.Printf("%-34s %6.0f%% %12.1f %10.0f %10.0f %10.2f\n", name, 100*s.HitRate, float64(s.Computed)/1e6, s.TTFTP50, s.TTFTP99, s.Imbalance)
	}
	off := c
	off.CacheBlocks = 0
	row("sin caché, en ronda", must(cluster.Run(off, *n, &cluster.RoundRobin{}, reqs, slo)))
	for _, r := range []cluster.Router{&cluster.RoundRobin{}, cluster.LeastLoaded{}, cluster.Sticky{}, cluster.PrefixAware{Cost: c.Engine.Cost}} {
		row(r.Name(), must(cluster.Run(c, *n, r, reqs, slo)))
	}
	pool := c
	pool.PoolBlocks = 1 << 16
	row("consciente del prefijo + pool", must(cluster.Run(pool, *n, cluster.PrefixAware{Cost: c.Engine.Cost}, reqs, slo)))
	fmt.Println("\nCaché = fracción de los tokens de prompt que no hubo que calcular. Desbalance = prefill de la")
	fmt.Println("réplica más cargada sobre el promedio. Pool = un caché compartido por todas las réplicas, más")
	fmt.Println("lento de leer que el de la GPU pero diez veces más barato que recalcular (Mooncake).")
}

func desagregado(args []string) {
	fs := flag.NewFlagSet("desagregado", flag.ExitOnError)
	chunk := fs.Int("chunk", 512, "tokens de prefill por iteración en el modo en pedazos")
	transfer := fs.Float64("transfer", 0.003, "ms por token para mandar el KV cache de prefill a decode")
	setup := chat(fs)
	fs.Parse(args)
	ch, c, slo := setup()
	reqs := ch.Requests()
	describe(ch, reqs, 4)
	fmt.Printf("SLO: primer token en menos de %.0f ms y nunca más de %.0f ms entre dos tokens.\n\n", slo.TTFT, slo.MaxGap)
	fmt.Printf("%-30s %10s %10s %9s %8s %10s\n", "", "TTFT p50", "TTFT p99", "TBT p99", "SLO", "goodput")
	fmt.Printf("%-30s %10s %10s %9s %8s %10s\n", "", "ms", "ms", "ms", "", "pedidos/s")
	row := func(name string, s cluster.Summary) {
		fmt.Printf("%-30s %10.0f %10.0f %9.0f %7.0f%% %10.2f\n", name, s.TTFTP50, s.TTFTP99, s.TBTP99, 100*s.SLOMet, s.Goodput)
	}
	pa := cluster.PrefixAware{Cost: c.Engine.Cost}
	row("colocado, 4 réplicas", must(cluster.Run(c, 4, pa, reqs, slo)))
	chunked := c
	chunked.Engine.ChunkPrefill = *chunk
	row(fmt.Sprintf("colocado, prefill de a %d", *chunk), must(cluster.Run(chunked, 4, pa, reqs, slo)))
	for _, pd := range [][2]int{{1, 3}, {2, 2}, {3, 1}} {
		row(fmt.Sprintf("desagregado, %dP + %dD", pd[0], pd[1]), must(cluster.RunDisaggregated(c, pd[0], pd[1], *transfer, reqs, slo)))
	}
	fmt.Println("\nTBT = tiempo entre tokens. Goodput = pedidos por segundo que cumplieron el SLO (DistServe).")
	fmt.Println("En el desagregado, el salto del primer token al segundo incluye mandar el KV cache.")
}

// especulativo calcula la aceleración esperada de la decodificación
// especulativa (Leviathan et al., ICML 2023): un modelo chico propone k
// tokens, el grande los verifica todos en una pasada y acepta cada uno con
// probabilidad alpha. Como el decode está limitado por la memoria, verificar
// k+1 tokens cuesta casi lo mismo que generar uno.
func especulativo(args []string) {
	fs := flag.NewFlagSet("especulativo", flag.ExitOnError)
	c := fs.Float64("c", 0.05, "costo de una pasada del modelo chico, relativo al grande")
	fs.Parse(args)
	fmt.Printf("Aceleración esperada con un borrador que cuesta %.0f%% de una pasada del modelo grande.\n\n", 100**c)
	fmt.Printf("%-22s %8s %8s %8s %8s\n", "tokens propuestos (k)", "α=0,6", "α=0,7", "α=0,8", "α=0,9")
	for _, k := range []int{1, 2, 4, 8} {
		fmt.Printf("%-22d", k)
		for _, a := range []float64{0.6, 0.7, 0.8, 0.9} {
			tokens := (1 - math.Pow(a, float64(k+1))) / (1 - a)
			fmt.Printf(" %7.2f×", tokens/(float64(k)**c+1))
		}
		fmt.Println()
	}
	fmt.Println("\nα = probabilidad de que el modelo grande acepte cada token propuesto. Tokens por pasada")
	fmt.Println("= (1 − α^(k+1)) / (1 − α); costo = k·c + 1 pasadas. Si α es bajo, proponer más empeora.")
}
