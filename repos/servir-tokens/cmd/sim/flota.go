package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/fiubaTA050/servir-tokens/flota"
)

// arranque compara, para la Clase 20, cómo se sirven muchos modelos en pocas
// GPUs: cuánto tarda una carga desde cada nivel, dónde conviene arrancar
// cada modelo y cuánto tiempo dejarlo cargado.
func arranque(args []string) {
	fs := flag.NewFlagSet("arranque", flag.ExitOnError)
	models := fs.Int("modelos", 64, "modelos distintos")
	rate := fs.Float64("rate", 1, "pedidos por segundo, en total")
	cv := fs.Float64("cv", 4, "coeficiente de variación de las llegadas (1 = Poisson)")
	keep := fs.Float64("keepalive", 30, "segundos que un modelo ocioso queda cargado")
	gpus := fs.Int("gpus", 4, "GPUs por servidor")
	fs.Parse(args)

	c := flota.DefaultConfig()
	c.KeepAlive, c.GPUs = *keep, *gpus
	h := c.Hardware

	fmt.Printf("1 · Cuánto tarda en llegar un checkpoint a la GPU, según dónde está (segundos).\n")
	fmt.Printf("    Red %.2f GB/s, SSD %.0f GB/s, PCIe %.0f GB/s.\n\n", h.Net, h.SSD, h.PCIe)
	fmt.Printf("%-28s %9s %9s %9s\n", "", "DRAM", "SSD", "remoto")
	for _, gb := range []float64{13, 130} {
		for _, l := range []flota.Loader{flota.Naive, flota.Fast} {
			fmt.Printf("%-28s", fmt.Sprintf("%.0f GB, cargador %s", gb, l.Name))
			for _, t := range []flota.Tier{flota.DRAM, flota.SSD, flota.Remote} {
				fmt.Printf(" %9.1f", h.LoadTime(l, t, gb))
			}
			fmt.Println()
		}
	}

	w := flota.DefaultWorkload()
	w.Models, w.Rate, w.CV = *models, *rate, *cv
	reqs := w.Requests()
	fmt.Printf("\n2 · Dónde arrancar cada modelo. %d modelos, %d pedidos en %.0f min (%.1f/s, CV %.0f),\n",
		w.Models, len(reqs), w.Duration/60, w.Rate, w.CV)
	fmt.Printf("    %d servidores de %d GPUs, checkpoints de %.0f GB, keep-alive de %.0f s.\n\n", c.Servers, c.GPUs, c.ModelGB, c.KeepAlive)
	header := func() {
		fmt.Printf("%-30s %6s %7s %7s %7s %9s %9s %9s\n", "", "frío", "DRAM", "SSD", "remoto", "demora", "demora", "mudadas")
		fmt.Printf("%-30s %6s %7s %7s %7s %9s %9s %9s\n", "", "", "", "", "", "media s", "p99 s", "/ cortadas")
	}
	row := func(name string, s flota.Summary) {
		loads := s.Loads[0] + s.Loads[1] + s.Loads[2]
		pct := func(n int) float64 { return 100 * float64(n) / float64(max(1, loads)) }
		fmt.Printf("%-30s %5.0f%% %6.0f%% %6.0f%% %6.0f%% %9.1f %9.1f %9s\n", name, 100*s.Cold,
			pct(s.Loads[0]), pct(s.Loads[1]), pct(s.Loads[2]), s.DelayMean, s.DelayP99,
			fmt.Sprintf("%d / %d", s.Migrations, s.Preemptions))
	}
	header()
	naive := c
	naive.Policy, naive.Loader = flota.Random, flota.Naive
	row("al azar, cargador ingenuo", flota.Run(naive, reqs))
	for _, p := range []flota.Policy{flota.Random, flota.Locality, flota.Preempt, flota.Migrate} {
		cc := c
		cc.Policy = p
		row(p.String(), flota.Run(cc, reqs))
	}
	fmt.Println("\nFrío = pedidos que esperaron una carga. DRAM, SSD, remoto = de dónde salieron las cargas.")
	fmt.Println("Demora = lo que tardó cada pedido de más respecto de tener el modelo cargado y la GPU para él.")

	fmt.Printf("\n3 · Cuánto tiempo dejar cargado un modelo ocioso.\n\n")
	classic := c
	classic.DRAMGB, classic.SSDGB, classic.Replicas = 0, 0, 0
	classic.Loader, classic.Policy = flota.Naive, flota.Random
	fmt.Printf("%-21s %26s   %26s\n", "", "descarga cada vez", "DRAM y SSD como caché")
	fmt.Printf("%-21s %26s   %26s\n", "", "(serverless clásico)", "(ServerlessLLM)")
	fmt.Printf("%-21s %6s %9s %10s   %6s %9s %10s\n", "keep-alive", "frío", "demora", "GPU", "frío", "demora", "GPU")
	fmt.Printf("%-21s %6s %9s %10s   %6s %9s %10s\n", "", "", "media s", "ociosa", "", "media s", "ociosa")
	for _, k := range []float64{0, 10, 60, 600, math.Inf(1)} {
		name := fmt.Sprintf("%.0f s", k)
		if math.IsInf(k, 1) {
			name = "hasta que haga falta"
		}
		fmt.Printf("%-21s", name)
		for i, base := range []flota.Config{classic, c} {
			cc := base
			cc.KeepAlive = k
			s := flota.Run(cc, reqs)
			sep := ""
			if i == 1 {
				sep = "  "
			}
			fmt.Printf("%s %5.0f%% %9.1f %9.0f%%", sep, 100*s.Cold, s.DelayMean, 100*s.IdleWarm)
		}
		fmt.Println()
	}
	fmt.Println("\nGPU ociosa = fracción del tiempo de GPU con un modelo cargado y sin pedidos: lo que se paga")
	fmt.Println("por el keep-alive. Si hace falta la GPU para otro modelo, el ocioso se desaloja.")

	d := flota.Dedicated(c, reqs)
	s := flota.Run(c, reqs)
	fmt.Printf("\n4 · Contra una GPU por modelo, siempre cargado.\n\n")
	fmt.Printf("%-26s %6s %12s %9s %9s\n", "", "GPUs", "trabajando", "demora", "demora")
	fmt.Printf("%-26s %6s %12s %9s %9s\n", "", "", "", "media s", "p99 s")
	fmt.Printf("%-26s %6d %11.0f%% %9.1f %9.1f\n", "una GPU por modelo", d.GPUs, 100*d.Busy, d.DelayMean, d.DelayP99)
	fmt.Printf("%-26s %6d %11.0f%% %9.1f %9.1f\n", "flota compartida", s.GPUs, 100*s.Busy, s.DelayMean, s.DelayP99)
	fmt.Println("\nCon una GPU por modelo no hay arranques en frío, pero casi todas están ociosas casi siempre.")
}

// lora compara dos formas de servir muchos adaptadores LoRA de un mismo
// modelo base en una GPU: fusionar el adaptador en los pesos y armar lotes de
// un solo adaptador, o dejar los pesos solos y mezclar adaptadores en el lote
// (S-LoRA).
func lora(args []string) {
	fs := flag.NewFlagSet("lora", flag.ExitOnError)
	rate := fs.Float64("rate", 10, "pedidos por segundo en la tabla de detalle")
	sat := fs.Float64("saturar", 40, "pedidos por segundo para medir la capacidad")
	n := fs.Int("adaptadores", 200, "adaptadores en la tabla de detalle")
	fs.Parse(args)

	fmt.Printf("1 · Capacidad: pedidos por segundo que se terminan con más carga de la que se puede atender\n")
	fmt.Printf("    (%.0f pedidos/s durante 5 min), según cuántos adaptadores distintos hay.\n\n", *sat)
	fmt.Printf("%-14s %14s %14s %16s\n", "adaptadores", "fusionado", "lote mixto", "adaptadores")
	fmt.Printf("%-14s %14s %14s %16s\n", "", "pedidos/s", "pedidos/s", "por lote, mixto")
	for _, k := range []int{1, 2, 5, 100, 1000, 2000} {
		reqs := flota.LoRAWorkload(k, *sat).Requests()
		mc := flota.DefaultLoRAConfig()
		mc.Mode = flota.Merged
		mg := flota.RunLoRA(mc, reqs)
		mx := flota.RunLoRA(flota.DefaultLoRAConfig(), reqs)
		fmt.Printf("%-14d %14.1f %14.1f %16.0f\n", k, mg.Throughput, mx.Throughput, mx.Distinct)
	}

	reqs := flota.LoRAWorkload(*n, *rate).Requests()
	fmt.Printf("\n2 · Con %d adaptadores y %.0f pedidos/s.\n\n", *n, *rate)
	fmt.Printf("%-34s %9s %9s %9s %7s %8s %8s %9s\n", "", "pedidos/s", "TTFT p50", "TTFT p99", "lote", "cambios", "cargas", "memoria")
	fmt.Printf("%-34s %9s %9s %9s %7s %8s %8s %9s\n", "", "", "ms", "ms", "prom.", "", "", "adapt.")
	show := func(name string, c flota.LoRAConfig) {
		s := flota.RunLoRA(c, reqs)
		fmt.Printf("%-34s %9.2f %9.0f %9.0f %7.1f %8d %8d %8.1f%%\n", name, s.Throughput, s.TTFTP50, s.TTFTP99, s.AvgBatch, s.Switches, s.Loads, 100*s.AdapterMem)
	}
	mc := flota.DefaultLoRAConfig()
	mc.Mode = flota.Merged
	show("fusionado, un adaptador por lote", mc)
	nc := flota.DefaultLoRAConfig()
	nc.Cache = false
	show("lote mixto, sin caché de adaptadores", nc)
	show("lote mixto, con caché (S-LoRA)", flota.DefaultLoRAConfig())
	fmt.Println("\nCambios = veces que se cambió el adaptador fusionado. Cargas = adaptadores traídos del host.")
	fmt.Println("Memoria adapt. = fracción del pool unificado que ocupan los adaptadores, en promedio.")
	fmt.Printf("\nY la alternativa más ingenua: una copia fusionada del modelo por adaptador. En 80 GB entran %d\n", flota.MergedCopies(80, 13.5))
	fmt.Println("copias de un modelo de 7B (13,5 GB), sin dejar lugar para el KV cache.")
}
