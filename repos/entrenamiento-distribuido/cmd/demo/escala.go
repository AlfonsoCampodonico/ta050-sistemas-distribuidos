package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/fiubaTA050/entrenamiento-distribuido/fallas"
	"github.com/fiubaTA050/entrenamiento-distribuido/layout"
	"github.com/fiubaTA050/entrenamiento-distribuido/localsgd"
	"github.com/fiubaTA050/entrenamiento-distribuido/moe"
	"github.com/fiubaTA050/entrenamiento-distribuido/train"
)

func layoutCmd(args []string) {
	fs := flag.NewFlagSet("4d", flag.ExitOnError)
	tp := fs.Int("tp", 8, "paralelismo de tensores")
	cp := fs.Int("cp", 1, "paralelismo de contexto")
	pp := fs.Int("pp", 16, "paralelismo de pipeline")
	dp := fs.Int("dp", 128, "paralelismo de datos (FSDP)")
	order := fs.String("orden", "tp,cp,pp,dp", "orden de las dimensiones, de la que varía más rápido a la más lenta")
	fs.Parse(args)

	all := map[string]layout.Dim{
		"tp": {Name: "TP", Size: *tp, When: "4 all-reduce por capa y microbatch"},
		"cp": {Name: "CP", Size: *cp, When: "all-gather de K y V por capa"},
		"pp": {Name: "PP", Size: *pp, When: "activaciones a la etapa vecina, por microbatch"},
		"dp": {Name: "DP", Size: *dp, When: "FSDP: all-gather de pesos y reduce-scatter de gradientes"},
	}
	var dims []layout.Dim
	for _, k := range strings.Split(*order, ",") {
		d, ok := all[strings.TrimSpace(k)]
		if !ok {
			fmt.Fprintln(os.Stderr, "dimensión desconocida:", k)
			os.Exit(2)
		}
		dims = append(dims, d)
	}
	gs, err := layout.Place(layout.Llama3, dims)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	total := *tp * *cp * *pp * *dp
	fmt.Printf("%d GPUs en servidores de 8, pods de %d GPUs (la red del clúster de Llama 3).\n\n", total, 8*layout.Llama3.NodesPerPod)
	fmt.Printf("%-4s %6s %8s %10s  %-22s %s\n", "dim", "tamaño", "salto", "servidores", "un grupo queda", "se comunica")
	for _, g := range gs {
		fmt.Printf("%-4s %6d %8d %10d  %-22s %s\n", g.Name, g.Size, g.Stride, g.Nodes, g.Scope, g.When)
	}
	fmt.Println("\nSalto = distancia en ranks entre dos GPUs del mismo grupo. La primera dimensión del orden")
	fmt.Println("queda entre GPUs vecinas (NVLink); la última, lo más lejos posible. Probá -orden dp,pp,cp,tp.")
}

func moeCmd(args []string) {
	fs := flag.NewFlagSet("moe", flag.ExitOnError)
	experts := fs.Int("expertos", 256, "expertos ruteados por capa")
	topk := fs.Int("k", 8, "expertos por token")
	nodes := fs.Int("nodos", 8, "nodos con expertos (EP = nodos × 8 GPUs)")
	fs.Parse(args)

	fmt.Printf("%d expertos en %d nodos (%d por nodo), %d expertos por token.\n\n", *experts, *nodes, *experts / *nodes, *topk)
	fmt.Printf("%-24s %14s %16s %10s\n", "", "nodos por token", "remotos (por IB)", "máximo")
	for _, m := range []int{0, 4, 2} {
		r := moe.Route(moe.Config{Experts: *experts, TopK: *topk, Nodes: *nodes, MaxNodes: m, Tokens: 20000, Seed: 1})
		name := "ruteo libre"
		if m > 0 {
			name = fmt.Sprintf("a lo sumo %d nodos", m)
		}
		fmt.Printf("%-24s %14.2f %16.2f %10d\n", name, r.Nodes, r.RemoteNodes, r.MaxNodes)
	}
	fmt.Println("\nCada nodo remoto es un envío del token por InfiniBand (unas 3,2 veces más lento que NVLink).")
	fmt.Println("DeepSeek-V3 limita a 4 nodos por token y manda una sola copia por nodo (§2.1.2, §3.2.2).")
}

func fallasCmd(args []string) {
	fs := flag.NewFlagSet("fallas", flag.ExitOnError)
	gpus := fs.Int("gpus", 16384, "GPUs del entrenamiento")
	days := fs.Float64("dias", 54, "días de cómputo útil que hacen falta")
	ckpt := fs.Float64("checkpoint", 30, "segundos que se frena el entrenamiento por cada checkpoint")
	async := fs.Float64("async", 3, "segundos que frena un checkpoint asincrónico")
	restart := fs.Float64("restart", 180, "segundos para reemplazar el nodo y volver a arrancar")
	fs.Parse(args)

	fmt.Printf("Cada GPU falla una vez cada %.0f horas en promedio (Llama 3: 419 fallas en 54 días con 16.384 GPUs).\n\n", fallas.LlamaGPUMTBF)
	fmt.Printf("%8s %14s %12s\n", "GPUs", "una falla cada", "fallas/día")
	for _, n := range []int{1024, 4096, 16384, 65536, 131072} {
		c := fallas.Config{GPUs: n, GPUMTBF: fallas.LlamaGPUMTBF}
		fmt.Printf("%8d %12.1f h %12.1f\n", n, c.ClusterMTBF(), 24/c.ClusterMTBF())
	}

	base := fallas.Config{GPUs: *gpus, GPUMTBF: fallas.LlamaGPUMTBF, Work: *days * 24,
		CheckpointCost: *ckpt / 3600, RestartCost: *restart / 3600, Runs: 200, Seed: 1}
	opt := base.YoungDaly()
	fmt.Printf("\n%d GPUs, %.0f días de cómputo, checkpoint de %.0f s, reinicio de %.0f s.\n", *gpus, *days, *ckpt, *restart)
	fmt.Printf("Intervalo óptimo (Young/Daly, √(2·C·M)): %.0f minutos.\n\n", 60*opt)
	fmt.Printf("%-22s %10s %10s %12s %12s %10s\n", "checkpoint cada", "fallas", "rehecho", "guardando", "reiniciando", "tiempo útil")
	for _, tau := range []float64{opt / 4, opt, 1, 4} {
		c := base
		c.CheckpointEvery = tau
		r := fallas.Simulate(c)
		label := fmt.Sprintf("%.0f min", 60*tau)
		if tau == opt {
			label += " (óptimo)"
		}
		fmt.Printf("%-22s %10.0f %8.0f h %10.0f h %10.0f h %9.1f%%\n", label, r.Failures, r.Lost, r.Checkpoint, r.Restart, 100*r.Goodput)
	}
	a := base
	a.CheckpointCost = *async / 3600
	a.CheckpointEvery = a.YoungDaly()
	r := fallas.Simulate(a)
	fmt.Printf("%-22s %10.0f %8.0f h %10.0f h %10.0f h %9.1f%%\n", fmt.Sprintf("asincrónico, %.0f min", 60*a.CheckpointEvery), r.Failures, r.Lost, r.Checkpoint, r.Restart, 100*r.Goodput)
	fmt.Println("\nRehecho = cómputo perdido desde el último checkpoint, que hay que volver a calcular.")
	fmt.Println("Costos ilustrativos, elegidos para que con 16.384 GPUs dé más de 90% como en Llama 3.")
}

func dilocoCmd(args []string) {
	fs := flag.NewFlagSet("diloco", flag.ExitOnError)
	workers := fs.Int("workers", 8, "réplicas")
	steps := fs.Int("steps", 400, "pasos locales por réplica")
	fs.Parse(args)

	ds := train.NewDataset(8192, 32, 7)
	fmt.Printf("%d réplicas, %d pasos cada una, regresión lineal de %d pesos.\n\n", *workers, *steps, len(ds.X[0]))
	fmt.Printf("%-26s %12s %16s %12s\n", "sincronizar cada", "rondas", "floats enviados", "loss")
	for _, h := range []int{1, 10, 50, 200} {
		r := localsgd.Run(ds, localsgd.Config{Workers: *workers, InnerSteps: h, TotalSteps: *steps, Batch: 16, LR: 0.02, OuterLR: 1})
		label := fmt.Sprintf("%d pasos", h)
		if h == 1 {
			label = "1 paso (datos común)"
		}
		fmt.Printf("%-26s %12d %16d %12.2e\n", label, r.Syncs, r.Floats, r.Loss)
	}
	fmt.Println("\nCada réplica avanza H pasos sola; después se promedian los cambios. H veces menos comunicación.")
	fmt.Println("Ojo: el modelo es convexo y todas las réplicas ven datos parecidos, el mejor caso posible.")
	fmt.Println("DiLoCo muestra que con LLMs también funciona, con un optimizador externo con momento.")
}
