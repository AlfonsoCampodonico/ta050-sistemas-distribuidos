// sim imprime las tablas de la Clase 22, en tiempo virtual:
//
//	go run ./cmd/sim cola        # un paso sincrónico espera a la respuesta más larga
//	go run ./cmd/sim ubicacion   # colocado vs pools separados, y cuánto cuesta mover los pesos
//	go run ./cmd/sim asinc       # sincrónico vs asincrónico con staleness acotada
//	go run ./cmd/sim parciales   # rollouts parciales (Kimi) e interrumpibles (AReaL)
//	go run ./cmd/sim aprender    # qué le hace la staleness al aprendizaje
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/fiubaTA050/post-entrenamiento/bucle"
	"github.com/fiubaTA050/post-entrenamiento/pesos"
	"github.com/fiubaTA050/post-entrenamiento/tarea"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	args := os.Args[2:]
	switch os.Args[1] {
	case "cola":
		cola(args)
	case "ubicacion":
		ubicacion(args)
	case "asinc":
		asinc(args)
	case "parciales":
		parciales(args)
	case "aprender":
		aprender(args)
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "uso: sim {cola|ubicacion|asinc|parciales|aprender} [flags]   (-h para ver los flags)")
	os.Exit(2)
}

// flags comunes
type comunes struct {
	pasos, semillas, gpus, rollout *int
	lr, pesosGB                    *float64
}

func flags(fs *flag.FlagSet) comunes {
	d := bucle.Default()
	return comunes{
		pasos:    fs.Int("pasos", d.Pasos, "pasos de entrenamiento"),
		semillas: fs.Int("semillas", 6, "corridas a promediar"),
		gpus:     fs.Int("gpus", d.GPUs, "GPUs en total"),
		rollout:  fs.Int("rollout", d.Rollout, "GPUs de rollout cuando los pools están separados"),
		lr:       fs.Float64("lr", d.LR, "tasa de aprendizaje"),
		pesosGB:  fs.Float64("pesos", d.PesosGB, "GB de pesos para mandar después de cada paso"),
	}
}

func (f comunes) config() bucle.Config {
	c := bucle.Default()
	c.Pasos, c.GPUs, c.Rollout, c.LR, c.PesosGB = *f.pasos, *f.gpus, *f.rollout, *f.lr, *f.pesosGB
	return c
}

// prom corre n semillas y promedia.
type prom struct {
	tiempo, mps, ocioso, ocGen, ocEntr, stale, final float64
	staleMax                                         int
	fg, fe, fs                                       float64
	paso75                                           float64
	curvas                                           [][]bucle.Punto
}

func correr(c bucle.Config, n int) prom {
	t := tarea.Default()
	var p prom
	f := 1 / float64(n)
	for s := 1; s <= n; s++ {
		c.Semilla = int64(s)
		r := bucle.Correr(c, t)
		p.tiempo += f * r.Tiempo
		p.mps += f * r.MuestrasPorSeg()
		p.ocioso += f * r.Ocioso
		p.ocGen += f * r.OciosoGen
		p.ocEntr += f * r.OciosoEntr
		p.stale += f * r.Stale
		p.staleMax = max(p.staleMax, r.StaleMax)
		p.final += f * r.Final()
		p.fg += f * r.FracGen
		p.fe += f * r.FracEntr
		p.fs += f * r.FracSync
		k := r.PasoHasta(0.75)
		if k < 0 {
			k = c.Pasos + 1
		}
		p.paso75 += f * float64(k)
		p.curvas = append(p.curvas, r.Curva)
	}
	return p
}

func pct(x float64) string { return fmt.Sprintf("%.0f%%", 100*x) }

func cola(args []string) {
	fs := flag.NewFlagSet("cola", flag.ExitOnError)
	f := flags(fs)
	seed := fs.Int64("semilla", 1, "semilla")
	fs.Parse(args)
	c := f.config()
	t := tarea.Default()
	opt, _ := t.Optimo()
	fmt.Printf("Un paso sincrónico: %d problemas × %d respuestas en %d GPUs de rollout, hasta %d secuencias por GPU.\n\n",
		c.Problemas, c.Grupo, c.GPUs, c.Slots)
	fmt.Printf("%-22s %8s %8s %8s %8s %9s %9s %9s\n", "política", "medio", "p50", "p99", "máx", "1ª GPU", "última", "ociosa")
	fmt.Printf("%-22s %8s %8s %8s %8s %9s %9s %9s\n", "", "tokens", "tokens", "tokens", "tokens", "s", "s", "")
	for _, x := range []struct {
		nombre string
		p      tarea.Politica
	}{{"al empezar", tarea.Inicial()}, {"ya entrenada", opt}} {
		rng := rand.New(rand.NewSource(*seed))
		var ls []int
		for range c.Problemas * c.Grupo {
			ls = append(ls, t.Muestrear(rng, x.p).Largo)
		}
		fin, util := bucle.Generar(c, ls)
		sort.Ints(ls)
		sort.Float64s(fin)
		s := 0
		for _, l := range ls {
			s += l
		}
		fmt.Printf("%-22s %8d %8d %8d %8d %9.0f %9.0f %9s\n", x.nombre, s/len(ls), ls[len(ls)/2], ls[len(ls)*99/100], ls[len(ls)-1],
			fin[0], fin[len(fin)-1], pct(1-util))
	}
	fmt.Println("\nEl paso dura lo que tarda la última GPU, y la última es la que tiene la respuesta más larga.")
	fmt.Println("Ociosa: GPU·tiempo sin trabajo útil (lugares vacíos en el lote) hasta que termina la última.")
}

func ubicacion(args []string) {
	fs := flag.NewFlagSet("ubicacion", flag.ExitOnError)
	f := flags(fs)
	fs.Parse(args)
	c := f.config()
	n := *f.semillas
	fmt.Printf("Sincrónico, %d pasos, %d GPUs. Tiempo por paso y en qué se va.\n\n", c.Pasos, c.GPUs)
	fmt.Printf("%-32s %8s %8s   %s\n", "ubicación", "s/paso", "ociosa", "en qué se va")
	col := correr(c, n)
	fmt.Printf("%-32s %8.0f %8s   generar %s, entrenar %s, transición %s\n", "colocado (verl)", col.tiempo/float64(c.Pasos), pct(col.ocioso), pct(col.fg), pct(col.fe), pct(col.fs))
	for _, r := range []int{c.GPUs * 3 / 4, c.GPUs / 2} {
		cs := c
		cs.Ubicacion, cs.Rollout, cs.Eta = bucle.Separado, r, 0
		p := correr(cs, n)
		fmt.Printf("%-32s %8.0f %8s   ociosas: rollout %s, trainer %s; pesos %s\n", fmt.Sprintf("separado, %d rollout + %d trainer", r, c.GPUs-r),
			p.tiempo/float64(c.Pasos), pct(p.ocioso), pct(p.ocGen), pct(p.ocEntr), pct(p.fs))
	}
	fmt.Println("\nMover 140 GB de pesos (un modelo de 70B en bf16, HybridFlow §2.3) a 25 GB/s (200 Gb/s):")
	fmt.Printf("%-20s %12s %12s\n", "GPUs de rollout", "ingenuo", "en anillo")
	for _, k := range []int{6, 48, 384} {
		fmt.Printf("%-20d %11.0fs %11.1fs\n", k, pesos.Broadcast(140, 25, k, false), pesos.Broadcast(140, 25, k, true))
	}
	entr, gen := pesos.Layout{P: 1, T: 8}, pesos.Layout{P: 1, T: 2}
	fmt.Println("\nColocado: volumen para pasar del layout de entrenamiento (t=8, p=1, d=2) al de generación (tg=2),")
	fmt.Println("como fracción del modelo (HybridFlow, tabla 2):")
	fmt.Printf("  DS-Chat         %.3f M\n", pesos.VolumenDSChat(entr, 2))
	fmt.Printf("  HybridFlow-V    %.3f M\n", pesos.VolumenHybridFlowV(entr))
	fmt.Printf("  HybridFlow      %.3f M\n", pesos.VolumenHybridFlow(entr, gen))
}

func asinc(args []string) {
	fs := flag.NewFlagSet("asinc", flag.ExitOnError)
	f := flags(fs)
	fs.Parse(args)
	c := f.config()
	n := *f.semillas
	fmt.Printf("%d pasos, %d GPUs; los asincrónicos con %d de rollout, interrumpibles y con corrección.\n\n", c.Pasos, c.GPUs, c.Rollout)
	fmt.Printf("%-26s %8s %8s %9s %10s %10s %10s\n", "", "tiempo", "resp/s", "ociosa", "staleness", "máx", "vs verl")
	fmt.Printf("%-26s %8s %8s %9s %10s %10s %10s\n", "", "s", "", "", "promedio", "", "")
	base := correr(c, n)
	fila := func(nombre string, p prom) {
		fmt.Printf("%-26s %8.0f %8.2f %9s %10.2f %10d %9.2f×\n", nombre, p.tiempo, p.mps, pct(p.ocioso), p.stale, p.staleMax, base.tiempo/p.tiempo)
	}
	fila("colocado, sincrónico", base)
	for _, eta := range []int{0, 1, 2, 4, 8, -1} {
		cs := c
		cs.Ubicacion, cs.Eta, cs.Interrumpible, cs.Corregir = bucle.Separado, eta, true, true
		nombre := fmt.Sprintf("separado, η = %d", eta)
		if eta == 0 {
			nombre = "separado, sincrónico"
		}
		if eta < 0 {
			nombre = "separado, sin límite"
		}
		fila(nombre, correr(cs, n))
	}
	fmt.Println("\nη = staleness máxima (AReaL): los rollouts pueden ir hasta η versiones adelante del trainer.")
	fmt.Println("Una respuesta larga que empezó con pesos viejos termina más tarde: por eso el máximo pasa a η.")
	fmt.Println("Con 6 + 2 el trainer da abasto, así que sin límite queda igual que η = 8; probá -rollout 7.")
}

func parciales(args []string) {
	fs := flag.NewFlagSet("parciales", flag.ExitOnError)
	f := flags(fs)
	fs.Parse(args)
	c := f.config()
	n := *f.semillas
	fmt.Printf("%d pasos, %d GPUs.\n\n", c.Pasos, c.GPUs)
	fmt.Printf("%-40s %8s %9s %10s %11s\n", "", "tiempo", "ociosa", "staleness", "recompensa")
	fila := func(nombre string, p prom) {
		fmt.Printf("%-40s %7.0fs %9s %10.2f %11.3f\n", nombre, p.tiempo, pct(p.ocioso), p.stale, p.final)
	}
	fila("colocado, sincrónico", correr(c, n))
	cp := c
	cp.Parciales = true
	fila("colocado, rollouts parciales (Kimi)", correr(cp, n))
	cp.Corregir = true
	fila("  … y con corrección", correr(cp, n))
	cs := c
	cs.Ubicacion, cs.Eta, cs.Corregir = bucle.Separado, 4, true
	fila("separado, η = 4, termina lo que tiene", correr(cs, n))
	cs.Interrumpible = true
	fila("separado, η = 4, interrumpible (AReaL)", correr(cs, n))
}

func aprender(args []string) {
	fs := flag.NewFlagSet("aprender", flag.ExitOnError)
	f := flags(fs)
	fs.Parse(args)
	c := f.config()
	n := *f.semillas
	t := tarea.Default()
	_, rmax := t.Optimo()
	fmt.Printf("Recompensa esperada al terminar (%d pasos, promedio de %d corridas; la mejor posible es %.3f).\n", c.Pasos, n, rmax)
	fmt.Printf("Paso: el primero en que la recompensa llega a 0,75.\n\n")
	fmt.Printf("%-22s %10s %10s %8s %10s %8s\n", "", "staleness", "sin corr.", "paso", "con corr.", "paso")
	var curvas [][][]bucle.Punto
	var nombres []string
	for _, eta := range []int{0, 1, 2, 4, 8, -1} {
		cs := c
		cs.Ubicacion, cs.Eta, cs.Interrumpible = bucle.Separado, eta, true
		sin := correr(cs, n)
		cs.Corregir = true
		con := correr(cs, n)
		nombre := fmt.Sprintf("η = %d", eta)
		if eta == 0 {
			nombre = "sincrónico"
		}
		if eta < 0 {
			nombre = "sin límite"
		}
		fmt.Printf("%-22s %10.2f %10.3f %8s %10.3f %8s\n", nombre, sin.stale, sin.final, paso(sin.paso75, c.Pasos), con.final, paso(con.paso75, c.Pasos))
		if eta == 0 || eta == 8 {
			curvas = append(curvas, sin.curvas)
			nombres = append(nombres, nombre)
		}
	}
	fmt.Println("\nLa recompensa por paso (promedio), sincrónico vs η = 8 sin corrección:")
	fmt.Printf("%-24s", "paso")
	for k := 0; k <= c.Pasos; k += 5 {
		fmt.Printf("%6d", k)
	}
	fmt.Println()
	for i, cs := range curvas {
		fmt.Printf("%-24s", nombres[i])
		for k := 0; k <= c.Pasos; k += 5 {
			s := 0.0
			for _, cv := range cs {
				s += cv[k].Recompensa
			}
			fmt.Printf("%6.2f", s/float64(len(cs)))
		}
		fmt.Println()
	}
	fmt.Println("\nSin corrección, el gradiente que sale de muestras viejas es el de la política vieja:")
	fmt.Println("sigue empujando hacia respuestas más largas cuando ya se pasó del tope, y se cortan.")
}

func paso(x float64, pasos int) string {
	if x > float64(pasos) {
		return "nunca"
	}
	return fmt.Sprintf("%.1f", x)
}
