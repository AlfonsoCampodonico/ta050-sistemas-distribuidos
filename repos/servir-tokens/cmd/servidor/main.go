// servidor levanta el motor detrás de HTTP, con batching continuo y KV cache
// paginado, y devuelve los tokens por SSE:
//
//	go run ./cmd/servidor
//	curl -N localhost:8080/v1/chat -d '{"messages":[{"role":"user","content":"¿Por qué el cielo es azul?"}]}'
//	curl localhost:8080/estado
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os/signal"
	"syscall"

	"github.com/fiubaTA050/servir-tokens/engine"
	"github.com/fiubaTA050/servir-tokens/kvcache"
	"github.com/fiubaTA050/servir-tokens/server"
)

func main() {
	addr := flag.String("addr", ":8080", "dirección donde escuchar")
	kvTokens := flag.Int("kv", 8192, "slots de KV cache (tokens)")
	block := flag.Int("block", 16, "tokens por bloque")
	batch := flag.Int("batch", 64, "secuencias por lote")
	speed := flag.Float64("speed", 1, "ms reales por ms simulado (2 = el doble de lento)")
	flag.Parse()

	eng := engine.New(engine.Config{Mode: engine.Continuous, MaxBatch: *batch, MaxLen: 2048, Cost: engine.DefaultCost()},
		kvcache.NewPaged(*kvTokens, *block))
	srv := server.New(eng, *speed)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go srv.Run(ctx)

	hs := &http.Server{Addr: *addr, Handler: srv.Handler()}
	go func() {
		<-ctx.Done()
		hs.Close()
	}()
	log.Printf("escuchando en %s: POST /v1/chat, GET /estado", *addr)
	if err := hs.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
