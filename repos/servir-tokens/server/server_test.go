package server

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/fiubaTA050/servir-tokens/engine"
	"github.com/fiubaTA050/servir-tokens/kvcache"
)

func start(t *testing.T) (*Server, *httptest.Server) {
	t.Helper()
	eng := engine.New(engine.Config{Mode: engine.Continuous, MaxBatch: 32, MaxLen: 2048, Cost: engine.DefaultCost()},
		kvcache.NewPaged(8192, 16))
	srv := New(eng, 0.05)
	ctx, cancel := context.WithCancel(context.Background())
	go srv.Run(ctx)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() { ts.Close(); cancel() })
	return srv, ts
}

func post(ctx context.Context, url, body string) (*http.Response, error) {
	req, _ := http.NewRequestWithContext(ctx, "POST", url+"/v1/chat", strings.NewReader(body))
	req.Header.Set("Accept", "text/event-stream")
	return http.DefaultClient.Do(req)
}

func TestStreamsOneEventPerToken(t *testing.T) {
	_, ts := start(t)
	resp, err := post(context.Background(), ts.URL, `{"max_tokens": 5, "messages": [{"role": "user", "content": "¿Por qué el cielo es azul?"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type %q", ct)
	}
	var events []string
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			events = append(events, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(events) != 6 || events[5] != "[DONE]" {
		t.Fatalf("eventos %q: want 5 tokens y [DONE]", events)
	}
	if events[0] != `{"delta":"La "}` {
		t.Fatalf("primer evento %s", events[0])
	}
}

// Si el cliente se va, el pedido se cancela y su KV cache se libera.
func TestDisconnectFreesTheKVCache(t *testing.T) {
	srv, ts := start(t)
	ctx, cancel := context.WithCancel(context.Background())
	resp, err := post(ctx, ts.URL, `{"messages": [{"role": "user", "content": "una respuesta larga"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() && !strings.HasPrefix(sc.Text(), "data: ") {
	}
	if srv.State().Running != 1 || srv.State().KVUsed == 0 {
		t.Fatalf("el pedido tenía que estar corriendo: %+v", srv.State())
	}
	cancel()
	resp.Body.Close()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if st := srv.State(); st.Running == 0 && st.KVAlloc == 0 && st.Canceled == 1 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("el pedido sigue ocupando el motor: %+v", srv.State())
}
