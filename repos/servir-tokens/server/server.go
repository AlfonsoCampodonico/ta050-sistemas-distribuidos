// Package server pone el motor detrás de HTTP: un endpoint que recibe un
// pedido de chat y devuelve los tokens como server-sent events (SSE), uno por
// evento, a medida que el motor los genera.
//
// El motor corre en una sola goroutine (Run) que hace una iteración, duerme lo
// que esa iteración "dura" y reparte los tokens a los streams de cada pedido.
// Cada pedido HTTP vive en su propia goroutine y lee de un canal. Si el
// cliente cierra la conexión, el handler cancela el pedido en el motor y su KV
// cache se libera: si no, el servidor seguiría generando tokens que nadie lee.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/fiubaTA050/servir-tokens/engine"
)

// Answer es la respuesta que "genera" el modelo, partida en tokens.
var Answer = strings.SplitAfter("La luz del sol choca con las moléculas del aire y se dispersa. "+
	"La luz azul, de longitud de onda más corta, se dispersa mucho más que la roja, "+
	"así que llega a nuestros ojos desde todas las direcciones del cielo. "+
	"Por eso el cielo se ve azul, y al atardecer, cuando la luz atraviesa más aire, se ve rojizo.", " ")

// Server es el servidor HTTP.
type Server struct {
	mu       sync.Mutex
	eng      *engine.Engine
	streams  map[int]chan engine.Token
	pending  []engine.Token
	nextID   int
	canceled int
	wake     chan struct{}
	speed    float64 // ms reales por ms del motor
}

// New crea un servidor sobre el motor. speed escala el tiempo: 1 es tiempo
// real, 0.01 es cien veces más rápido (para los tests).
func New(eng *engine.Engine, speed float64) *Server {
	s := &Server{eng: eng, streams: map[int]chan engine.Token{}, wake: make(chan struct{}, 1), speed: speed}
	eng.OnToken(func(t engine.Token) { s.pending = append(s.pending, t) })
	return s
}

// Handler devuelve las rutas: POST /v1/chat y GET /estado.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat", s.chat)
	mux.HandleFunc("GET /estado", s.status)
	return mux
}

// Run es el loop del motor. Corre hasta que ctx se cancela.
func (s *Server) Run(ctx context.Context) {
	for {
		s.mu.Lock()
		dt, idle := s.eng.Step()
		tokens := s.pending
		s.pending = nil
		s.mu.Unlock()

		if idle {
			select {
			case <-s.wake:
				continue
			case <-ctx.Done():
				return
			}
		}
		// La iteración "dura" dt: recién al terminar existen sus tokens.
		select {
		case <-time.After(time.Duration(dt * s.speed * float64(time.Millisecond))):
		case <-ctx.Done():
			return
		}
		s.mu.Lock()
		for _, t := range tokens {
			if ch, ok := s.streams[t.Request]; ok {
				ch <- t // el canal tiene lugar para todos los tokens del pedido
			}
		}
		s.mu.Unlock()
	}
}

type chatRequest struct {
	MaxTokens int `json:"max_tokens"`
	Messages  []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "JSON inválido: "+err.Error(), http.StatusBadRequest)
		return
	}
	// Una aproximación al tokenizer: unas 1,3 piezas por palabra, más los
	// tokens especiales del chat template.
	prompt := 8
	for _, m := range req.Messages {
		prompt += 4 + len(strings.Fields(m.Content))*13/10
	}
	output := req.MaxTokens
	if output <= 0 || output > len(Answer) {
		output = len(Answer)
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "el servidor no puede hacer streaming", http.StatusInternalServerError)
		return
	}

	ch := make(chan engine.Token, output)
	s.mu.Lock()
	id := s.nextID
	s.nextID++
	err := s.eng.Submit(engine.Request{ID: id, Arrival: s.eng.Now(), Prompt: prompt, Output: output})
	if err == nil {
		s.streams[id] = ch
	}
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	select {
	case s.wake <- struct{}{}:
	default:
	}
	defer func() {
		s.mu.Lock()
		delete(s.streams, id)
		if s.eng.Cancel(id) {
			s.canceled++
		}
		s.mu.Unlock()
	}()

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no") // que nginx no junte el stream en un solo bloque
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return // el cliente se fue: el defer cancela el pedido y libera su KV cache
		case t := <-ch:
			delta, _ := json.Marshal(map[string]string{"delta": Answer[t.Index%len(Answer)]})
			fmt.Fprintf(w, "data: %s\n\n", delta)
			if t.Last {
				fmt.Fprint(w, "data: [DONE]\n\n")
			}
			flusher.Flush()
			if t.Last {
				return
			}
		}
	}
}

// Status es el estado del motor que devuelve GET /estado.
type Status struct {
	Running   int     `json:"en_el_lote"`
	Waiting   int     `json:"en_la_cola"`
	KVUsed    int     `json:"kv_tokens_usados"`
	KVAlloc   int     `json:"kv_slots_asignados"`
	KVCap     int     `json:"kv_capacidad"`
	KVUtil    float64 `json:"kv_utilizacion"`
	Canceled  int     `json:"cancelados"`
	ClockSecs float64 `json:"reloj_s"`
}

// State devuelve el estado actual.
func (s *Server) State() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	kv := s.eng.KV()
	return Status{
		Running: s.eng.Running(), Waiting: s.eng.Waiting(),
		KVUsed: kv.Used, KVAlloc: kv.Allocated, KVCap: kv.Capacity, KVUtil: kv.Utilization(),
		Canceled: s.canceled, ClockSecs: s.eng.Now() / 1000,
	}
}

func (s *Server) status(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	enc.Encode(s.State())
}
