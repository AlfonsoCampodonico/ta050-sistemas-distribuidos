// proxy se pone delante del servidor para mostrar el problema más común del
// streaming: un proxy que junta la respuesta antes de reenviarla.
//
//	go run ./cmd/proxy               # reenvía cada evento apenas llega
//	go run ./cmd/proxy -buffer       # junta todo y lo manda al final: un bloque tardío
//
// Con -buffer, el cliente no ve nada hasta que termina la respuesta, aunque el
// servidor haya mandado cada token a tiempo. En nginx el equivalente es
// proxy_buffering on; se desactiva con proxy_buffering off o con el header
// X-Accel-Buffering: no que manda el servidor.
package main

import (
	"flag"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
)

func main() {
	listen := flag.String("addr", ":8081", "dirección donde escuchar")
	target := flag.String("target", "http://localhost:8080", "servidor de atrás")
	buffer := flag.Bool("buffer", false, "juntar la respuesta entera antes de reenviarla")
	flag.Parse()

	u, err := url.Parse(*target)
	if err != nil {
		log.Fatal(err)
	}
	var h http.Handler
	if *buffer {
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			req, _ := http.NewRequestWithContext(r.Context(), r.Method, u.JoinPath(r.URL.Path).String(), r.Body)
			req.Header = r.Header.Clone()
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body) // acá se pierde el streaming
			for k, v := range resp.Header {
				w.Header()[k] = v
			}
			w.WriteHeader(resp.StatusCode)
			w.Write(body)
		})
	} else {
		p := httputil.NewSingleHostReverseProxy(u)
		p.FlushInterval = -1 // reenviar cada escritura apenas llega
		h = p
	}
	log.Printf("proxy en %s → %s (buffer=%v)", *listen, *target, *buffer)
	log.Fatal(http.ListenAndServe(*listen, h))
}
