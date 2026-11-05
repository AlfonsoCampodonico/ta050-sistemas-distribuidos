// Package kvcache modela la memoria de GPU que guarda el KV cache: para cada
// token de cada secuencia en curso, las keys y values de todas las capas.
//
// Hay dos administradores con la misma interfaz:
//
//   - Contiguous reserva, al admitir una secuencia, un tramo contiguo para su
//     longitud máxima, como los sistemas previos a vLLM (Orca, FasterTransformer).
//   - Paged parte la memoria en bloques de tamaño fijo y asigna un bloque nuevo
//     sólo cuando el último se llena, como PagedAttention: bloques = páginas,
//     tokens = bytes, secuencias = procesos.
//
// La memoria se mide en "slots": un slot guarda el KV de un token.
package kvcache

// Allocator administra la memoria del KV cache.
type Allocator interface {
	// Admit reserva memoria para una secuencia que ya tiene tokens tokens
	// (el prompt, o prompt + lo generado si se la vuelve a admitir) y que
	// puede llegar a maxTokens. Devuelve false si no entra.
	Admit(id, tokens, maxTokens int) bool
	// Append hace lugar para un token más de la secuencia. Devuelve false si
	// no hay memoria.
	Append(id int) bool
	// Free libera toda la memoria de la secuencia.
	Free(id int)
	// Stats describe el uso de la memoria en este momento.
	Stats() Stats
}

// Stats describe la memoria del KV cache, en slots de un token.
type Stats struct {
	Capacity  int // slots que hay en total
	Allocated int // slots asignados a alguna secuencia
	Used      int // slots que guardan un token de verdad
}

// Utilization es la fracción de la memoria asignada que guarda tokens de
// verdad. El resto es desperdicio: reservado para el futuro o fragmentación
// interna. vLLM mide que los sistemas previos usan entre 20,4% y 38,2%.
func (s Stats) Utilization() float64 {
	if s.Allocated == 0 {
		return 0
	}
	return float64(s.Used) / float64(s.Allocated)
}
