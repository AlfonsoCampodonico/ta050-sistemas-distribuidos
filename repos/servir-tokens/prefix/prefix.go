// Package prefix es un caché de prefijos de KV cache: guarda qué bloques de
// prompt ya fueron calculados para no volver a hacer su prefill.
//
// Cada bloque se identifica por un hash encadenado: el hash del bloque i
// depende de su contenido y del hash del bloque i-1 (Mooncake §4.1, y así lo
// hace también vLLM). Dos prompts comparten el bloque i sólo si comparten
// todo lo anterior, así que alcanza con buscar los hashes en orden y frenar
// en el primero que falta: eso es lo mismo que recorrer un árbol de prefijos.
//
// El caché tiene capacidad fija en bloques y desaloja el menos usado
// recientemente (LRU), como el pool de Mooncake (§4.2) y RadixAttention.
package prefix

import "container/list"

// Hash encadena el hash del bloque anterior con el contenido de este.
func Hash(prev, content uint64) uint64 {
	// FNV-1a sobre los 16 bytes; alcanza para un simulador.
	h := uint64(14695981039346656037)
	for _, v := range [2]uint64{prev, content} {
		for i := range 8 {
			h ^= (v >> (8 * i)) & 0xff
			h *= 1099511628211
		}
	}
	return h
}

// Cache es un conjunto de bloques con desalojo LRU.
type Cache struct {
	capacity int
	order    *list.List // frente = usado más recientemente
	blocks   map[uint64]*list.Element
}

// New crea un caché de capacity bloques.
func New(capacity int) *Cache {
	return &Cache{capacity: capacity, order: list.New(), blocks: map[uint64]*list.Element{}}
}

// Peek cuenta cuántos bloques del principio de hashes ya están, sin tocar el
// orden LRU. Es lo que usa un router para estimar sin modificar nada.
func (c *Cache) Peek(hashes []uint64) int {
	n := 0
	for _, h := range hashes {
		if _, ok := c.blocks[h]; !ok {
			break
		}
		n++
	}
	return n
}

// Match es Peek, pero marca como recién usados los bloques encontrados.
func (c *Cache) Match(hashes []uint64) int {
	n := c.Peek(hashes)
	for _, h := range hashes[:n] {
		c.order.MoveToFront(c.blocks[h])
	}
	return n
}

// Insert agrega los bloques (o los marca como recién usados) y desaloja los
// menos usados si se pasa de capacidad.
func (c *Cache) Insert(hashes []uint64) {
	for _, h := range hashes {
		if el, ok := c.blocks[h]; ok {
			c.order.MoveToFront(el)
			continue
		}
		c.blocks[h] = c.order.PushFront(h)
	}
	for c.order.Len() > c.capacity {
		el := c.order.Back()
		c.order.Remove(el)
		delete(c.blocks, el.Value.(uint64))
	}
}

// Len es la cantidad de bloques guardados.
func (c *Cache) Len() int { return c.order.Len() }
