package kvcache

// Paged implementa la idea central de PagedAttention (Kwon et al., SOSP 2023):
// la memoria se parte en bloques físicos de blockSize tokens, y cada secuencia
// tiene una tabla de bloques que traduce sus bloques lógicos (0, 1, 2, …) a
// bloques físicos que pueden estar en cualquier lugar. Es memoria virtual con
// paginación: bloques = páginas, tokens = bytes, secuencias = procesos.
//
// Como un bloque se asigna recién cuando el anterior se llena, el desperdicio
// por secuencia es menor a un bloque, y no hay fragmentación externa: cualquier
// bloque libre sirve.
//
// Además los bloques tienen contador de referencias, así que varias secuencias
// pueden compartir el prompt (muestreo paralelo, beam search) y se copia un
// bloque recién cuando alguien quiere escribir en él: copy-on-write.
type Paged struct {
	blockSize int
	numBlocks int
	free      []int // pila de bloques físicos libres
	refs      []int // referencias a cada bloque físico
	tables    map[int]*table
	copies    int
	tokens    int // suma de los tokens de todas las tablas
	extra     int // referencias de más: suma de (refs-1) sobre los bloques compartidos
}

type table struct {
	blocks []int // bloque lógico i → bloque físico blocks[i]
	tokens int
}

// NewPaged crea un administrador con capacity slots partidos en bloques de
// blockSize tokens (vLLM usa 16 por defecto).
func NewPaged(capacity, blockSize int) *Paged {
	n := capacity / blockSize
	p := &Paged{blockSize: blockSize, numBlocks: n, refs: make([]int, n), tables: map[int]*table{}}
	for b := n - 1; b >= 0; b-- {
		p.free = append(p.free, b)
	}
	return p
}

func (p *Paged) blocksFor(tokens int) int { return (tokens + p.blockSize - 1) / p.blockSize }

func (p *Paged) alloc() int {
	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.refs[b] = 1
	return b
}

func (p *Paged) release(b int) {
	if p.refs[b] > 1 {
		p.extra--
	}
	p.refs[b]--
	if p.refs[b] == 0 {
		p.free = append(p.free, b)
	}
}

// Admit ignora maxTokens: no hace falta reservar para el futuro.
func (p *Paged) Admit(id, tokens, _ int) bool {
	need := p.blocksFor(tokens)
	if need > len(p.free) {
		return false
	}
	t := &table{tokens: tokens}
	for range need {
		t.blocks = append(t.blocks, p.alloc())
	}
	p.tables[id] = t
	p.tokens += tokens
	return true
}

func (p *Paged) Append(id int) bool {
	t := p.tables[id]
	if t.tokens%p.blockSize == 0 {
		// El último bloque está lleno (o no hay ninguno): hace falta otro.
		if len(p.free) == 0 {
			return false
		}
		t.blocks = append(t.blocks, p.alloc())
	} else if last := t.blocks[len(t.blocks)-1]; p.refs[last] > 1 {
		// El último bloque está compartido y vamos a escribir en él:
		// copy-on-write. Se copia ese bloque y sólo ese.
		if len(p.free) == 0 {
			return false
		}
		p.refs[last]--
		p.extra--
		t.blocks[len(t.blocks)-1] = p.alloc()
		p.copies++
	}
	t.tokens++
	p.tokens++
	return true
}

func (p *Paged) Free(id int) {
	t, ok := p.tables[id]
	if !ok {
		return
	}
	for _, b := range t.blocks {
		p.release(b)
	}
	p.tokens -= t.tokens
	delete(p.tables, id)
}

// Fork crea la secuencia child compartiendo todos los bloques de parent, sin
// copiar nada: es lo que hace vLLM con las n muestras de un mismo prompt.
func (p *Paged) Fork(parent, child int) {
	src := p.tables[parent]
	t := &table{blocks: append([]int(nil), src.blocks...), tokens: src.tokens}
	for _, b := range t.blocks {
		p.refs[b]++
	}
	p.extra += len(t.blocks)
	p.tokens += t.tokens
	p.tables[child] = t
}

// BlockTable devuelve la tabla de bloques de la secuencia.
func (p *Paged) BlockTable(id int) []int {
	return append([]int(nil), p.tables[id].blocks...)
}

// Copies es la cantidad de bloques copiados por copy-on-write.
func (p *Paged) Copies() int { return p.copies }

// FreeBlocks es la cantidad de bloques físicos libres.
func (p *Paged) FreeBlocks() int { return len(p.free) }

// BlockSize devuelve el tamaño de bloque en tokens.
func (p *Paged) BlockSize() int { return p.blockSize }

func (p *Paged) Stats() Stats {
	st := Stats{
		Capacity:  p.numBlocks * p.blockSize,
		Allocated: (p.numBlocks - len(p.free)) * p.blockSize,
	}
	if p.extra == 0 {
		// Nada compartido: cada token de cada tabla ocupa un slot propio.
		st.Used = p.tokens
		return st
	}
	// Un bloque compartido cuenta una sola vez: con lo que guarda la
	// secuencia que más lo llenó.
	filled := map[int]int{}
	for _, t := range p.tables {
		for i, b := range t.blocks {
			filled[b] = max(filled[b], min(p.blockSize, t.tokens-i*p.blockSize))
		}
	}
	for _, f := range filled {
		st.Used += f
	}
	return st
}
