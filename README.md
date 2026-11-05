# TA050 · Clases 16–19: entrenar y servir modelos

| Clase | Tema | Paper ancla |
| --- | --- | --- |
| 16 | Fundamentos del entrenamiento distribuido | Megatron-LM (SC 2021) |
| 17 | Entrenamiento a escala de frontera | Llama 3, DeepSeek-V3, DiLoCo |
| 18 | Cómo se sirve un modelo | vLLM / PagedAttention (SOSP 2023) |
| 19 | Del motor al clúster | Mooncake (FAST 2025) |

## Contenido

- `repos/entrenamiento-distribuido`: clases 16 y 17 (Go, sin dependencias). `go test ./...`, `go run ./cmd/demo <allreduce|train|pipeline|4d|moe|fallas|diloco>`.
- `repos/servir-tokens`: clases 18 y 19 (Go, sin dependencias). `go test ./...`, `go run ./cmd/sim <batching|memoria|cow|router|desagregado|especulativo>`, `go run ./cmd/servidor`, `go run ./cmd/proxy -buffer`.
- `slides/clase-NN/`: cada deck en el formato de Claude Slides (`deck.json` + `slides/*.html`, una `<section>` por slide, notas del presentador en `<aside>`). `index.html` es un visor local: abrilo en el navegador y tildá "notas" para ver las notas.
- `simuladores/`: Motor de inferencia (Clase 18), Clúster de inferencia (Clase 19) y Entrenamiento a escala (Clase 17). HTML autocontenido; sólo carga las fuentes de Google Fonts.
- `apuntes/`: las notas de cada clase en Markdown, exportadas de los docs.
- `herramientas/`: los generadores de los decks.
  - `gen17.py` y `gen19.py` regeneran los decks 17 y 19 en `herramientas/out/`.
  - `slidekit.py` y `cover.py` tienen las plantillas del estilo (Geist, oscuro primero).
  - `restyle.py` es el script que pasó los decks 16 y 18 al estilo nuevo (necesita `beautifulsoup4`).
  - `render/render.py <deck> <out>` saca PNGs de cada slide con Playwright.
