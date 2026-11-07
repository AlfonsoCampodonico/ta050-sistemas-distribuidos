# TA050 · Clases 16–22: entrenar y servir modelos

| Clase | Tema | Paper ancla |
| --- | --- | --- |
| 16 | Fundamentos del entrenamiento distribuido | Megatron-LM (SC 2021) |
| 17 | Entrenamiento a escala de frontera | Llama 3, DeepSeek-V3, DiLoCo |
| 18 | Cómo se sirve un modelo | vLLM / PagedAttention (SOSP 2023) |
| 19 | Del motor al clúster | Mooncake (FAST 2025) |
| 20 | Muchos modelos, pocas GPUs | ServerlessLLM (OSDI 2024), S-LoRA (MLSys 2024) |
| 21 | Repartir un clúster de GPUs | Philly (ATC 2019), Tiresias (NSDI 2019) |
| 22 | Entrenar con lo que se sirve | HybridFlow (EuroSys 2025) |

## Contenido

- `repos/entrenamiento-distribuido`: clases 16 y 17 (Go, sin dependencias). `go test ./...`, `go run ./cmd/demo <allreduce|train|pipeline|4d|moe|fallas|diloco>`.
- `repos/servir-tokens`: clases 18, 19 y 20 (Go, sin dependencias). `go test ./...`, `go run ./cmd/sim <batching|memoria|cow|router|desagregado|especulativo|arranque|lora>`, `go run ./cmd/servidor`, `go run ./cmd/proxy -buffer`.
- `repos/planificar-gpus`: clase 21 (Go, sin dependencias). `go test ./...`, `go run ./cmd/sim <carga|traba|gang|localidad|duraciones>`.
- `repos/post-entrenamiento`: clase 22 (Go, sin dependencias). `go test ./...`, `go run ./cmd/sim <cola|ubicacion|asinc|parciales|aprender>`.
- `slides/clase-NN/`: cada deck en el formato de Claude Slides (`deck.json` + `slides/*.html`, una `<section>` por slide, notas del presentador en `<aside>`). `index.html` es el visor: muestra una slide por vez (← → o deslizar para avanzar, N notas, G todas, F pantalla completa) y guarda la slide actual en la URL (`#5`). El modo presentación está en `slides/present.js`.
- Sitio: https://alfonsocampodonico.github.io/ta050-sistemas-distribuidos/
- `simuladores/`: Entrenamiento a escala (Clase 17), Motor de inferencia (Clase 18), Clúster de inferencia (Clase 19), Flota de GPUs (Clase 20), Planificador de GPUs (Clase 21) y Bucle de RL (Clase 22). HTML autocontenido; sólo carga las fuentes de Google Fonts.
- `apuntes/`: las notas de cada clase en Markdown, exportadas de los docs.
- `herramientas/`: los generadores de los decks.
  - `gen17.py`, `gen19.py`, `gen20.py`, `gen21.py` y `gen22.py` regeneran esos decks en `herramientas/out/`.
  - `visor.py <out>/clase-NN/project slides/clase-NN` arma la carpeta del deck con su `index.html` y le aplica el estilo de la materia. `visor.py slides/clase-NN slides/clase-NN` rearma un deck que ya está en `slides/` (los decks 16 y 18 no tienen generador).
  - `estilo.py` pasa cada deck al estilo de la materia en oscuro: las fuentes de fiubata050.github.io (Chakra Petch, Rubik, IBM Plex Mono), un turquesa apagado como acento, separadores por sección y número de slide. Se puede correr más de una vez.
  - `slidekit.py` y `cover.py` son las plantillas que usan los generadores; `estilo.py` convierte su salida.
  - `restyle.py` es el script viejo que pasó los decks 16 y 18 al estilo anterior (necesita `beautifulsoup4`).
  - `render/render.py <deck> <out>` saca PNGs de cada slide con Playwright.
