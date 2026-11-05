# Post-entrenamiento con RL — demo de clase

Material práctico de la clase **22 — Entrenar con lo que se sirve** de TA050
Sistemas Distribuidos I (FIUBA). Es una simulación en Go, en tiempo virtual,
del bucle con el que se post-entrenan los modelos de razonamiento: generar
respuestas, puntuarlas con un verificador y actualizar los pesos. Generar es
una carga de serving (Clase 18) y actualizar es una carga de entrenamiento
(Clase 16), y las dos están en el mismo bucle. Lo que se simula es lo que
importa para la materia: **quién espera a quién, cuánto cuesta llevar los
pesos nuevos a donde se genera, y con qué pesos se generó cada muestra que se
entrena**.

Paper de la clase: Sheng et al., *HybridFlow: A Flexible and Efficient RLHF
Framework*, EuroSys 2025 ([arXiv:2409.19256](https://arxiv.org/abs/2409.19256)),
el paper de verl. Complementarios: DeepSeek-R1
([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)), Kimi k1.5
([arXiv:2501.12599](https://arxiv.org/abs/2501.12599)) y AReaL
([arXiv:2505.24298](https://arxiv.org/abs/2505.24298)).

## Requisitos

Go 1.24 o superior. No hay dependencias externas.

```bash
go test ./...
```

## Estructura

| Paquete | Qué modela | Concepto de la clase |
|---|---|---|
| `tarea` | Un "modelo" que decide cuánto pensar: más largo acierta más, pero pasar el tope vale 0. GRPO con ventajas por grupo | RL con recompensas verificables (R1); las respuestas se alargan al aprender; corrección por importancia |
| `pesos` | Cuánto cuesta llevar los pesos a donde se genera | Broadcast ingenuo vs en anillo (Clase 16); volumen de la transición de layout (HybridFlow, tabla 2) |
| `bucle` | El bucle completo en tiempo virtual: GPUs de rollout con batching continuo, un trainer, la sincronización de pesos | Colocado vs separado; sincrónico vs asincrónico con staleness acotada; rollouts parciales e interrumpibles |
| `cmd/sim` | Las tablas de la clase | — |

### El modelo de costo

Cada GPU de rollout es un motor con batching continuo, como el de la Clase 18:
una iteración de decode cuesta `Base + PorSeq·secuencias` milisegundos (20 +
0,2·n) y produce un token para cada secuencia del lote, hasta 64. Entrenar
cuesta `Entrenar` ms·GPU por token (0,2). Recalcular el KV cache de una
respuesta a medias cuesta 0,05 ms por token. Los números son ilustrativos; lo
que importa es la forma: generar está limitado por el decode, y una GPU con
una sola secuencia paga casi lo mismo que una con 64.

### La tarea

`log(largo) ~ N(Mu, 0,6²)`: la política elige cuánto pensar, y la dispersión es
la cola larga de los rollouts. Una respuesta de L tokens acierta con
probabilidad `1 − exp(−L/1400)`; si pasa de 8.192 tokens se corta y vale 0. Al
empezar la mediana es de 300 tokens y la recompensa esperada 0,22; la mejor
política tiene una mediana de unos 3.300 tokens y recompensa 0,80, con un 6,5%
de respuestas cortadas. El óptimo está en el medio, y por eso la staleness se
nota: un gradiente calculado con la política de hace varios pasos sigue
empujando hacia respuestas más largas cuando ya se pasó del óptimo.

Cada paso son 32 problemas × 8 respuestas (GRPO: la ventaja de cada respuesta
es su recompensa menos la media de su grupo, dividido por el desvío).

## Demo 1 — la cola larga

```bash
go run ./cmd/sim cola
```

Un paso sincrónico, 256 respuestas en 8 GPUs. Con la política inicial la
respuesta mediana tiene 305 tokens y la más larga 1.388; con la ya entrenada,
3.478 y el tope. La primera GPU termina mucho antes que la última, y el paso
dura lo que la última: entre 75% y 86% del GPU·tiempo de generación se pierde
en lugares vacíos del lote.

## Demo 2 — dónde va cada trabajo

```bash
go run ./cmd/sim ubicacion
go run ./cmd/sim ubicacion -pesos 140
```

Colocado (verl): las 8 GPUs generan y después entrenan. Separado y
sincrónico: mientras unas generan, las otras esperan, y al revés. Con 8 GPUs
el colocado gana. Abajo, cuánto cuesta mover 140 GB de pesos (un 70B en bf16,
el ejemplo de HybridFlow §2.3) a 200 Gb/s: ingenuo crece con la cantidad de
GPUs de rollout; en anillo, casi no. Y los volúmenes de la tabla 2 de
HybridFlow para pasar del layout de entrenamiento al de generación.

## Demo 3 — asincrónico con staleness acotada

```bash
go run ./cmd/sim asinc
go run ./cmd/sim asinc -rollout 7
```

6 GPUs generan y 2 entrenan. Con η = 0 es sincrónico. Con η > 0 los rollouts
siguen generando con pesos de hasta η versiones atrás (la ecuación 3 de AReaL:
el controlador no deja empezar una respuesta si eso adelanta los rollouts más
de η versiones). Mirá cómo baja la GPU ociosa y sube el throughput, y cómo
crece la staleness. El máximo pasa a η: una respuesta larga que empezó con
pesos viejos termina tarde. Con `-rollout 7` el trainer no da abasto y sin
límite la staleness crece.

## Demo 4 — rollouts parciales e interrumpibles

```bash
go run ./cmd/sim parciales
```

Rollouts parciales (Kimi k1.5): la generación termina cuando hay un lote
completo, y lo que quedó a medias sigue en la vuelta siguiente, con los pesos
nuevos. Interrumpibles (AReaL): cuando llegan pesos nuevos, el worker descarta
el KV cache, lo recalcula con los pesos nuevos y sigue; sin eso, tiene que
terminar lo que tiene antes de cargarlos.

## Demo 5 — qué le hace la staleness al aprendizaje

```bash
go run ./cmd/sim aprender
go run ./cmd/sim aprender -lr 0.3
```

La recompensa al terminar para cada η, sin y con corrección por importancia
(cada muestra pesa π_actual/π_behav, truncado en 2: la idea del objetivo
desacoplado de AReaL). Sin corrección, η = 1 alcanza y η ≥ 2 empieza a costar;
sin límite, la política se pasa del tope, todas las respuestas se cortan, todas
valen 0 y GRPO se queda sin señal. Con corrección se recupera la recompensa, a
cambio de algunos pasos más.

## Ejercicios

1. **Descartar lo viejo.** Hacé que el trainer descarte las respuestas con
   staleness mayor que η en lugar de usarlas. ¿Qué respuestas se descartan
   más? ¿Qué le hace eso al largo medio que aprende la política?
2. **Corrección con recorte.** Reemplazá el peso truncado por el objetivo de
   PPO con recorte (ε = 0,2) respecto de la política próxima. ¿Cambia algo con
   un solo paso de gradiente por lote? ¿Y con varios?
3. **Ratio adaptativo.** Con pools separados, mové GPUs de rollout a trainer (o
   al revés) según cuál de los dos está más ocioso. ¿Le gana al mejor ratio
   fijo a medida que las respuestas se alargan?
4. **Fallas.** Hacé que una GPU de rollout se caiga en la mitad de un paso. En
   el sincrónico, ¿qué pasa con el paso? En el asincrónico, ¿qué hay que hacer
   con las respuestas a medias? (Pista: son idempotentes, como un map de
   MapReduce.)
5. **Muestreo dinámico.** Un grupo en el que todos aciertan o todos fallan no
   da señal. Hacé que el controlador genere más grupos hasta tener B con
   señal. ¿Cuánto cuesta en tiempo de rollout al principio y al final?

## Referencias

- G. Sheng et al. *HybridFlow: A Flexible and Efficient RLHF Framework*.
  EuroSys 2025. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256) ·
  [verl](https://github.com/volcengine/verl)
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning*. 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. 2025.
  [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
- W. Fu et al. *AReaL: A Large-Scale Asynchronous Reinforcement Learning
  System for Language Reasoning*. 2025. [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)
- J. Hu et al. *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF
  Framework*. [arXiv:2405.11143](https://arxiv.org/abs/2405.11143)
- B. Wu et al. *LlamaRL: A Distributed Asynchronous Reinforcement Learning
  Framework for Efficient Large-scale LLM Training*.
  [arXiv:2505.24034](https://arxiv.org/abs/2505.24034)
- Q. Ho et al. *More Effective Distributed ML via a Stale Synchronous Parallel
  Parameter Server*. NIPS 2013.
