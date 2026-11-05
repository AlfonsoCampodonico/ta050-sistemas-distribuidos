# Servir tokens — demo de clase

Material práctico de las clases **18 — Cómo se sirve un modelo de lenguaje** y
**19 — Del motor al clúster** de TA050 Sistemas Distribuidos I (FIUBA). Es un
servidor de inferencia de juguete en Go: el modelo no existe, pero el
scheduler, el KV cache, el batching, la preemption, el caché de prefijos, el
router y el streaming son los de un servicio de verdad.

Papers de las clases: Kwon et al., *Efficient Memory Management for Large
Language Model Serving with PagedAttention*, SOSP 2023
([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)), y Qin et al.,
*Mooncake: Trading More Storage for Less Computation*, FAST 2025
([USENIX](https://www.usenix.org/conference/fast25/presentation/qin)).

## Requisitos

Go 1.24 o superior. No hay dependencias externas.

```bash
go test ./...
```

## Estructura

| Paquete | Qué modela | Concepto de la clase |
|---|---|---|
| `kvcache` | Memoria del KV cache: reserva contigua y bloques paginados con copy-on-write | Desperdicio por reserva y fragmentación; tabla de bloques; compartir el prompt |
| `engine` | El scheduler: batching estático o continuo, admisión FCFS, desalojo con recomputación, prefill en pedazos, caché de prefijos | Iteration-level scheduling (Orca); el KV cache decide el tamaño del lote; chunked prefill (Sarathi-Serve) |
| `prefix` | Caché de prefijos: hashes encadenados por bloque y desalojo LRU | Reusar el KV cache entre pedidos (Mooncake, RadixAttention) |
| `cluster` | Varias réplicas detrás de un router, colocadas o con prefill y decode separados | Routing consciente del prefijo; desagregación (DistServe, Mooncake); goodput |
| `server` | HTTP + server-sent events, cancelación cuando el cliente se va | Streaming de tokens; desconexiones |
| `cmd/sim` | Comparaciones en tiempo virtual | Las tablas de la clase |
| `cmd/servidor` | El motor detrás de HTTP, en tiempo real | Demo en vivo |
| `cmd/proxy` | Un proxy que puede juntar la respuesta | El bug de streaming más común |

El motor es uno solo: `cmd/sim` avanza un reloj virtual lo que dura cada
iteración, y `cmd/servidor` duerme ese tiempo de verdad y manda los tokens por
SSE. Cada iteración cuesta `Base + PerSeq·secuencias + PerPrefillToken·tokens
de prefill + PerLoadToken·tokens traídos del pool` milisegundos
(`engine.Cost`): `Base` es leer todos los pesos una vez, que en decode se paga
igual con 1 secuencia que con 64. Por eso batchear sale casi gratis, y por eso
importa tanto cuántas secuencias entran. Traer un token de KV cache ya
calculado cuesta un décimo de recalcularlo. Los números son ilustrativos, del
orden de un modelo de 70B; lo que importa es la forma.

## Demo 1 — batching estático vs continuo (Orca)

```bash
go run ./cmd/sim batching
go run ./cmd/sim batching -rate 2
```

Con batching estático, el lote corre hasta que termina la respuesta más larga:
las cortas esperan, y como nadie ve nada hasta el final, el tiempo al primer
token es el de la respuesta entera. Con batching continuo, después de cada
iteración salen las que terminaron y entran las que esperan.

## Demo 2 — memoria contigua vs bloques (PagedAttention)

```bash
go run ./cmd/sim memoria
go run ./cmd/sim memoria -block 64
go run ./cmd/sim memoria -kv 8192
```

Tres administradores con la misma memoria y la misma carga:

- **contiguo, reserva máx** reserva 2048 tokens por pedido, como Orca (Max) en
  el paper. Mirá la columna *uso KV*: cae en el rango de 20,4% a 38,2% que el
  paper midió en sistemas reales.
- **contiguo, oráculo** reserva exactamente lo que cada respuesta va a usar,
  como si se supiera de antemano. No se puede en la práctica, y aun así sufre
  reserva y fragmentación externa.
- **paginado** asigna bloques de 16 tokens a medida que hacen falta.

Con más secuencias en el lote, cada lectura de los pesos sirve a más gente: el
throughput sube. Fijate también en la columna *desaloj*: con la memoria llena,
el paginado desaloja la secuencia más reciente y la recalcula después.

## Demo 3 — copy-on-write

```bash
go run ./cmd/sim cow
go run ./cmd/sim cow -n 8 -prompt 2000
```

Varias muestras del mismo prompt comparten sus bloques; se copia un bloque
recién cuando alguna quiere escribir en él.

## Demo 4 — streaming de verdad

En una terminal:

```bash
go run ./cmd/servidor -speed 2
```

En otra:

```bash
curl -N localhost:8080/v1/chat \
  -d '{"messages":[{"role":"user","content":"¿Por qué el cielo es azul?"}]}'
curl localhost:8080/estado
```

Después, con un proxy en el medio:

```bash
go run ./cmd/proxy                      # en :8081, reenvía cada evento
go run ./cmd/proxy -buffer -addr :8082  # en :8082, junta todo: un bloque al final
curl -N localhost:8081/v1/chat -d '{"messages":[{"role":"user","content":"hola"}]}'
curl -N localhost:8082/v1/chat -d '{"messages":[{"role":"user","content":"hola"}]}'
```

Con `-speed 2`, directo o por `:8081` llega un evento cada ~40 ms y la
respuesta de 60 tokens termina a los ~2,4 s. Por `:8082` no llega nada durante
~2,5 s y después llegan los 60 de golpe.

Y un cliente que se va a la mitad: cortá el `curl` con Ctrl-C y mirá
`/estado`. El pedido sale del lote y su KV cache se libera; si no, el servidor
seguiría generando tokens que nadie lee.

## Clase 19 — la carga de chat

Las demos de la Clase 19 usan `engine.Chat`: conversaciones de varios turnos
en las que cada turno manda la conversación entera otra vez, con un prompt de
sistema de 2.000 tokens compartido por las conversaciones de cada app. Con la
configuración por defecto son 1.000 conversaciones y 4.539 pedidos, con un
prompt promedio de unos 3.900 tokens y respuestas de unos 180: prompts largos,
respuestas cortas y casi todo el prompt repetido, como en la traza de Kimi
(8.590 y 182 tokens en promedio, Mooncake §4).

## Demo 5 — el router y el caché de prefijos

```bash
go run ./cmd/sim router
go run ./cmd/sim router -cache 256
go run ./cmd/sim router -rate 7
```

Cuatro réplicas, cada una con un caché de prefijos de 65.536 tokens, y cinco
formas de elegir réplica: en ronda, la de menor carga, siempre la misma por
conversación (sticky) y la que minimiza el tiempo estimado al primer token
contando lo que ya tiene en caché (el Conductor de Mooncake). La última fila
agrega un pool compartido por todas las réplicas: el KV cache de cualquier
prompt se puede traer de ahí en vez de recalcularlo.

Mirá tres cosas: cuánto prefill se evita, que sticky consigue el mismo caché
que el router consciente del prefijo pero se le cargan algunas réplicas
(desbalance, y el p99 del TTFT), y cuánto suma el pool.

## Demo 6 — prefill y decode: juntos, en pedazos o separados

```bash
go run ./cmd/sim desagregado
go run ./cmd/sim desagregado -chunk 1024
go run ./cmd/sim desagregado -slo-tbt 150
```

Cuatro réplicas en cinco configuraciones. Colocadas, un prompt largo frena a
todos los decodes de su lote: el TBT p99 se va de la mano. Con el prefill en
pedazos de 512 tokens (Sarathi-Serve) eso desaparece. Desagregadas, el decode
nunca espera un prefill, pero hay que elegir cuántas réplicas van a cada lado:
con esta carga, 1P+3D se queda corto de prefill y 3P+1D de decode. El SLO pide
el primer token en menos de 2 s y nunca más de 100 ms entre dos tokens;
goodput es cuántos pedidos por segundo lo cumplen (DistServe).

## Demo 7 — decodificación especulativa

```bash
go run ./cmd/sim especulativo
go run ./cmd/sim especulativo -c 0.2
```

La aceleración esperada de proponer k tokens con un modelo chico y
verificarlos todos en una pasada del grande, según la probabilidad α de que
cada uno se acepte (Leviathan et al., ICML 2023).

## Ejercicios

1. **Swapping.** El motor desaloja por recomputación. Agregá la alternativa de
   vLLM: copiar los bloques a "memoria de CPU" (otro `Paged`) y traerlos de
   vuelta, con un costo proporcional a los bloques. ¿Cuándo conviene cada una?
2. **Rate limiting por tokens.** Agregá al servidor un límite de tokens por
   minuto por cliente (header `Authorization`) que cuente los tokens de salida
   a medida que se mandan. ¿Por qué no alcanza con contar pedidos?
3. **Tamaño de bloque.** Corré `sim memoria` con `-block` 1, 4, 16, 64 y 256.
   El desperdicio por secuencia es menor a un bloque: ¿por qué vLLM no usa
   bloques de 1 token? (Pista: §7.2 del paper.)
4. **Compartir bloques físicos.** El caché de prefijos de `engine` ahorra
   cómputo pero no memoria: cada secuencia sigue teniendo sus propios
   bloques. Hacé que `Paged` reuse los bloques llenos de un prefijo (por su
   hash) con contador de referencias, como el prefix caching de vLLM. ¿Cuánto
   sube el lote promedio en `sim router`?
5. **Rechazo temprano.** Con mucha carga, todos los pedidos esperan y nadie
   cumple el SLO. Hacé que el router devuelva "429" cuando el TTFT estimado de
   la mejor réplica supera el SLO (Mooncake §7.2). ¿Qué pasa con el goodput
   con `-rate 10`?
6. **Hot spots.** Con sticky, una conversación muy larga satura su réplica.
   Agregá al router consciente del prefijo la replicación de Mooncake (§6.2):
   si el mejor prefijo está en otra réplica y traerlo es más barato que
   recalcularlo, copiarlo a la réplica elegida.
7. **Ratio adaptativo.** En `RunDisaggregated` el ratio P:D es fijo. Hacé que
   una réplica pase de prefill a decode (o al revés) cuando la cola de un lado
   crece mucho más que la del otro. ¿Le gana a la mejor configuración fija?

## Referencias

- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving
  with PagedAttention*. SOSP 2023.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- G.-I. Yu et al. *Orca: A Distributed Serving System for Transformer-Based
  Generative Models*. OSDI 2022.
  [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
- R. Qin et al. *Mooncake: Trading More Storage for Less Computation — A
  KVCache-centric Architecture for Serving LLM Chatbot*. FAST 2025.
  [USENIX](https://www.usenix.org/conference/fast25/presentation/qin)
- Y. Zhong et al. *DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized Large Language Model Serving*. OSDI 2024.
  [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- A. Agrawal et al. *Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve*. OSDI 2024. [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
- L. Zheng et al. *SGLang: Efficient Execution of Structured Language Model
  Programs*. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via
  Speculative Decoding*. ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- A. Campodónico. *What happens after you hit send*.
  [alfonsocampodonico.com](https://alfonsocampodonico.com/posts/what-happens-after-you-hit-send)
