# Clase 19 — Del motor al clúster

Nov 4, 2026 · @Alfonso Campodonico

## 1. De una réplica a un servicio

En la Clase 18 vimos vLLM por dentro: cómo una GPU arma lotes con batching continuo y administra el KV cache de los pedidos en curso con PagedAttention. Desde 2024 casi todo lo nuevo en serving pasa **afuera** de esa réplica. El motor sigue adentro de cada una; lo que cambia es lo que lo rodea.

El paper de hoy es **Mooncake** (Qin et al., FAST 2025, mejor paper), el sistema que sirve a Kimi, el chatbot de Moonshot AI. Que haya ganado en una conferencia de almacenamiento no es casualidad: su idea central es cambiar cómputo por almacenamiento.

La clase tiene cuatro partes:

| Parte | Qué cambia | Pregunta de sistemas |
| --- | --- | --- |
| Guardar | el KV cache no se tira al terminar el pedido: se guarda en GPU, DRAM del clúster o disco | caché distribuido |
| Encontrar | el router manda cada pedido a donde ya está su prefijo, sin sobrecargar a nadie | routing por contenido |
| Separar | prefill y decode tienen necesidades opuestas, y se separan en el tiempo o en el espacio | dimensionar pools |
| Escalar | los modelos de frontera son de expertos, y servirlos es un problema de paralelismo | Clase 17, otra vez |

Es la última de las cuatro clases: 16 y 17 sobre entrenar, 18 y 19 sobre servir. En las cuatro el estado que importa es grande y vive en la GPU; en serving ese estado es el KV cache.

## 2. La carga: cuánto se repite

Mooncake publicó una traza de una hora de pedidos reales de Kimi. El prompt promedio tiene **8.590 tokens** y la respuesta **182**: unas 47 veces más tokens de entrada que de salida. En la Clase 18 pensábamos en prompts de cientos de tokens; en un chatbot real, el prefill domina.

¿Por qué tan largos? El modelo no tiene memoria entre pedidos: cada turno manda el prompt de sistema, toda la historia y el mensaje nuevo, más documentos, herramientas o el contexto de un agente. Y **cada turno es el prefijo del siguiente**.

En producción, más o menos la mitad del prefill no hace falta calcularlo:

- **DeepSeek**, en las estadísticas de un día de servicio (27 al 28 de febrero de 2025): 608 mil millones de tokens de entrada, de los cuales 342 mil millones, el **56,3%**, salieron de su caché de KV en disco.
- **Mooncake**, sobre la traza de Kimi: con capacidad infinita se podría reusar hasta un 55% del KV cache. La popularidad es muy despareja: más de la mitad de los bloques no se vuelve a usar, y algunos (prompts de sistema, documentos populares) se usan decenas de miles de veces. Con 1.000 bloques de caché la tasa de aciertos es 34%; con 50.000, 55%.

La carga del repositorio (`engine.DefaultChat`) imita esto: 300 conversaciones de 5 turnos, 4 apps con un prompt de sistema de 2.000 tokens cada una. El prompt promedio da unos 3.900 tokens y la respuesta unos 178.

**Conexión con la materia.** Es la Clase 12, consistencia en caches, con dos diferencias amables: el contenido es inmutable (el KV cache de un prefijo dado es siempre el mismo, no hay invalidación) y fallar es barato (se recalcula). Lo difícil es dónde ponerlo y cómo encontrarlo.

## 3. El caché de prefijos

**Por qué prefijos.** El KV cache de un token depende de todos los tokens anteriores. Un bloque solo se puede reusar si todo lo que vino antes es igual: el mismo texto en otra posición no sirve.

**Hashes encadenados.** El prompt se parte en bloques de tokens (Mooncake usa 512; el repositorio, 64). El identificador de cada bloque es el hash de sus tokens **y del identificador del bloque anterior**:

> h₀ = hash(bloque₀), hᵢ = hash(hᵢ₋₁, bloqueᵢ)

Si dos prompts tienen el mismo h₃, tienen los mismos tokens en los bloques 0 a 3. Dos conversaciones de la misma app comparten los bloques del prompt de sistema; desde el primer bloque distinto los hashes divergen y no vuelven a coincidir.

**Buscar y desalojar.** Para un prompt nuevo se recorren sus hashes en orden y se frena en el primero que no está: es recorrer un árbol de prefijos, y SGLang lo implementa literalmente como un radix tree (RadixAttention). Para desalojar, LRU: los bloques del prompt de sistema se tocan todo el tiempo y quedan; los de conversaciones abandonadas se van. Es el mismo truco que los bloques de PagedAttention de la Clase 18, pero entre pedidos distintos.

En el repositorio, el paquete `prefix` es eso en unas 80 líneas: `Hash(prev, content)`, y un `Cache` con `Match` (cuántos bloques del prompt ya están), `Insert` y LRU.

**Niveles.** El subtítulo de Mooncake es literal: *trading more storage for less computation*. La memoria de la GPU alcanza para el lote en curso, pero las máquinas tienen mucha DRAM de CPU y SSDs casi ociosos.

| Nivel | Qué es | Quién lo usa |
| --- | --- | --- |
| HBM de la GPU | el lote en curso y los prefijos calientes | todos |
| DRAM del clúster | un pool distribuido, accesible por RDMA | Mooncake, LMCache |
| SSD / sistema de archivos | capacidad grande y barata | DeepSeek (3FS), Dynamo |

La cuenta que lo justifica, con el modelo de costo del repositorio: calcular un token de prefill cuesta 0,05 ms y traerlo del pool 0,005 ms, diez veces menos. Mientras traer sea más barato que recalcular, conviene guardar.

## 4. El router

Con caché de prefijos, elegir réplica es un compromiso entre **ir a donde está el prefijo** y **no sobrecargar a nadie**. Las cuatro políticas del repositorio (paquete `cluster`):

- **En ronda.** Reparte parejo, pero cada turno de una conversación cae en otra réplica: solo aprovecha el prompt de sistema.
- **Menor carga.** Balancea bien y tampoco sabe dónde está cada conversación.
- **Por sesión (sticky).** Un hash de la sesión elige siempre la misma réplica: aprovecha el caché pero no mira la carga. Es consistent hashing en su forma más simple.
- **Consciente del prefijo.** La del Conductor de Mooncake (§6.1): para cada réplica estima el tiempo al primer token como la espera por lo que ya tiene encolado más el prefill de lo que le falta del prompt, y elige la menor. Si el prefijo está en otra máquina, suma traerlo; si ninguna cumple el SLO, devuelve 429 antes de gastar cómputo.

**La demo** (`go run ./cmd/sim router`): 1.000 conversaciones, 4.539 pedidos, cuatro réplicas.

| Router | Aciertos de caché | TTFT p50 | TTFT p99 | Desbalance |
| --- | --- | --- | --- | --- |
| En ronda, sin caché | 0% | 19 s | 63 s |  |
| En ronda | 56% | 119 ms | 790 ms |  |
| Menor carga | 57% |  | 512 ms |  |
| Por sesión | 73% |  | 4,1 s | 1,24 |
| Consciente del prefijo | 73% | 79 ms | 456 ms | 1,07 |
| Consciente del prefijo + pool | 88% |  | 142 ms |  |

Sin caché hay que calcular 18,3 millones de tokens de prefill y el clúster se satura. Sticky y consciente del prefijo aprovechan lo mismo; la diferencia está en la cola: sticky deja que una réplica con conversaciones largas se sature. El pool compartido permite traer un prefijo calculado en otra réplica a un décimo del costo.

**Con poco caché** (`-cache 256`, un cuarto por réplica) la diferencia crece: en ronda 40% (p99 7,2 s), sticky 44% (p99 26 s), consciente del prefijo 52% (p99 3,0 s), y con pool 88% (155 ms).

**Un caso real.** El postmortem que publicó Anthropic en septiembre de 2025 cuenta que su routing es sticky, y que un cambio de balanceo de carga mandó pedidos de contexto corto a servidores configurados para un millón de tokens; por ser sticky, los siguientes pedidos de esas conversaciones también caían ahí. El router no solo decide la latencia: decide en qué hardware y configuración cae cada pedido.

**En Kubernetes.** Un balanceador HTTP común elige por conexiones o en ronda, que es lo peor de la tabla. La **Gateway API Inference Extension** deja que el gateway elija el pod mirando el estado del modelo: la cola, el uso de KV cache, qué prefijos tiene. **llm-d**, en el sandbox de la CNCF desde marzo de 2026, está armado sobre vLLM y esa extensión y trae las piezas de esta clase: scheduler que rutea por prefijo, prefill y decode como servicios separados, offloading del KV cache en niveles y autoscaling. NVIDIA Dynamo hace algo parecido. El Endpoint pasa a ser un objeto con estado, y el scheduler necesita información fresca de cada réplica: el problema clásico de la información de carga en un balanceador distribuido.

## 5. Prefill y decode

**La interferencia.** Con batching continuo (Orca, Clase 18), prefill y decode comparten iteraciones. Con prompts de 8.000 tokens, el prefill de uno solo cuesta, con el modelo de costo del repositorio, 20 ms de leer los pesos más 0,05 ms por token: unos 420 ms. Las iteraciones de decode duran unos 30 ms; en la que entra ese prefill, cada usuario que estaba leyendo ve el texto detenerse casi medio segundo.

Dos métricas en tensión: el **tiempo al primer token** (TTFT), que ve el usuario nuevo, y el **tiempo entre tokens** (TBT o TPOT), que ve el que ya está leyendo.

**Partir el prefill (Sarathi-Serve, OSDI 2024).** El prompt se procesa de a pedazos de tamaño fijo, por ejemplo 512 tokens por iteración, junto con los decodes del lote (*stall-free scheduling*). Ninguna iteración explota; el prompt largo tarda un poco más en terminar su prefill. Contra vLLM reportan 2,6 veces más capacidad con Mistral-7B en una A100, hasta 3,7 con Yi-34B y hasta 5,6 con Falcon-180B en pipeline. Hoy viene prendido por defecto en los servidores principales.

**Separar las máquinas (DistServe, OSDI 2024).** Un pool de GPUs solo hace prefill: calcula el prompt, produce el primer token y manda el KV cache. Otro pool solo hace decode. Ventajas: no hay interferencia, cada pool usa el paralelismo que le conviene y escala con su propia demanda. La métrica es **goodput**: pedidos por segundo que cumplen a la vez el SLO de TTFT y el de TBT. Reportan hasta 7,4 veces más pedidos o un SLO 12,6 veces más estricto. Según el retrospectivo de sus propios autores, en 2025 la desagregación pasó a ser lo estándar en producción.

**Cuánto cuesta mover el KV.** Llama 3 70B ocupa 320 KiB por token; un prompt de 8.000 tokens son unos 2,6 GB. Con 800 Gb/s de RDMA (unos 100 GB/s), 26 ms, contra unos 400 ms de calcular ese prefill: del orden de un 6%. Y Mooncake manda el KV de cada capa apenas termina, superpuesto con las siguientes, el mismo truco que DualPipe en la Clase 17.

**La demo** (`go run ./cmd/sim desagregado`), misma carga, cuatro réplicas:

| Configuración | Cumple el SLO | Por qué |
| --- | --- | --- |
| Colocado | 25% | TBT p99 de 194 ms por los prefills largos |
| Colocado, prefill de a 512 | 100% | TBT p99 de 60 ms, goodput 7,90 |
| 1 prefill + 3 decode | 25% | el prefill no da abasto: TTFT p50 de 22 s |
| 2 prefill + 2 decode | 62% |  |
| 3 prefill + 1 decode | 8% | el decode se satura: TBT p99 de 244 ms |

**Lo honesto.** Con este modelo de costo y cuatro réplicas, partir el prefill gana. Lo que el modelo no captura es la ventaja de DistServe a escala: cada pool con su propio paralelismo y hardware, escalando por separado. Desagregar cambia un problema de interferencia por uno de **dimensionamiento**, y ese dimensionamiento tiene que ser dinámico: con agentes y documentos crecen los prompts; con modelos de razonamiento crece el decode. En Kubernetes es un HorizontalPodAutoscaler por pool, con la cola de prefill o el uso de KV cache como métrica en lugar de CPU.

## 6. Mooncake

Mooncake junta las dos ideas de la clase, caché de prefijos compartido y desagregación, alrededor de un pool de KV cache distribuido (figura 1 y §3).

**Las piezas.**

- **Conductor:** el scheduler global. Para cada pedido elige una instancia de prefill y una de decode mirando dónde está el KV cache y cuánta carga tiene cada una.
- **Pools de prefill y de decode:** grupos de máquinas de GPU separados.
- **Pool de KV cache:** no son máquinas aparte, sino la DRAM y los SSDs de las mismas máquinas de GPU, organizados en bloques.
- **Messenger:** un componente en cada nodo que mueve el KV cache con GPUDirect RDMA.

**El flujo de un pedido.**

1. La instancia de prefill elegida trae del pool el KV cache del prefijo que ya existe.
2. Calcula solo el resto, en pedazos y en pipeline si es largo.
3. Manda el KV cache resultante al nodo de decode capa por capa, mientras calcula.
4. El pedido entra al batching continuo del decode, que todavía puede rechazarlo si no cumpliría el SLO de TBT.

**Prompts muy largos: chunked pipeline parallelism (CPP).** Con cientos de miles de tokens, un prefill en una sola máquina rompe el SLO de TTFT. Mooncake agrupa X nodos de prefill y parte el prompt en pedazos; cada pedazo corre en otra etapa a la vez, en pipeline. Tensores entre nodos haría dos all-reduce por capa por RDMA, y paralelismo de secuencia comunicaría al menos una vez por capa; el pipeline solo comunica en los bordes de las etapas. Es el razonamiento de la Clase 17: lo que menos comunica es lo que cruza la red.

**Sobrecarga: a quién no atender.** Kimi tenía más demanda que GPUs. Si un pedido va a terminar fuera del SLO, el cómputo que se le dedicó se pierde; el peor caso es hacer el prefill y que el decode lo rechace. Con **rechazo temprano**, el Conductor rechaza al llegar si la carga no da. Pero el decode se carga recién cuando termina el prefill, así que el rechazo mira una carga vieja y prefill y decode oscilan en contrafase (figura 9). La solución es **predecir** la carga del decode: con la traza a doble velocidad en 8 + 8 máquinas, los rechazos bajan de 4.183 a 3.589. Es control con realimentación demorada, como el control de congestión de TCP.

**Resultados** (versión de FAST). En el despliegue real de Kimi, Mooncake atiende un 115% más de pedidos en los clústeres de A800 y un 107% más en los de H800 que el sistema anterior. En experimentos, la capacidad efectiva (pedidos que cumplen el SLO) sube entre 59% y 498% según el escenario. Corre en miles de nodos y procesa más de 100 mil millones de tokens por día. La versión de arXiv se fue actualizando y tiene otros números; usamos los del paper publicado.

## 7. La frontera

**Servir un modelo de expertos: DeepSeek.** En febrero de 2025 DeepSeek publicó su sistema de inferencia para V3 y R1. Prefill y decode van separados: en prefill cada unidad son 4 nodos (32 GPUs) con **EP32**; en decode, 18 nodos (144 GPUs) con **EP144**, 2 expertos ruteados y uno compartido por GPU. ¿Por qué tan grande? Cada token activa 8 de 256 expertos; para que cada experto multiplique de forma eficiente necesita muchos tokens, o sea un lote global enorme, y repartir los expertos en más GPUs hace que cada una lea menos pesos, que es lo que limita el decode. Como DualPipe en el entrenamiento, superponen comunicación y cómputo: dos microbatches alternados en prefill, la atención partida en un pipeline de 5 etapas en decode.

**Un día de DeepSeek** (27 al 28 de febrero de 2025):

| Dato | Valor |
| --- | --- |
| Nodos de 8 H800 | 226,75 en promedio, pico de 278 |
| Tokens de entrada | 608 mil millones, 56,3% del caché en disco |
| Tokens de salida | 168 mil millones |
| Por nodo de prefill | \~73.700 tokens/s (con aciertos de caché) |
| Por nodo de decode | \~14.800 tokens/s |
| Por usuario | 20 a 22 tokens/s |

La cuenta de servilleta: 14.800 / 21 ≈ 700 usuarios generando a la vez por nodo de decode. El informe calcula un margen teórico de 545% si todo se cobrara a la tarifa de R1, y aclara que el ingreso real es mucho menor.

**Decodificación especulativa** (Leviathan et al., ICML 2023). Un modelo chico propone k tokens; el grande verifica los k en una sola pasada, acepta los primeros que coinciden y en el primero que no pone el suyo. El resultado es exactamente el que habría generado el grande solo. Conviene porque el decode está limitado por leer los pesos: una pasada sobre k+1 tokens cuesta casi lo mismo que sobre uno. Si cada token se acepta con probabilidad α:

> tokens por pasada = (1 − α^(k+1)) / (1 − α), costo = k·c + 1

donde c es el costo del borrador relativo a una pasada del grande. Con α = 0,8, k = 4 y c = 0,05: 3,36 tokens por 1,2 pasadas, unas 2,8 veces más rápido (`go run ./cmd/sim especulativo`). Con α bajo, proponer 8 es peor que 4. Hoy el borrador suele ser una cabeza chica sobre el mismo modelo (EAGLE), y DeepSeek-V3 entrena un módulo de predicción de varios tokens (MTP) que sirve para esto.

**KV cache más chico.** Como el KV cache decide casi todo, la arquitectura cambió para que ocupe menos:

- **Grouped-query attention:** Llama 3 70B tiene 64 cabezas de query y solo 8 de keys y values: 320 KiB por token en lugar de 2,5 MiB.
- **Multi-head latent attention (MLA):** guarda un vector latente comprimido y reconstruye keys y values al calcular. DeepSeek-V2 reporta un 93,3% menos de KV cache que su modelo denso de 67B.
- **Cuantizar el KV cache** a 8 bits o menos.

Cada byte menos por token se multiplica tres veces: más secuencias en el lote, más prefijos en cada nivel del caché, menos para transferir de prefill a decode. Es el mismo argumento que FP8 en el entrenamiento.

**¿Y los laboratorios?** DeepSeek publicó su sistema completo y Moonshot publicó Mooncake. De Anthropic, lo público es el postmortem de septiembre de 2025: sirven Claude en Trainium, GPUs de NVIDIA y TPUs, con pools separados por largo de contexto y routing sticky. Google y OpenAI publicaron poco sobre cómo sirven. Todo lo de esta clase está en código abierto: vLLM y SGLang tienen caché de prefijos, prefill en pedazos y desagregación; llm-d y Dynamo los orquestan en un clúster.

## 8. Demo, preguntas y lecturas

**El repositorio.** `servir-tokens` es el mismo motor de la Clase 18, ahora con caché de prefijos, prefill en pedazos y varias réplicas detrás de un router, todo en tiempo virtual. Paquetes nuevos: `prefix` (hashes encadenados y LRU) y `cluster` (routers y desagregación).

| Comando | Qué muestra |
| --- | --- |
| `go run ./cmd/sim router` | el router y el caché de prefijos |
| `go run ./cmd/sim desagregado` | colocado, en pedazos o separado |
| `go run ./cmd/sim especulativo` | decodificación especulativa |
| `go test ./...` | los tests de lo que dice la clase |

Los tests verifican que el router consciente del prefijo reusa más y tiene mejor p99 que el de ronda, que sticky tiene peor cola, que el pool compartido sube el reuso, que el prefill en pedazos deja el TBT bajo el SLO y que el desagregado entrega todos los tokens con menor TBT que el colocado. El simulador *Clúster de inferencia* permite cambiar el router, el caché y la cantidad de réplicas de prefill en el navegador.

Ejercicios del README: compartir bloques físicos entre secuencias con el mismo prefijo, rechazo temprano, replicación de hot spots como Mooncake y un ratio de prefill y decode adaptativo.

**Preguntas de repaso.**

1. ¿Por qué un bloque solo se puede reusar si es igual todo lo anterior? ¿Cómo lo garantiza el hash encadenado? *Las keys y values de un token dependen de todos los anteriores. El hash del bloque i incluye el del i−1, así que si coincide en i, coincide todo el prefijo.*
2. ¿Qué tiene el router consciente del prefijo que no tienen el de ronda ni el sticky? *Estima el TTFT en cada réplica combinando caché y cola: va a donde está el prefijo salvo que esa réplica esté cargada. Ronda ignora el caché; sticky ignora la carga.*
3. Un prompt de 8.000 tokens en un lote de decodes: ¿qué ve el usuario que ya estaba leyendo? *Que el texto se detiene unos 400 ms. Sarathi parte el prefill en pedazos fijos por iteración.*
4. ¿Por qué con prefill y decode separados el ratio tiene que seguir a la carga? *Cada pool tiene capacidad fija: si hay más prefill del que puede hacer, el TTFT explota aunque el decode esté libre, y al revés.*
5. ¿Por qué verificar k tokens propuestos cuesta casi lo mismo que generar uno? *El decode está limitado por leer los pesos: una pasada sobre k+1 tokens los lee una sola vez.*

**Lecturas.**

- Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://www.usenix.org/conference/fast25/presentation/qin), FAST 2025. Es el paper de la clase.
- Zhong et al., [DistServe](https://arxiv.org/abs/2401.09670), OSDI 2024.
- Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024.
- DeepSeek, [Day 6: One More Thing, DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index), 2025.
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
- Zheng et al., [SGLang](https://arxiv.org/abs/2312.07104) (RadixAttention), 2024.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA), 2024.
- Anthropic, [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues), septiembre de 2025.
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) y [llm-d](https://llm-d.ai/).

**Cierre de las cuatro clases.** En la 16 y la 17 el estado era el modelo y se partía en dimensiones; en la 18 y la 19 el estado es el KV cache, y se pagina, se comparte, se cachea y se transfiere. Las preguntas son las mismas: qué se parte, qué se comunica, qué pasa cuando falla.
