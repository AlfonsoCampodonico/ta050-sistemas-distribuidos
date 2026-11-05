# Clase 18 — Cómo se sirve un modelo de lenguaje

Dec 4, 2026 · @Alfonso Campodonico

En las Clases 16 y 17 entrenamos un modelo en miles de GPUs. Esta clase es la otra mitad: cómo se sirve. Un pedido de chat cruza infraestructura conocida —HTTPS, balanceadores, un gateway— y termina en un loop que corre el modelo una vez por token. Casi todo el tiempo y casi todo el costo están en ese loop, y lo que decide cuánta gente puede atender una GPU a la vez es cómo se administra una sola estructura: el KV cache.

**Paper:** W. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023. Es el paper de vLLM.

**Lectura complementaria:** G.-I. Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022. Para el recorrido completo de un pedido: A. Campodónico, [What happens after you hit send](https://alfonsocampodonico.com/posts/what-happens-after-you-hit-send).

**Material práctico:** el simulador **Motor de inferencia**, para mostrar en clase, y el repositorio `servir-tokens` (Go, sin dependencias), con el mismo motor detrás de un servidor HTTP de verdad.

La idea central del paper es vieja para nosotros: es la memoria virtual con paginación de Sistemas Operativos, aplicada a la memoria de una GPU. Vamos a ver por qué hacía falta y qué cambia.

## 1. El viaje de un pedido

La primera mitad del viaje es infraestructura que ya conocemos, y es rápida: decenas de milisegundos. El chat arma un JSON con la conversación y lo manda por HTTPS con un `POST`, pidiendo la respuesta como stream (`Accept: text/event-stream`). La conexión **queda abierta**, porque la respuesta no va a llegar de una vez.

1. **Edge.** DNS o anycast lleva el pedido a un punto de presencia cercano. Ahí termina TLS, se filtra tráfico inválido y se aplican límites por IP, y un balanceador elige un backend sano y fija ahí la conexión larga.
2. **Gateway.** Autentica la clave, revisa cuotas y límites. Los límites se cuentan en **tokens**, no en pedidos: el token es la unidad de costo.
3. **Router.** Elige un pool de réplicas del modelo y una región. Algunos prefieren una réplica que vio hace poco el mismo prefijo, para reaprovechar su caché.

Una réplica de un modelo de 70 mil millones de parámetros ocupa unos 140 GB de pesos en 16 bits, más que una GPU H100 de 80 GB. Se reparte entre las 8 GPUs de un nodo con paralelismo de tensores, que ya vimos en la Clase 16: dentro del nodo, por NVLink. Levantar una réplica lleva minutos, porque hay que traer esos pesos y calentar el servidor. Por eso la capacidad se planifica de antemano y está esperando.

Desde acá, todo pasa adentro del **servidor de inferencia** (vLLM, SGLang, TensorRT-LLM). Es la parte que estudia esta clase.

> El recorrido completo, con un simulador de cada salto, está en el post de la lectura complementaria. Para nosotros alcanza con una conclusión: la red es rápida y aburrida, y el tiempo se va en la GPU.

## 2. Prefill y decode

Generar texto es un loop: correr el modelo una vez, obtener **un** token, agregarlo a la secuencia y volver a correr. Cada vuelta depende de la anterior, porque el token siguiente se calcula a partir de todos los anteriores. No hay forma de paralelizar una respuesta a lo largo del tiempo.

El texto se parte primero en tokens, cada uno con un número de un vocabulario fijo, y cada número se convierte en un vector. Ese es el input del modelo. Después hay dos fases muy distintas.

**Prefill.** La primera pasada procesa todo el prompt de una vez: todos sus tokens atraviesan todas las capas en paralelo. Son multiplicaciones de matrices grandes, la GPU está ocupada haciendo cuentas: es una fase limitada por el **cómputo**. Termina con el primer token de la respuesta.

**Decode.** Cada pasada siguiente procesa un solo token nuevo por secuencia. Para hacerlo, la GPU tiene que leer **todos los pesos** desde su memoria, más el estado guardado de los tokens anteriores, para hacer muy poca cuenta con ellos. Son productos de matriz por vector: la fase está limitada por el **ancho de banda de memoria**, y la GPU pasa la mayor parte del tiempo esperando datos.

|  | Prefill | Decode |
| --- | --- | --- |
| Cuántas veces | Una por pedido | Una por token de salida |
| Tokens por pasada | Todo el prompt | Uno por secuencia |
| Cuello de botella | Cómputo | Ancho de banda de memoria |
| Se nota como | Tiempo al primer token (TTFT) | Tiempo entre tokens (TPOT) |
| Qué gana con batching | Poco: ya es paralelo | Casi todo: una lectura de pesos sirve a todo el lote |

La última fila es la clave de toda la clase. Si una pasada de decode cuesta lo mismo con una secuencia que con treinta, porque lo caro es leer los pesos una vez, entonces conviene meter en cada pasada la mayor cantidad posible de secuencias. El throughput de un servidor de inferencia es, casi literalmente, **cuántas secuencias entran en el lote**.

> El paper de vLLM lo dice en §2.2: la generación autorregresiva procesa un token por paso con operaciones de matriz por vector, subutiliza el cómputo de la GPU y queda limitada por la memoria. Esa fase explica la mayor parte de la latencia de un pedido. Y en §2.3: el batching amortiza la carga de los pesos entre todos los pedidos del lote.

## 3. El KV cache

En cada capa, la atención compara el token actual con todos los anteriores. Para eso necesita, de cada token anterior, dos vectores: su **key** y su **value**. Recalcularlos en cada paso sería rehacer todo el prompt por cada token generado. En lugar de eso, se calculan una vez —en el prefill para el prompt, y en el paso en que aparece cada token generado— y se guardan. Eso es el **KV cache**: el estado de cada secuencia en curso.

Crece un poco con cada token y ocupa mucho. El tamaño por token es:

```latex
2 \times \text{capas} \times \text{cabezas KV} \times \text{dimensión por cabeza} \times \text{bytes por número}
```

| Modelo | Cuenta | Por token | Una secuencia |
| --- | --- | --- | --- |
| OPT-13B (el del paper) | 2 × 40 capas × 5120 × 2 bytes | ≈ 800 KB | ≈ 1,6 GB a 2048 tokens |
| Llama 3 70B | 2 × 80 capas × 8 cabezas × 128 × 2 bytes | 320 KiB | ≈ 2,6 GB a 8000 tokens |

Llama 3 ocupa menos por token que un modelo mucho más chico porque usa *grouped-query attention*: varias cabezas de query comparten las mismas keys y values.

Con eso se ve el reparto de la memoria de una GPU. En el ejemplo del paper (§1, figura 1), un modelo de 13B en una A100 de 40 GB usa alrededor del 65% para los pesos y casi el 30% para el KV cache. Los pesos son fijos: lo que se puede administrar es ese 30%. **El KV cache es lo que limita el tamaño del lote**, y por lo tanto el throughput.

> La clase entera se puede resumir en una frase del paper (§3): el throughput de un servidor de inferencia está limitado por la memoria. Lo que sigue es cómo usar bien esa memoria.

## 4. Batching por iteración: Orca

Si el lote es lo que importa, la primera pregunta es cómo se arma. Los sistemas anteriores a Orca, como FasterTransformer de NVIDIA, hacían **batching a nivel de pedido**: juntar varios pedidos, correr el modelo sobre ese lote hasta que todos terminen y recién entonces armar el siguiente.

Con generación de texto eso funciona mal, porque las respuestas tienen largos muy distintos. Una respuesta de 10 tokens que cae en un lote con una de 1.000 queda terminada después de 10 iteraciones, pero no puede volver al cliente hasta que termina la larga. Mientras tanto ocupa un lugar en el lote, y los pedidos nuevos que llegaron esperan afuera, aunque haya lugar de sobra para ellos.

La propuesta de Orca (OSDI 2022) es el **scheduling a nivel de iteración**: el scheduler le pide al motor que corra **una sola iteración** del modelo sobre el lote actual, y entre iteración e iteración decide de nuevo. Las secuencias que terminaron salen y vuelven al cliente; las que esperan entran. Hoy se lo llama **continuous batching**.

Hay un detalle técnico para que esto funcione: en una misma iteración conviven secuencias que están en prefill, con muchos tokens, y otras en decode, con uno. Orca lo resuelve con **selective batching**: batchea las operaciones que no dependen de la secuencia, como las capas lineales, y procesa la atención de cada secuencia por separado.

> El resultado que reporta Orca, sobre GPT-3 de 175B: 36,9 veces más throughput que FasterTransformer con la misma latencia. Es la misma idea que la clase pasada vimos en el pipeline: no dejar que el pedido más lento frene a todos.

Con el simulador se ve directamente: en batching estático, el primer token llega junto con la respuesta completa y el throughput queda clavado en lo que da un lote que espera a su respuesta más larga. En el repositorio, `go run ./cmd/sim batching` lo cuantifica.

## 5. El problema de la memoria

Orca resuelve cuándo entra cada secuencia al lote, pero no cuántas entran. Eso lo decide la memoria del KV cache, y ahí está el problema que ataca vLLM.

Los sistemas previos guardaban el KV cache de cada pedido en un **tramo contiguo** de memoria, porque los kernels de atención esperaban los datos uno al lado del otro. Pero cuando llega un pedido nadie sabe cuánto va a generar. La única opción segura es reservar de entrada el máximo posible, por ejemplo 2048 tokens. Eso produce tres desperdicios, que el paper describe en §3.1:

1. **Reserva.** Lugar apartado para tokens que todavía no se generaron. Se va a usar, pero mientras tanto nadie más puede usarlo.
2. **Fragmentación interna.** Lugar reservado que nunca se va a usar, porque la respuesta terminó antes del máximo. Se descubre recién al final.
3. **Fragmentación externa.** Huecos libres entre tramos, que suman mucho pero que, por separado, son demasiado chicos para un pedido nuevo.

El paper midió cuánto de la memoria del KV cache guardaba tokens de verdad en los sistemas existentes: **entre el 20,4% y el 38,2%**. Entre el 60% y el 80% de la memoria que más importa se desperdiciaba.

El simulador lo muestra con colores: con memoria contigua, cada pedido pinta su tramo de 512 slots con un color fuerte para los tokens que ya tiene y uno pálido para el resto. Casi todo el mapa queda pálido. En el repositorio, `go run ./cmd/sim memoria` da una utilización de alrededor del 28% para la reserva del máximo: dentro del rango que midió el paper.

> Fijate el parecido con Sistemas Operativos: reservar contiguo con first-fit, huecos que no sirven, memoria que sobra en total pero no en un solo lugar. Es el problema clásico de la asignación de memoria antes de la paginación.

## 6. PagedAttention: el KV cache como memoria virtual

La solución de vLLM es la misma que encontraron los sistemas operativos para la memoria de los procesos: **paginar**. La memoria del KV cache se parte en **bloques** de tamaño fijo, de 16 tokens por defecto (§7.2). Cada secuencia ve sus tokens como si estuvieran contiguos, en bloques lógicos 0, 1, 2…, y una **tabla de bloques** traduce cada bloque lógico a un bloque físico que puede estar en cualquier lugar de la GPU. Cada entrada de la tabla anota además cuántas posiciones del bloque están llenas (§4.2).

El paper hace explícita la analogía (§1):

| Sistema operativo | vLLM |
| --- | --- |
| Página | Bloque de KV cache |
| Byte | Token |
| Proceso | Secuencia |
| Tabla de páginas | Tabla de bloques |

&#91;embedded content: tablas de bloques · dos secuencias, bloques de 4 tokens\]

La secuencia A tiene sus tokens en los bloques físicos 7 y 1, y la B en el 3 y el 6: ninguna está contigua, y no hace falta. Los únicos lugares sin usar son el final del último bloque de cada una.

De ahí salen las tres propiedades que importan:

- **Se asigna a medida que hace falta.** Al admitir una secuencia se le dan los bloques justos para el prompt. Cuando el último bloque se llena, se le asigna otro. Desaparece la reserva.
- **El desperdicio es menor a un bloque por secuencia.** Lo único que queda sin usar es el final del último bloque: a lo sumo 15 tokens de cada secuencia, sea cual sea su largo.
- **No hay fragmentación externa.** Todos los bloques son iguales, así que cualquier bloque libre sirve para cualquier secuencia.

El precio está en el kernel de atención. Ahora las keys y values de una secuencia están desparramadas por la memoria, así que el kernel tiene que leerlas bloque por bloque siguiendo la tabla. Eso es lo que significa el nombre: **PagedAttention** es un algoritmo de atención que funciona sobre memoria paginada.

> ¿Por qué no bloques de un token, para no desperdiciar nada? Porque bloques muy chicos no aprovechan el paralelismo de la GPU al leer, y la tabla crece. El paper estudia el tamaño de bloque en §7.2 y elige 16 por defecto: lo bastante grande para leer bien la memoria, lo bastante chico para desperdiciar poco. Es el mismo compromiso del tamaño de página en un sistema operativo.

## 7. Compartir y desalojar

Una vez que hay una tabla en el medio, aparecen las otras herramientas de la memoria virtual.

**Compartir con copy-on-write (§4.4).** Cuando se piden varias respuestas para el mismo prompt (muestreo paralelo) o se hace beam search, todas las secuencias empiezan con el mismo prompt. vLLM no lo copia: las tablas de todas apuntan a los mismos bloques físicos, y cada bloque lleva un **contador de referencias**. Cuando una secuencia quiere escribir en un bloque compartido, recién ahí se le asigna uno nuevo, se copian los datos y se baja el contador del original. Es el `fork()` de Unix. El paper mide ahorros de memoria de 6,1% a 9,8% con muestreo paralelo en Alpaca y de 37,6% a 55,2% con beam search; con ShareGPT, de 16,2% a 30,5% y de 44,3% a 66,3% (§6.3).

**Scheduling FCFS (§4.5).** Los pedidos se atienden por orden de llegada. Como ya no se reserva el máximo, puede pasar que las secuencias en curso crezcan y la memoria se acabe. Entonces vLLM **desaloja** las secuencias que llegaron último. El desalojo es todo o nada: una secuencia se va con todos sus bloques, porque los necesita todos para cada paso. Los grupos de secuencias que comparten bloques se desalojan juntos.

Para traer de vuelta una secuencia desalojada hay dos caminos:

- **Swapping.** Copiar sus bloques a la memoria de la CPU y traerlos después. Es el swap de un sistema operativo.
- **Recomputación.** Tirar sus bloques y, al volver a admitirla, recalcularlos con un prefill de prompt más lo ya generado. Funciona porque el prefill es rápido: procesa todos esos tokens en paralelo.

> El paper compara las dos en §7.3: la recomputación nunca agrega más de un 20% sobre la latencia del swapping, y con bloques chicos el swapping rinde mal porque hace muchas transferencias pequeñas. En el rango de 16 a 64 tokens por bloque las dos andan parecido. El repositorio implementa recomputación; agregar swapping es el ejercicio 2.

En el simulador se ve: con memoria paginada y bastante carga, el contador de desalojos sube y algunas secuencias vuelven a empezar con un prefill.

## 8. Resultados y ejecución distribuida

**Ejecución distribuida (§4.6).** Un modelo grande no entra en una GPU, así que vLLM lo parte con paralelismo de tensores al estilo Megatron-LM, como vimos en la Clase 16: cada GPU tiene una parte de las cabezas de atención, y se sincronizan con all-reduce. El administrador del KV cache es **uno solo**, dentro del scheduler centralizado. Todas las GPUs comparten la misma traducción de bloques lógicos a físicos; cada una guarda en esos bloques sólo las keys y values de sus propias cabezas. En cada iteración, el scheduler les manda a todas la misma tabla.

**Evaluación (§6).** Los modelos son OPT de 13B, 66B y 175B y LLaMA de 13B, en A100 de Google Cloud: una GPU de 40 GB para 13B, cuatro para 66B y ocho de 80 GB para 175B. Las cargas salen de dos conjuntos de datos: ShareGPT, conversaciones reales que usuarios compartieron de ChatGPT, y Alpaca, instrucciones generadas por un modelo. ShareGPT tiene prompts unas 8,4 veces más largos y respuestas unas 5,8 veces más largas que Alpaca.

La comparación es contra FasterTransformer y contra tres versiones de Orca que se distinguen por cuánta memoria reservan por pedido:

| Sistema | Qué reserva | Pedidos por segundo que sostiene vLLM, en ShareGPT |
| --- | --- | --- |
| Orca (Max) | Siempre 2048 tokens | 2,7× a 8× más |
| Orca (Pow2) | Hasta el doble de lo necesario | — |
| Orca (Oracle) | Exactamente lo necesario, sabiéndolo de antemano | 1,7× a 2,7× más |
| FasterTransformer | Batching por pedido | Hasta 22× más |

Orca (Oracle) es imposible en la práctica: nadie sabe de antemano cuánto va a generar un pedido. Que vLLM le gane igual muestra que el problema no era sólo la reserva, sino también la fragmentación. El resumen del paper: **2 a 4 veces más throughput** que los sistemas del estado del arte con la misma latencia.

## 9. El stream de vuelta

Cada token sale del motor al final de una iteración, y se manda al cliente en ese momento por la conexión que quedó abierta desde el principio. El formato más común es **server-sent events (SSE)**: una respuesta HTTP con `Content-Type: text/event-stream` que nunca termina de llegar, y en la que cada evento es una línea `data:` seguida de una línea vacía:

```text
data: {"delta":"La "}

data: {"delta":"luz "}

data: [DONE]
```

El formato exacto cambia según el proveedor. En el camino de vuelta, el gateway cuenta los tokens de salida para el cobro y las cuotas, y el navegador agrega cada fragmento a medida que llega.

Dos problemas de sistemas distribuidos aparecen acá:

- **Proxies que juntan la respuesta.** Un proxy con buffering lee la respuesta entera antes de reenviarla. El servidor manda cada token a tiempo, pero el cliente recibe todo junto al final. En nginx se desactiva con `proxy_buffering off` o con el header `X-Accel-Buffering: no` que manda el servidor. En la demo, con `cmd/proxy -buffer`, los 60 tokens que el servidor mandó cada 40 ms llegan todos juntos a los 2,5 s.
- **Clientes que se van.** Si el usuario cierra la pestaña, el servidor tiene que darse cuenta y cancelar el pedido. Si no, sigue generando tokens que nadie va a leer, ocupando un lugar en el lote y bloques de KV cache que otro pedido podría usar. En Go, eso es mirar `r.Context().Done()` en el handler y avisarle al motor.

Los números de un pedido típico, con valores ilustrativos: la red, decenas de milisegundos; la cola y el prefill, decenas a cientos; cada token de decode, del orden de 10 a 50 ms. Una respuesta de 300 tokens a 30 ms por token tarda 9 segundos, y casi todo es decode. Por eso el streaming importa: el usuario empieza a leer después del primer token, no del último.

## 10. Demo, preguntas y referencias

El simulador **Motor de inferencia** corre en el navegador: muestra el mapa de memoria de la GPU llenándose token por token, con batching estático o continuo y memoria contigua o paginada, y compara las cuatro combinaciones con la misma carga. El repositorio `servir-tokens` tiene el mismo motor en Go:

```text
go run ./cmd/sim batching      # estático vs continuo (Orca)
go run ./cmd/sim memoria       # reserva contigua vs bloques (vLLM)
go run ./cmd/sim cow           # muestreo paralelo con copy-on-write
go run ./cmd/servidor          # el motor detrás de HTTP + SSE
go run ./cmd/proxy -buffer     # un proxy que rompe el streaming
```

**Preguntas de repaso**

1. ¿Por qué el decode está limitado por el ancho de banda de memoria y no por el cómputo? ¿Qué cambia en el costo de una iteración al pasar de 1 a 32 secuencias en el lote?
2. Calculá el tamaño del KV cache por token de un modelo de 32 capas con 8 cabezas KV de 128 dimensiones, en 16 bits. ¿Cuántas secuencias de 4.000 tokens entran en 40 GB?
3. En batching estático, ¿por qué el primer token llega con la respuesta completa? ¿Qué cambia Orca para evitarlo?
4. Nombrá los tres desperdicios de la reserva contigua y explicá cuáles elimina la paginación y cuál reduce a menos de un bloque.
5. ¿Qué analogía hace el paper entre vLLM y un sistema operativo? ¿Qué cumple el rol de la tabla de páginas?
6. Cuatro muestras de un prompt de 500 tokens: ¿cuántos bloques de 16 se copian con copy-on-write cuando cada una escribe su primer token? ¿Por qué no todos?
7. ¿Por qué el desalojo es todo o nada? ¿Cuándo conviene recomputar en lugar de hacer swapping?
8. Un usuario cierra la pestaña a la mitad de una respuesta larga. ¿Qué recursos quedan ocupados si el servidor no se entera, y cómo se entera?

**Referencias**

- W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
- G.-I. Yu et al. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
- A. Campodónico. [What happens after you hit send](https://alfonsocampodonico.com/posts/what-happens-after-you-hit-send). 2026.
- D. Narayanan et al. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). SC 2021 (Clase 16).
