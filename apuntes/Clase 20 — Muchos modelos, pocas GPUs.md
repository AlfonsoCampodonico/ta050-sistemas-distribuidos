# Clase 20 — Muchos modelos, pocas GPUs

Nov 5, 2026 · @Alfonso Campodonico

## 1. De un modelo muy pedido a muchos modelos poco pedidos

En las clases 18 y 19 servimos un modelo: primero una réplica por dentro, después muchas réplicas detrás de un router. Una plataforma real no sirve un modelo: sirve cientos de modelos abiertos y miles de versiones ajustadas por sus clientes, y casi todos reciben un pedido cada tanto. Darle una GPU a cada uno es tirar plata; compartir las GPUs obliga a cargar el modelo cuando llega un pedido, y un modelo son decenas de gigabytes.

El paper de hoy es **ServerlessLLM** (Fu et al., Universidad de Edimburgo, OSDI 2024): cómo hacer que ese arranque en frío tarde segundos y no minutos. El compañero es **S-LoRA** (Sheng et al., Berkeley, MLSys 2024): cómo servir miles de fine-tunes de un mismo modelo base en una GPU.

| | Clase 19: un modelo | Hoy: una flota |
| --- | --- | --- |
| Cuántos modelos | uno, con muchas réplicas | cientos, casi todos ociosos |
| Lo que se mueve | el KV cache | los pesos: decenas de GB |
| El scheduler decide | dónde está el prefijo | dónde está el checkpoint |
| Fine-tunes | — | miles de adaptadores sobre un modelo base |

La estructura es la de la Clase 19 —un estado grande y un scheduler que sabe dónde está—, pero el estado ahora son los pesos.

## 2. El problema

**La carga.** No hay trazas públicas de serverless para modelos de lenguaje, así que ServerlessLLM usa la de Azure Functions (Shahrad et al., *Serverless in the Wild*, ATC 2020). Con un keep-alive de 5 minutos, **más del 40%** de las funciones arranca en frío en más de un cuarto de sus pedidos, y **cerca del 25%** en más del 60%. Para una función, un arranque en frío es levantar un contenedor; para un modelo de lenguaje es traer el checkpoint, y los proveedores reportan decenas de segundos.

**Serverless para modelos** (§2.2, figura 1). El cliente sube su checkpoint y paga por pedido. Un **router de pedidos** manda cada pedido a una GPU que ya tiene el modelo; si no hay, le pide al **scheduler de carga** que arranque uno, y ese servidor baja el checkpoint, lo pasa por SSD y DRAM y lo sube a la GPU. Lo ofrecen SageMaker, Azure, Google, Hugging Face, Together, Replicate, Fireworks y otros; KServe es la versión abierta sobre Kubernetes.

**Multiplexación estadística.** Con una GPU por modelo, cada una se dimensiona para el pico de su modelo y el resto del tiempo está ociosa. Con una flota compartida alcanza con el pico de la suma: las ráfagas de distintos modelos no coinciden. **AlpaServe** (Li et al., OSDI 2023) lleva la idea más lejos —partir un modelo entre varias GPUs aunque entre en una, para que una ráfaga use muchas a la vez— y reporta hasta **10×** más pedidos por segundo o **6×** más ráfagas con más del 99% de los pedidos dentro del SLO.

## 3. Anatomía de un arranque en frío

**El checkpoint es enorme** (§2.3, en FP16): LLaMA-2-70B 130 GB, DBRX 250 GB, Mixtral-8x22B unos 280 GB, Grok-1 más de 600 GB. Bajar 130 GB de S3 con una red de 5 GB/s lleva **al menos 26 s**. Generar un token, en cambio, tarda menos de 100 ms.

**El servidor ya tiene dónde guardarlo** (§3.1). Un servidor de 8 GPUs puede tener hasta 4 TB de DRAM, 64 TB en NVMe y 192 TB en SATA. Con PCIe 5.0 cada GPU tiene su enlace con la DRAM: 512 GB/s sumados; de NVMe en RAID 0 a DRAM, unos 60 GB/s. KServe y Ray Serve usaban una fracción de la memoria y casi nada de los SSDs. Guardar los checkpoints ahí es barato (el hardware ya está), escala solo y va a mejorar (un Grace Hopper tiene 1 TB de memoria de CPU con 900 GB/s hasta la HBM).

Las alternativas no alcanzan (§2.4): mantener GPUs calientes es caro con modelos grandes; cachear sólo en DRAM no da para cientos de GB; y servidores de almacenamiento aparte cuestan: aun con 100 Gb/s la descarga pasa los 20 s, y un servidor de caché de AWS con 210 GB y 200 Gb/s cuesta 16,3 dólares la hora, lo mismo que un servidor de 8 GPUs.

| Nivel | Capacidad en un servidor de 8 GPUs | Ancho de banda |
| --- | --- | --- |
| GPU (HBM) | lo que usa el modelo | 512 GB/s desde DRAM, sumando las 8 |
| DRAM | hasta 4 TB | ≈ 60 GB/s desde NVMe en RAID 0 |
| SSD | 64 TB NVMe, 192 TB SATA | — |
| Remoto | sin límite práctico | ≥ 26 s para 130 GB a 5 GB/s |

**Aun desde el SSD local, cargar tarda** (§7.2, figura 6a, NVMe en RAID 0 de 12 GB/s, cachés del sistema vaciadas):

| Modelo | PyTorch | Safetensors | ServerlessLLM |
| --- | --- | --- | --- |
| OPT-2.7B | 3,0 s | 1,8 s | 0,5 s |
| OPT-30B | 34 s | 18,5 s | 4,5 s |
| Falcon-40B | 50 s | 25 s | 6,2 s |
| LLaMA-2-70B | 84 s | 48 s | 10,3 s |

Entre 3,6× y 8,2× más rápido. PyTorch lee tensor por tensor y copia dos veces (host y después GPU); Safetensors usa mmap y en frío cada página es un fallo de página (112 mil para LLaMA-2-7B). El tiempo de ServerlessLLM depende sólo del tamaño. Con un adaptador LoRA de 1 GB de LLaMA-70B: 83,5 ms contra 370 ms de Safetensors.

**Un formato pensado para leer** (§4.1). En el entrenamiento los checkpoints se escriben mucho y se leen poco; en serverless se suben una vez y se cargan miles. ServerlessLLM los convierte al subirlos:

- **Particiones por GPU:** un archivo por GPU con sólo los bytes de sus tensores, seguidos. Se lee de corrido, en bloques grandes.
- **Índice de tensores:** nombre → (GPU, offset, tamaño), alineado. Restaurar un tensor es base + offset.
- **Cargar aparte:** un *model manager* reserva la memoria y carga los bytes mientras el proceso de inferencia arma el modelo; al final le pasa las direcciones base (CUDA IPC) y cada tensor se apunta a su lugar.

**El cargador multinivel** (§4.2, figura 7). Partiendo de leer tensor por tensor, cada optimización multiplica el throughput: leer en bloques de 16 MB, **1,2×** (un tercio de los tensores mide menos de 1 MB); O_DIRECT, sin la caché de páginas del kernel, **2,1×**; varios hilos, porque un SSD tiene muchos canales, **2,3×**; memoria fijada, para que la copia a la GPU sea DMA, **1,4×**; y pipeline entre niveles, **1,5×**. Multiplicadas, unas 12 veces. Cuatro núcleos de CPU alcanzan. La DRAM es un pool de pedazos de tamaño fijo y cada GPU usa su PCIe en paralelo.

Lo importante del pipeline es que hace la carga **predecible**: mientras un pedazo sube a la GPU, el siguiente ya está en DRAM, así que el tiempo es tamaño sobre el ancho de banda del nivel más lento, no la suma.

**Conexión con la materia.** Es la misma jerarquía de almacenamiento que el KV cache de Mooncake (Clase 19), con otro contenido. Y el formato es una decisión de layout por patrón de acceso, como un log o un formato columnar.

## 4. Dónde arrancar cada modelo

**El scheduler estima** (§6). Para cargar un modelo en un servidor:

> tiempo de arranque = q + n / b

con q la cola de carga del servidor (las cargas van de a una a propósito, para que no se peleen el SSD y la estimación siga valiendo), n el tamaño y b el ancho de banda del nivel donde está el checkpoint en ese servidor, corregido con lo que va midiendo. Para mudar una inferencia en curso:

> tiempo de reanudar = α · (t_in + t_out) + β

con α y β ajustados con lo medido; lineal en los tokens de entrada y los generados; t_out se estima como duración sobre tiempo por token, sin preguntarle al servidor. El error medido está acotado en 5 ms en GPU y 40 ms desde SSD; la limpieza del estado de la GPU después de migrar lo empeora (en promedio 26 ms; una vez, 623 ms en 119 migraciones). Es el router consciente del prefijo de Mooncake con otra fórmula: allá cola más prefill de lo que falta, acá cola más carga de lo que no está.

**El servidor bueno está ocupado** (§5.1, figura 3). S1 tiene la GPU libre, A en DRAM y B en SSD; S2 corre una inferencia de A y tiene B en DRAM. Llega un pedido para B:

| Política | Modelo B (llega) | Modelo A (corriendo) |
| --- | --- | --- |
| Disponibilidad | carga desde el SSD de S1: lento | no se entera |
| Localidad | espera a que A termine en S2 | no se entera; S1 queda ociosa |
| Desalojo (Shepherd) | arranca ya en S2, desde DRAM | se corta: recarga y recalcula en S1 |
| Migración en vivo | arranca en S2 después de mudar A | sigue en S1 con una pausa mínima |

Esperar no sirve porque la duración de una inferencia de un LLM no se conoce: depende de cuántos tokens genere.

**Migración en vivo** (§5.2). Un snapshot del proceso tarda decenas de segundos o minutos, y la migración por páginas sucias no está soportada con GPUs. La observación: el estado de una inferencia son sus **tokens** (decenas a cientos de KB); el KV cache (de 1 a decenas de GB) es derivado y se puede recalcular, y recalcular es un prefill: el de 1.000 tokens tarda lo que generar unos 100. Entonces se hace en rondas: el origen manda los tokens y sigue generando; el destino recalcula; la ronda siguiente tiene muchos menos tokens. Cuando la diferencia es chica, el origen para y manda lo último.

**Conexión con la Clase 19.** Mooncake trae el KV cache en lugar de recalcularlo, porque tiene RDMA de 800 Gb/s y prompts de miles de tokens. ServerlessLLM tiene Ethernet de 10 Gb/s y el destino está ocioso. Misma pregunta —traer o recalcular—, otra respuesta: depende del ancho de banda contra el cómputo. Y es la pre-copia de la migración en vivo de máquinas virtuales, con un truco: se copia el estado que genera (los tokens), no el derivado.

**El protocolo** (§5.3, figura 4):

1. Cargar A en el destino, si no hay una instancia ociosa.
2. El scheduler le pide al origen que migre.
3. El origen manda los tokens al destino y sigue generando.
4. El destino recalcula el KV cache; se repite con lo nuevo.
5. El origen para y le devuelve todo al **router** con la marca «migrado».
6. Se descarga A del origen y se carga B.
7. El router cambia el destino en su tabla y sigue el stream. El cliente no se entera.

Fallas (§5.4): mientras tanto el origen sigue generando, así que si falla el destino, el origen continúa. Si falla el origen antes de migrar, se aborta y se descarga el destino; durante la migración, el destino descarta el KV cache. Si la inferencia termina sola en el medio, se avisa y el destino deja de recalcular. Hasta el paso 5 nada es irreversible.

**El estado del scheduler** (§6, §6.3). Qué GPUs están ocupadas, qué checkpoints hay en DRAM y SSD de cada servidor y sus colas de carga viven en un almacén de claves confiable (etcd, ZooKeeper). El orden importa: al asignar una carga marca la GPU; cuando el servidor confirma, actualiza el almacenamiento y recién ahí avisa al router. Si el scheduler se cae, el nuevo lee el estado y lo sincroniza con los servidores. Atiende miles de cargas por segundo en un servidor; repartirlo queda como trabajo futuro. El mapa es una caché del estado real y puede estar viejo (un checkpoint ya desalojado): la estimación falla pero la carga es correcta.

**Resultados** (§7.3 y §7.4, 4 servidores con 4 A40, 512 GB de DRAM, un NVMe de 2 TB y Ethernet de 10 Gb/s; traza de Azure con ráfagas):

| | ServerlessLLM | Ray Serve | Ray Serve con caché en SSD |
| --- | --- | --- | --- |
| OPT-6.7B, GSM8K, arranque medio | 0,8 s | 12,1 s | 8,2 s |
| OPT-30B, GSM8K, arranque medio | 7,5 s | 213 s | 199,2 s |
| OPT-30B, pedidos en menos de 300 s | 89% | | 26% |

Con una red 10 veces más rápida, Ray Serve bajaría a 3,8 s: todavía 4,7 veces más lento. KServe tarda 128 s hasta el primer token, 114 de ellos bajando el checkpoint por 1 Gb/s (28 s con las mismas mejoras que Ray Serve). Con carga alta, el p99 del scheduler es 1,27 veces mejor que Shepherd* (que desaloja) y 1,95 veces mejor que el serverless al azar. Con una GPU por servidor ServerlessLLM logra 4 s; Ray Serve con caché necesita cuatro para llegar a 12. En total, entre **10 y 200 veces** menos latencia.

## 5. Keep-alive

Después del último pedido, ¿cuánto tiempo queda el modelo en la GPU? Es un **TTL**: más largo, menos arranques en frío y más GPUs ocupadas sin trabajo, que en la nube es plata. *Serverless in the Wild* propone un keep-alive adaptativo por aplicación en lugar de uno fijo; en la evaluación de ServerlessLLM, el keep-alive de cada sistema es igual a su tiempo de carga.

El valor correcto de un TTL depende de cuánto cuesta un fallo. Con un checkpoint en DRAM y un cargador en pipeline, un arranque cuesta medio segundo: se puede liberar la GPU enseguida y dejar el modelo en DRAM, que es mucho más barata. En la demo del repositorio, la flota además desaloja un modelo ocioso si hace falta su GPU, como una entrada de un caché LRU: TTL y LRU juntos, como en un caché web.

## 6. Muchos fine-tunes: S-LoRA

**LoRA** (Hu et al., 2021). Un fine-tune cambia cada matriz W por

> W′ = W + A·B, con A de h × r, B de r × d y r mucho menor (8, 16, 64)

y se entrenan sólo A y B, típicamente en las proyecciones de query, key, value y salida. Para una matriz de 4096 × 4096 y rango 8 son 65 mil parámetros en lugar de 16 millones. Reduce los parámetros entrenables en órdenes de magnitud (10.000×, por ejemplo) con calidad comparable, y hay cientos de miles de adaptadores publicados. Un adaptador de un modelo de 7B pesa decenas de MB; el modelo, 13 GB.

**Fusionar o calcular aparte** (§4.1).

- **Fusionar:** sumar A·B a W. No cuesta nada por token, pero un lote es de un solo adaptador y cambiar es restar uno y sumar otro con la GPU parada. La versión ingenua, una copia fusionada por adaptador (vLLM-packed), entra menos de 5 veces en una A100 de 80 GB.
- **Calcular aparte:** x·W + x·A·B. La parte cara, x·W, en un solo lote con todos los pedidos; x·A·B, chica, aparte para cada pedido con su adaptador. Lo que se gana en batching es mucho más que lo que cuesta x·A·B. Los adaptadores tienen distinto rango y los pedidos distinto largo, así que en lugar de un batched GEMM con relleno usan kernels propios que juntan pesos de memoria no contigua: MBGMM (prefill) y MBGMV (decode), este último basado en el kernel de **Punica** (Chen et al., 2023), que con la misma idea reporta 12× más throughput sumando unos 2 ms por token.

Es batching otra vez (Clase 18): el decode está limitado por leer los pesos, así que cuantos más pedidos compartan una lectura de W, mejor.

**Unified Paging** (§5.1). Todos los adaptadores viven en la memoria del host; a la GPU se traen los del lote actual. Traer y sacar adaptadores de distinto tamaño fragmenta la memoria, como el KV cache antes de PagedAttention. La observación: un KV cache de S tokens en una capa es S × H; un peso LoRA de rango R es R × H. Comparten H. Entonces una página es un vector de H, y KV cache y adaptadores comparten un único pool paginado que usa toda la memoria que no son los pesos base. Además (§5.2), mientras corre una iteración miran la cola de espera y adelantan los adaptadores del próximo lote.

**S-LoRA también hace tensor parallelism** (§6): parte A y B alineados con el particionado de Megatron del modelo base, y la comunicación extra es 5(N−1)Br/N contra 2(N−1)Bh/N del modelo base, despreciable porque r ≪ h.

**Resultados** (tabla 3, Llama-7B en una A100 de 80 GB, adaptadores de rango 8, pedidos por segundo):

| Adaptadores | S-LoRA | vLLM-packed | PEFT |
| --- | --- | --- | --- |
| 5 | 8,05 | 2,04 | 0,88 |
| 100 | 7,99 | sin memoria | 0,25 |
| 1.000 | 7,64 | sin memoria | — |
| 2.000 | 7,61 | sin memoria | — |

Hasta **4×** más que vLLM-packed y **30×** más que PEFT. Pasados unos 100 adaptadores (figura 5) el throughput deja de bajar: el costo extra depende de los adaptadores de un lote, acotados por el tamaño del lote, no de cuántos existen. Y la ablación honesta (figura 9): con un solo adaptador, fusionar es mejor; con más de dos, gana calcular aparte.

S-LoRA también tiene control de admisión (§4.3): su *early abort* descarta los pedidos que ya no van a llegar al SLO (primer token en 6 s), como el rechazo temprano de Mooncake.

## 7. La lente de la materia

| Hoy | Ya lo vimos como |
| --- | --- |
| Checkpoints en DRAM, SSD y remoto | jerarquía de caché; LRU por nivel |
| Keep-alive | TTL; el costo de un fallo decide el TTL |
| Arrancar donde está el checkpoint | llevar el cómputo a los datos, como MapReduce |
| El mapa de qué está dónde | estado en etcd o ZooKeeper: consenso |
| Migración por rondas | migración en vivo de VMs por pre-copia |
| Rechazar si no llega al SLO | control de admisión (S-LoRA, Mooncake) |
| Flota compartida | multiplexación estadística |

¿Qué consistencia necesita el mapa del scheduler? Si cree que un checkpoint está en DRAM y no está, la carga tarda más pero es correcta: para la eficiencia alcanza con información eventual. Lo que no puede pasar es que el router mande pedidos a una GPU que todavía no tiene el modelo, y por eso ese aviso va recién después de la confirmación del servidor.

## 8. Demo, preguntas y lecturas

**El repositorio.** `servir-tokens` suma el paquete `flota`: una plataforma serverless en tiempo virtual, con eventos (llegadas en ráfagas, cargas por nivel, instancias que atienden hasta 8 pedidos con keep-alive, migración y desalojo), y aparte un servidor con muchos adaptadores LoRA. Los anchos de banda (red de 10 Gb/s, SSD de 6 GB/s, PCIe de 25 GB/s) son del orden del testbed del paper, no medidos.

| Comando | Qué muestra |
| --- | --- |
| `go run ./cmd/sim arranque` | cargas por nivel, ubicación, keep-alive y GPU por modelo |
| `go run ./cmd/sim arranque -gpus 2` | con pocas GPUs: migrar o desalojar |
| `go run ./cmd/sim lora` | fusionado contra lote mixto |
| `go test ./flota` | los tests de lo que dice la clase |

**Carga desde cada nivel** (segundos hasta la GPU):

| | DRAM | SSD | Remoto |
| --- | --- | --- | --- |
| 13 GB, cargador ingenuo | 1,7 | 9,0 | 19,4 |
| 13 GB, cargador en pipeline | 0,5 | 2,2 | 10,4 |
| 130 GB, cargador ingenuo | 17,3 | 89,6 | 193,6 |
| 130 GB, cargador en pipeline | 5,2 | 21,7 | 104,0 |

**Dónde arrancar** (64 modelos, 3.996 pedidos en una hora en ráfagas, 4 servidores de 4 GPUs):

| Política | Frío | Cargas de DRAM | De remoto | Demora media | p99 |
| --- | --- | --- | --- | --- | --- |
| Al azar, cargador ingenuo | 62% | 33% | 34% | 24,5 s | 188 s |
| Al azar | 53% | 31% | 35% | 7,5 s | 80 s |
| Consciente de la localidad | 47% | 70% | 7% | 5,5 s | 83 s |
| Localidad + desalojo (49) | 50% | 68% | 7% | 6,3 s | 102 s |
| Localidad + migración (6) | 48% | 69% | 7% | 5,5 s | 83 s |

Lo que más pesa es el cargador; después, arrancar donde está el checkpoint. Desalojar empeora la media y la cola. Con GPUs de sobra la migración casi no se activa; con 2 GPUs por servidor migra 27 veces y desaloja 114, y la demora media queda en 9,4 s contra 14,6 s. En el paper la migración pesa más porque las inferencias son más largas y las GPUs están llenas. La cola la dominan las ráfagas, con cualquier política.

**Keep-alive:**

| Keep-alive | Clásico: frío | Demora | Con caché local: frío | Demora |
| --- | --- | --- | --- | --- |
| 0 s | 86% | 60 s | 68% | 5,6 s |
| 10 s | 79% | 56 s | 60% | 5,5 s |
| 60 s | 61% | 47 s | 42% | 5,4 s |
| 600 s | 58% | 47 s | 35% | 5,4 s |

Clásico: descarga cada vez, cargador ingenuo, al azar. Con caché local los arranques en frío bajan igual pero la demora casi no cambia: un fallo es barato, así que un keep-alive largo deja de pagar (con 600 s la GPU queda ociosa un 82% del tiempo para ganar 0,2 s). Y contra una GPU por modelo: 64 GPUs ocupadas un 4% del tiempo, contra 16 ocupadas un 16%.

**LoRA** (capacidad en pedidos por segundo, con carga de sobra):

| Adaptadores | Fusionado | Lote mixto | Adaptadores por lote |
| --- | --- | --- | --- |
| 1 | 21,0 | 18,9 | 1 |
| 2 | 15,7 | 18,9 | 2 |
| 100 | 10,4 | 18,0 | 47 |
| 2.000 | 2,2 | 16,9 | 85 |

Con 200 adaptadores a 10 pedidos por segundo, el fusionado atiende 3 y la cola crece sin límite; el lote mixto, los 10 con TTFT p99 de 44 ms. Dejar los adaptadores en el pool los trae una vez: 200 cargas contra 1.545, aunque como traer uno cuesta menos de un milisegundo, la latencia casi no cambia.

El simulador *Flota de GPUs* muestra las 16 GPUs en vivo, con el color de cada modelo y de qué nivel salió cada carga; se puede cambiar la política, el keep-alive y el cargador.

Ejercicios del README: ubicación inicial según popularidad, keep-alive adaptativo y prefetch de adaptadores desde disco.

**Preguntas de repaso.**

1. ¿Por qué con un cargador en pipeline el tiempo de carga es el del nivel más lento y no la suma? *Los niveles trabajan a la vez sobre pedazos distintos: el flujo lo limita el caño más angosto.*
2. ¿Qué estima el scheduler de ServerlessLLM para elegir servidor? *q + n/b: la cola de carga del servidor, el tamaño del checkpoint y el ancho de banda del nivel donde está en ese servidor. Para migrar, α·(t_in + t_out) + β.*
3. ¿Por qué ServerlessLLM manda los tokens y Mooncake el KV cache? *Depende de la red contra el cómputo. Mooncake tiene RDMA de 800 Gb/s y prompts largos; ServerlessLLM, Ethernet de 10 Gb/s, KV caches de gigabytes y un destino libre para recalcular.*
4. Si un arranque desde DRAM cuesta medio segundo, ¿qué pasa con el keep-alive óptimo? *Baja: un fallo es barato, así que conviene liberar la GPU y dejar el modelo en DRAM. El TTL óptimo crece con el costo del fallo.*
5. ¿Por qué el throughput de S-LoRA casi no cae de 100 a 2.000 adaptadores? *El costo extra depende de los adaptadores distintos de un lote, acotados por el tamaño del lote.*
6. ¿Por qué desalojar una inferencia es peor que migrarla? *La desalojada pierde lo que había generado: recarga el modelo y recalcula todo. La migrada sigue generando mientras el destino se pone al día, y sólo para la última ronda.*
7. ¿En qué orden actualiza el scheduler su almacén y avisa al router, y por qué? *Marca la GPU al asignar, actualiza el almacenamiento cuando el servidor confirma, y recién ahí avisa al router: así nunca se rutea a un modelo que todavía no está.*

**Lecturas.**

- Fu et al., [ServerlessLLM: Low-Latency Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351), OSDI 2024. Es el paper de la clase.
- Sheng et al., [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285), MLSys 2024.
- Chen et al., [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547), 2023.
- Li et al., [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665), OSDI 2023.
- Shahrad et al., Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider, USENIX ATC 2020.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.

**Cierre.** Servir muchos modelos en pocas GPUs se reduce a tres decisiones: **acercar** los pesos (DRAM y SSD del servidor, con un cargador que satura cada nivel), **ubicar** cada modelo donde ya está su checkpoint (y mudar, no cortar, lo que estorba) y **compartir** lo que es común (un modelo base para miles de adaptadores). La Clase 21 sigue con cómo repartir un clúster de GPUs entre trabajos distintos.
