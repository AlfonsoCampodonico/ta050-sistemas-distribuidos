# Clase 22 — Entrenar con lo que se sirve

Nov 5, 2026 · @Alfonso Campodonico

## 1. Donde se juntan las dos mitades

Las clases 16 y 17 fueron sobre entrenar: partir un modelo entre muchas GPUs y sobrevivir a las fallas. Las 18 y 19 fueron sobre servir: KV cache, batching continuo, router, prefill y decode. Las 20 y 21, sobre compartir GPUs entre modelos y entre trabajos. Esta clase cierra la serie con el caso en el que todo aparece junto: el **post-entrenamiento con refuerzo** (RL) que convirtió a los modelos de lenguaje en modelos de razonamiento.

El paper de hoy es **HybridFlow** (Sheng et al., EuroSys 2025), cuya implementación abierta es verl. Los complementarios son DeepSeek-R1 (por qué RL y GRPO), Kimi k1.5 (rollouts parciales y despliegue híbrido) y AReaL (RL asincrónico con staleness acotada).

| Clases | Qué se hace | El estado que importa |
| --- | --- | --- |
| 16 y 17 | entrenar: partir el modelo en dimensiones | pesos, gradientes, estado del optimizador |
| 18 y 19 | servir: lotes continuos, router, prefill y decode | el KV cache |
| 20 y 21 | compartir GPUs entre modelos y entre trabajos | qué ocupa cada GPU |
| Hoy | generar, puntuar y actualizar, una y otra vez | los pesos, en dos layouts, y su **versión** |

## 2. El bucle

Un paso de RL tiene tres partes:

1. **Generar.** El modelo actual contesta G veces cada problema de un lote. Es decode, la carga de serving de la Clase 18.
2. **Puntuar.** Un verificador decide qué respuestas están bien: la respuesta final coincide, o el código pasa los tests.
3. **Actualizar.** Un paso de gradiente con esas respuestas: la carga de entrenamiento de la Clase 16.

Y vuelta a generar con los pesos nuevos. A diferencia del pre-entrenamiento, los datos los genera el modelo que se está entrenando: hay una dependencia entre la generación del lote siguiente y el entrenamiento con el actual.

**Por qué desde 2025.** DeepSeek-R1-Zero se entrenó sólo con RL, con recompensas por reglas: exactitud (respuesta correcta, o tests que pasan) y formato (el razonamiento entre `<think>` y `</think>`). No usan un modelo de recompensa neuronal porque "puede sufrir reward hacking" a gran escala. En AIME 2024, R1-Zero pasa de **15,6% a 71,0%** de acierto con una respuesta y llega a **86,7%** votando entre varias (versión de enero de 2025; la revisada reporta 77,9%). DeepSeek-R1 llega a 79,8%; Kimi k1.5, del mismo mes, a 77,5.

**GRPO.** El gradiente de política necesita la ventaja de cada respuesta: si fue mejor o peor de lo esperado. PPO la estima con un crítico, un modelo "típicamente del mismo tamaño que la política". GRPO lo saca: genera G respuestas al mismo problema y usa el grupo como referencia,

> Aᵢ = (rᵢ − media(r₁…r_G)) / desvío(r₁…r_G)

Si todo el grupo acertó o todo falló, la ventaja es cero y ese problema no enseña nada en ese paso. El resto es como PPO (razón entre política nueva y vieja, recortada) más un término de KL contra la política de referencia. La versión revisada de R1 detalla: 16 respuestas por problema, tope de 32.768 tokens que sube a 65.536 desde el paso 8.200, y 512 H800 durante unas 198 horas para R1-Zero.

**Los modelos.** HybridFlow describe el RLHF clásico (PPO) con hasta cuatro modelos y tres etapas:

| Modelo | Qué hace | Etapas |
| --- | --- | --- |
| Actor | la política: genera y aprende | generación, entrenamiento |
| Crítico | estima el valor (PPO; GRPO no lo usa) | preparación, entrenamiento |
| Referencia | la política original, para el KL | preparación |
| Recompensa | puntúa; o un verificador por reglas | preparación |

El actor está en dos etapas con cargas opuestas: generar es decode, limitado por leer los pesos, y quiere poco paralelismo de tensores y muchas réplicas; entrenar está limitado por el cómputo y la memoria del optimizador, y quiere partir el modelo. **El mismo modelo, dos layouts.**

**Las respuestas se alargan.** En R1-Zero el largo medio de las respuestas crece durante todo el RL (figura 3): el modelo aprende solo a pensar más. Kimi k1.5 hace el RL con ventana de 128k tokens. La carga de serving del bucle no es fija: crece, y su cola se alarga.

## 3. El rollout domina

**La cola larga.** Los largos de las respuestas tienen cola larga. Un paso sincrónico no puede entrenar hasta que termina la última respuesta del lote. En la demo (`go run ./cmd/sim cola`, 256 respuestas en 8 GPUs, hasta 64 por GPU):

| Política | Media | p50 | p99 | Máx | 1.ª GPU termina | Última | GPU ociosa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Al empezar | 353 | 305 | 1.165 | 1.388 | 16 s | 30 s | 86% |
| Ya entrenada | 3.822 | 3.478 | 8.192 | 8.192 | 178 s | 192 s | 75% |

"Ociosa" cuenta los lugares vacíos del lote: por la Clase 18, una iteración de decode cuesta casi lo mismo con una secuencia que con 64. Una GPU con una respuesta larga y 63 lugares vacíos está ocupada y produce el 2% de lo que podría. AReaL lo dice así: la generación tiene que esperar a la salida más larga del lote antes de actualizar, y eso subutiliza las GPUs.

**Es la barrera de la Clase 16.** En paralelismo de datos sincrónico, un straggler frena el all-reduce de todos; acá, una respuesta larga frena el paso de todos. Las soluciones se parecen: no esperar al más lento, acotar cuánto se adelanta el rápido, o partir el trabajo del lento.

**Cuánto pesa cada parte** (HybridFlow §2.3 y §8.2):

- Entrenar y generar el actor son el **58,9%** del tiempo de RLHF con HybridFlow.
- En NeMo-Aligner, que genera con el motor de entrenamiento, la generación llega al **81,2%** del paso.
- Con generación y entrenamiento en GPUs distintas, mandar los 140 GB de pesos de un 70B puede llevarse hasta el **36,4%** de una iteración (citando a OpenRLHF).

## 4. Dónde va cada trabajo

**Tres ubicaciones** (HybridFlow §8.3):

| Ubicación | Qué es | Quién |
| --- | --- | --- |
| Colocado | todos los modelos en las mismas GPUs, por turnos | DeepSpeed-Chat; el caso del 3D-HybridEngine |
| Separado | cada modelo en su grupo de GPUs | OpenRLHF (Ray, vLLM, DeepSpeed) |
| Partido | actor y referencia en un grupo, crítico y recompensa en otro | NeMo-Aligner |

Para PPO hay 15 ubicaciones posibles. De 16 a 64 GPUs gana colocar todo junto; con 96 a 128 GPUs y modelos de 34B (o 96 GPUs con 13B) gana partir, y con 13B en 128 GPUs, separar. Colocado no deja GPUs esperando a otras pero no superpone etapas; separado deja a cada modelo usar su paralelismo, pero cada grupo espera al otro.

**Colocado: la transición de layout.** Si el actor genera y entrena en las mismas GPUs, entre una etapa y otra hay que cambiar cómo está partido. En la notación del paper, entrenamiento p-t-d y generación pg-tg-dg-d, con N = p·t·d = pg·tg·dg·d. En el ejemplo de la figura 7, 4 GPUs entrenan con t = 2, d = 2 y generan con tg = 1: cada GPU pasa a tener el modelo entero. Vale la pena: con 16 GPUs y entrenamiento en t = 8, generar con tg = 2 (7B) o tg = 4 (13B) baja la latencia de generación 60,3% y 36,4%.

La transición es un all-gather (el colectivo de ZeRO, Clase 16), y la tabla 2 compara tres formas:

| Transición | Volumen | Memoria repetida | Ejemplo t = 8, d = 2 → tg = 2 |
| --- | --- | --- | --- |
| DS-Chat | (tpd − 1)/(tpd) · M | M/(tpd) | 0,94 M |
| HybridFlow-V | (tp − 1)/(tp) · M | M/(tp) | 0,88 M |
| HybridFlow | (tp − tg·pg)/(tg·pg·tp) · M | 0 | 0,38 M |

El **3D-HybridEngine** elige los grupos de generación para que cada GPU ya tenga parte de lo que necesita, junta sólo lo que le falta y no deja copias. Medido: el tiempo de transición baja **55,2%** en promedio (11,7 s) y hasta **89,1%** (78,2 s) con modelos de 70B. En total, HybridFlow reporta entre **1,53 y 20,57** veces más throughput que DeepSpeed-Chat, OpenRLHF y NeMo-Aligner, en 128 A100 (16 máquinas de 8, NVLink de 600 GB/s, 200 Gb/s entre máquinas).

**Separado: un broadcast por paso.** Con pools separados, el trainer manda los pesos a las GPUs de rollout después de cada paso. Con 140 GB a 25 GB/s (`go run ./cmd/sim ubicacion`):

| GPUs de rollout | Ingenuo (una copia a cada una) | En anillo / pipeline |
| --- | --- | --- |
| 6 | 34 s | 5,9 s |
| 48 | 269 s | 8,2 s |
| 384 | 2.150 s | 27 s |

Es el argumento del ring all-reduce de la Clase 16: partido en pedazos y reenviando mientras se recibe, el tiempo casi no depende de n.

**Kimi k1.5, un caso intermedio.** Megatron (entrenamiento) y vLLM (inferencia) corren en contenedores separados del mismo pod de Kubernetes, compartiendo las GPUs. Un proceso que llaman checkpoint engine pasa los pesos de Megatron a vLLM, y entre nodos los mueve **Mooncake** por RDMA, el sistema de la Clase 19. vLLM arranca con pesos de mentira y los reemplaza. Reportan menos de un minuto de entrenar a generar y unos diez segundos al revés.

**La demo, todo sincrónico** (`go run ./cmd/sim ubicacion`, 8 GPUs):

| Ubicación | s por paso | GPU ociosa | En qué se va |
| --- | --- | --- | --- |
| Colocado | 217 | 69% | generar 89%, entrenar 10%, transición 1% |
| Separado, 6 + 2 | 279 | 76% | rollout ociosas 77%, trainer 70% |
| Separado, 4 + 4 | 237 | 71% | rollout ociosas 60%, trainer 82% |

Sincrónico, separar sólo agrega espera: mientras unas generan, las otras miran.

## 5. Asincrónico

**La idea.** Dejar que el rollout genere el lote siguiente mientras el trainer entrena con el actual. Ningún pool espera, pero el lote con el que se produce v2 lo generó v0: los datos tienen **staleness**. Es aprendizaje off-policy: el gradiente de política supone que las muestras salen de la política que se actualiza, y si salen de una vieja está sesgado. Es lo que pasaba con un parameter server asincrónico, donde un worker calculaba su gradiente con pesos que ya habían cambiado.

**Stale Synchronous Parallel** (Ho et al., NIPS 2013). Cada worker tiene un reloj; el más rápido y el más lento no pueden estar a más de s relojes; un worker en el reloj c ve seguro todas las actualizaciones hasta c − s − 1. Con s = 0 es Bulk Synchronous Parallel.

| | Parameter server con SSP | RL asincrónico (AReaL) |
| --- | --- | --- |
| Los rápidos | workers que calculan gradientes | GPUs de rollout que generan |
| El estado compartido | los parámetros en el servidor | los pesos del trainer |
| El límite | a lo sumo s relojes del más lento | a lo sumo η versiones adelante |
| Con límite 0 | BSP | RL sincrónico |

AReaL tiene un hiperparámetro η, la staleness máxima. El controlador no deja empezar una respuesta si ⌊(N_r − 1)/B⌋ > i + η, con N_r las respuestas pedidas, B las de un lote e i la versión actual del trainer (ecuación 3). Con η = 0, dice el paper, degenera en RL sincrónico.

**Lo honesto:** el límite se aplica cuando una respuesta empieza. Una respuesta larga que empezó temprano termina tarde, y se entrena con más staleness que η. En la demo, con η = 4 el promedio es 3,6 y el máximo 10.

**AReaL** (Fu et al., 2025), tres piezas:

- **Rollouts interrumpibles.** Cuando llegan pesos nuevos, se interrumpe la generación, se descarta el KV cache calculado con los viejos, se recalcula con los nuevos y se sigue. Una respuesta puede tener pedazos de varias versiones. Sólo eso sube el throughput 12% y 17% con modelos de 1,5B y 7B.
- **PPO desacoplado.** Separa la política que generó (behavior) de la que sirve de ancla para el recorte (proximal, una reciente). Si el ancla fuera la vieja, tiraría la política nueva hacia versiones viejas y peores.
- **Sin barrera.** Tres cuartos de las GPUs generan y un cuarto entrena; η = 8 para matemática y 4 para código.

Resultado: hasta **2,77×** menos tiempo de entrenamiento que su versión sincrónica con las mismas GPUs (1,5B: 41,0 h contra 14,8 h), con igual o mejor resultado final, en un clúster de 512 H800 y modelos de 1,5B a 32B.

**Qué le hace la staleness al aprendizaje** (AReaL, tabla 2, 1,5B, AIME 2024):

| η | PPO común | Desacoplado |
| --- | --- | --- |
| 0 (sincrónico) | 42,0 | 42,0 |
| 1 | 41,8 | 42,1 |
| 2 | 40,0 | 41,8 |
| 4 | 23,3 | 42,2 |
| 8 | 35,7 | 41,0 |
| 16 | 35,8 | 38,7 |
| sin límite | 34,0 | 36,9 |

Con el objetivo desacoplado, "una staleness moderada (η ≤ 8) tiene un impacto mínimo en el rendimiento final". La columna del medio no es monótona: son corridas únicas, y un RL que se desestabiliza no siempre se recupera igual. Lo robusto es la diferencia entre columnas.

**La demo, asincrónico** (`go run ./cmd/sim asinc`, 8 GPUs, 40 pasos; 6 generan y 2 entrenan, interrumpibles y con corrección):

| | Tiempo | Resp./s | GPU ociosa | Staleness media | Máx | vs colocado |
| --- | --- | --- | --- | --- | --- | --- |
| Colocado, sincrónico | 8.692 s | 1,18 | 69% | 0 | 0 | 1,00× |
| Separado, sincrónico | 11.141 s | 0,92 | 76% | 0 | 0 | 0,78× |
| η = 1 | 5.785 s | 1,77 | 53% | 0,97 | 2 | 1,50× |
| η = 2 | 3.867 s | 2,65 | 33% | 1,92 | 5 | 2,25× |
| η = 4 | 2.901 s | 3,53 | 12% | 3,60 | 10 | 3,00× |
| η = 8 | 2.874 s | 3,56 | 8% | 3,95 | 10 | 3,02× |

**La demo, aprender** (`go run ./cmd/sim aprender`). La tarea: el modelo decide cuánto pensar; pensar más acierta más, pero pasar los 8.192 tokens vale cero. El óptimo está en el medio (mediana de unos 3.300 tokens, recompensa 0,799).

| η | Staleness | Sin corrección | Con corrección |
| --- | --- | --- | --- |
| 0 | 0 | 0,795 | 0,795 |
| 1 | 0,97 | 0,791 | 0,796 |
| 2 | 1,92 | 0,695 | 0,794 |
| 4 | 3,30 | 0,663 | 0,794 |
| 8 | 4,41 | 0,205 | 0,792 |
| sin límite | 7,00 | 0,000 | 0,792 |

¿Por qué? Un gradiente calculado con muestras de una política vieja es, en promedio, el gradiente de la política vieja. Cuando la vieja pensaba poco, ese gradiente dice "pensá más" aunque la actual ya esté en el óptimo: se pasa del tope, las respuestas se cortan y valen cero, y si todas valen cero GRPO no tiene señal y no se recupera. Es la inestabilidad clásica de los gradientes con demora. Con **corrección por importancia** (cada muestra pesa π_actual/π_vieja, truncado en 2; la misma idea que el objetivo desacoplado) todos los η llegan a la misma recompensa, a cambio de algunos pasos más.

**Rollouts parciales (Kimi k1.5).** Kimi es colocado. Para que una respuesta larga no frene todo, fija un presupuesto de tokens por vuelta: si una respuesta lo pasa, "la parte sin terminar se guarda en el replay buffer y sigue en la próxima iteración". Sólo el pedazo actual se calcula con la política actual; los anteriores se reusan. Así "ninguna trayectoria larga monopoliza los recursos del sistema". Es asincronía de otra forma, con la staleness concentrada en las respuestas largas. El paper no publica una ablación con números. En la demo (`go run ./cmd/sim parciales`):

| | Tiempo | GPU ociosa | Staleness | Recompensa |
| --- | --- | --- | --- | --- |
| Colocado, sincrónico | 8.692 s | 69% | 0 | 0,795 |
| Colocado, rollouts parciales | 5.121 s | 47% | 0,97 | 0,787 |
| … y con corrección | 5.108 s | 48% | 0,97 | 0,795 |
| Separado, η = 4, termina lo que tiene | 4.251 s | 41% | 2,30 | 0,791 |
| Separado, η = 4, interrumpible | 2.901 s | 12% | 3,60 | 0,794 |

## 6. Cuando algo falla

| Pieza | Si se cae | Qué se hace |
| --- | --- | --- |
| Una GPU de rollout | se pierden respuestas a medias | se vuelven a generar: no tienen efectos, como un map de MapReduce |
| El verificador | un sandbox que cuelga o tarda | timeout y reintento; es un servicio aparte (Kimi describe un servicio de ejecución de código como parte del modelo de recompensa) |
| El trainer | se pierde el estado del optimizador | checkpoints, como en la Clase 17 |
| Los pesos en viaje | un rollout queda con una versión vieja | la versión viaja con cada respuesta: se detecta y se acota |

Sincrónico, una GPU de rollout que se cae frena el paso como un straggler infinito. Asincrónico, el controlador tiene un worker menos: menos throughput, no una barrera rota.

## 7. Cerrar la serie

| | Entrenar (16–17) | Servir (18–19) | El bucle de RL (22) |
| --- | --- | --- | --- |
| Qué se parte | el modelo, en 4 dimensiones | el KV cache, en bloques y niveles | el clúster, entre generar y entrenar |
| Qué se comunica | gradientes y activaciones | KV cache entre prefill y decode | pesos, después de cada paso |
| A quién se espera | al straggler, en cada all-reduce | al prefill largo, en cada iteración | a la respuesta más larga, en cada paso |
| Si algo falla | volver al checkpoint | recalcular el pedido | regenerar; checkpoint del trainer |

**Para llevarse.**

- **No esperar a la cola.** La respuesta más larga decide el paso sincrónico; rollouts parciales o asincrónicos sacan esa espera.
- **Mover los pesos bien.** Colocado: cambiar de layout juntando sólo lo que falta. Separado: broadcast en anillo, no uno por uno.
- **Acotar lo viejo.** Staleness acotada, como SSP, y una corrección para que los datos viejos no empujen de más.

## 8. Demo, preguntas y lecturas

**El repositorio.** `post-entrenamiento` simula el bucle en tiempo virtual: GPUs de rollout con batching continuo (una iteración cuesta 20 + 0,2·n ms), un trainer, los pesos que viajan, y un problema de juguete que aprende de verdad.

| Paquete | Qué tiene |
| --- | --- |
| `tarea` | la política que elige cuánto pensar, la recompensa por verificador y GRPO |
| `pesos` | broadcast ingenuo y en anillo; los volúmenes de la tabla 2 de HybridFlow |
| `bucle` | colocado o separado, sincrónico o con η, parciales e interrumpibles, con o sin corrección |

| Comando | Qué muestra |
| --- | --- |
| `go run ./cmd/sim cola` | un paso espera a la respuesta más larga |
| `go run ./cmd/sim ubicacion` | colocado vs separado, y mover los pesos |
| `go run ./cmd/sim asinc` | staleness acotada: throughput y espera |
| `go run ./cmd/sim parciales` | rollouts parciales e interrumpibles |
| `go run ./cmd/sim aprender` | qué le hace lo viejo al aprendizaje |
| `go test ./...` | los tests de lo que dice la clase |

Los tests verifican que un paso sincrónico espera a la respuesta más larga y deja ociosa la mayor parte del GPU·tiempo; que con η = 0 no hay staleness; que más η da más throughput y más staleness; que el asincrónico termina en menos de la mitad del tiempo con la misma recompensa; que sin corrección η = 8 arruina el aprendizaje y con corrección no; que los parciales y las interrupciones aceleran; y que las respuestas se alargan al aprender. El simulador *Bucle de RL* muestra los lotes de cada GPU, el trainer, los pesos que viajan y un slider de η en vivo.

Ejercicios del README: descartar lo viejo en lugar de usarlo, PPO con recorte, un ratio de GPUs adaptativo, fallas de una GPU de rollout y muestreo dinámico de grupos con señal.

**Preguntas de repaso.**

1. ¿Por qué un paso de RL tiene una carga de serving y una de entrenamiento? ¿Qué depende de qué? *Generar es decode con el modelo actual; actualizar es un paso de entrenamiento. El entrenamiento necesita las respuestas, y la generación siguiente necesita los pesos nuevos.*
2. ¿Qué le saca GRPO a PPO, y por qué importa para el sistema? *El crítico, típicamente del tamaño de la política: la ventaja se calcula contra el grupo. Un modelo menos para entrenar, guardar y mover; a cambio, el trainer necesita el grupo completo.*
3. ¿Por qué un paso sincrónico tiene la GPU ociosa la mayor parte del tiempo aunque todas "estén trabajando"? *El paso espera a la respuesta más larga, y el decode cuesta casi lo mismo con una secuencia que con 64: las GPUs con pocas respuestas están ocupadas pero producen poco.*
4. Colocado o separado: ¿qué se paga para llevar los pesos a donde se genera? *Colocado, la transición de layout, un all-gather que se puede diseñar para mover poco. Separado, un broadcast por paso, que tiene que ser en anillo o en pipeline para escalar.*
5. ¿Qué es η en AReaL y qué es s en SSP? ¿Por qué el máximo medido puede pasar a η? *Cuántas versiones puede adelantarse la generación, y cuántos relojes el worker más rápido. Con 0, los dos son sincrónicos. El límite se aplica al empezar una respuesta, y una larga termina tarde.*
6. Sin corrección, ¿por qué la staleness hace que el modelo de la demo se pase del tope? *Porque el gradiente de muestras viejas es el de la política vieja, que todavía pedía pensar más.*
7. ¿Qué tienen en común los rollouts parciales de Kimi y los interrumpibles de AReaL? *Una respuesta queda hecha de pedazos generados por versiones distintas, y ninguna respuesta larga frena al resto.*

**Lecturas.**

- Sheng et al., [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256), EuroSys 2025, y [verl](https://github.com/volcengine/verl). Es el paper de la clase.
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), 2025.
- Kimi Team, [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599), 2025.
- Fu et al., [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298), 2025.
- Hu et al., [OpenRLHF](https://arxiv.org/abs/2405.11143).
- Wu et al., [LlamaRL](https://arxiv.org/abs/2505.24034), 2025.
- Ho et al., More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server, NIPS 2013.

**Cierre de la serie.** En la 16 partimos un modelo para entrenarlo; en la 18 lo servimos; en la 22 vimos que, para los modelos de razonamiento, entrenar es servir miles de veces: el mismo modelo, en dos layouts, yendo y viniendo, con una versión pegada a cada dato. Las preguntas siguen siendo las mismas: qué se parte, qué se comunica, qué pasa cuando falla.
