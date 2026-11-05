# Clase 21 — Repartir un clúster de GPUs

Nov 5, 2026 · @Alfonso Campodonico

## 1. Del trabajo con todas las GPUs al clúster que las reparte

En las clases 16 y 17 el entrenamiento ya tenía sus GPUs: Llama 3 en 16.384, todas para él. En la realidad, un clúster de GPUs lo comparten cientos de usuarios y decenas de equipos, con trabajos de una GPU y trabajos de cientos, y alguien tiene que decidir **quién corre, cuándo y en qué GPUs**.

El paper de hoy es el análisis de **Philly**, el clúster de entrenamiento de Microsoft (Jeon et al., USENIX ATC 2019): la primera traza grande de un clúster de deep learning, la que usan casi todos los planificadores que vinieron después. Lo acompañan **Tiresias** (Gu et al., NSDI 2019), que planifica sin conocer la duración de los trabajos, **Gandiva** (Xiao et al., OSDI 2018), que reparte las GPUs en el tiempo, y el análisis del clúster de Alibaba, **MLaaS in the Wild** (Weng et al., NSDI 2022).

Es una clase de planificación, con problemas clásicos y dos vueltas nuevas: los trabajos piden muchas GPUs a la vez, y dónde corren cambia cuánto tardan.

| Parte | La pregunta | Lo clásico |
| --- | --- | --- |
| La carga | ¿quién pide GPUs, cuántas, por cuánto? | trazas de clústeres |
| Todo o nada | ¿qué pasa si un trabajo recibe sólo parte de sus GPUs? | deadlock, bloqueo de cabeza de cola |
| Localidad | ¿esperar el lugar bueno o arrancar en uno peor? | fragmentación |
| Sin duraciones | ¿a quién atender primero si nadie sabe cuánto tarda? | SRTF, LAS, preempción |
| Compartir | ¿y entre equipos, o una GPU entre varios? | time-sharing, cuotas |

## 2. La carga

**Philly** (Jeon et al., §2): 75 días de logs, de octubre a diciembre de 2017, con **96.260 trabajos** de **14 clústeres virtuales**. Un clúster virtual es la porción de un equipo, con su cuota de GPUs. El paper no da el número exacto de GPUs: cientos de servidores y miles de GPUs del mismo modelo, en dos tipos de servidor, de 2 y de 8 GPUs. El planificador es YARN extendido para GPUs, con gang scheduling y localidad.

El dato que sorprende: casi todas las GPUs están asignadas, pero **las asignadas se usan en promedio sólo un 52%**. Las dos razones que encuentra el paper son de esta clase: trabajos repartidos en muchos servidores, que esperan a la red, y la interferencia entre trabajos que comparten un servidor.

**La forma.** Muchísimos trabajos chicos y unos pocos enormes:

- Más del 82% de los trabajos de Philly corría en una sola GPU (lo resume Weng et al., §2).
- La duración mediana fue de **26 minutos**, y el p90 de **25 horas** (Weng et al., §3.2).
- El **0,5%** de los trabajos tardó más de una semana, y los que piden más GPUs tienden a durar más (Jeon et al., §2.4).

Es una distribución de **cola pesada**: la mediana dice poco. En big data, en cambio, las tareas duran de milisegundos a pocas horas.

La carga sintética del repositorio tiene esa forma (los porcentajes son ilustrativos, no de Philly):

| GPUs | Trabajos | GPU·tiempo |
| --- | --- | --- |
| 1 | 70% | 10% |
| 2 a 8 | 24% | 46% |
| más de 8 | 6% | 45% |

El 1% de los trabajos que más piden se lleva la mitad del GPU·tiempo (GPUs por duración: lo que un trabajo le cuesta al clúster). Si se optimiza la espera promedio por trabajo, se favorece a los chicos, que son casi todos; si se optimiza el uso, a los grandes, que son casi todo el GPU·tiempo.

**Nadie sabe cuánto va a durar su trabajo, ni si va a terminar** (Jeon et al., §4, tabla 6):

| Cómo terminó | Trabajos | GPU·tiempo |
| --- | --- | --- |
| Bien | 69,3% | 44,5% |
| Lo mató el usuario | 13,5% | 37,7% |
| Falló | 17,2% | 17,8% |

El 30,7% de los trabajos no terminó bien y se llevó alrededor del 55% del GPU·tiempo. Matar no siempre es malo: muchos entrenan hasta ver que la curva no mejora, o descartan configuraciones en una búsqueda de hiperparámetros. Las fallas son sobre todo errores del usuario (la más frecuente, quedarse sin memoria en la CPU; después, entradas incorrectas y errores en el código); las de infraestructura son el 6,2% de los intentos. Los trabajos de más de 4 GPUs reintentan más y fallan más.

Las tres recomendaciones del paper, que recorren la clase: (1) como los trabajos corren mucho, el planificador debería **cambiar espera por localidad**; (2) aislar los trabajos en servidores dedicados, con migración para desfragmentar; (3) **atrapar las fallas temprano**, antes de pedir un clúster grande.

## 3. Todo o nada

**Gang scheduling.** Como vimos en la Clase 16, el all-reduce es una barrera: con 500 de 512 GPUs, el trabajo no da ni un paso. Si un orquestador ubica cada pod por su cuenta, dos trabajos de 8 GPUs en un clúster de 8 pueden quedar con 4 cada uno, esperando las del otro para siempre. Están las cuatro condiciones de Coffman: exclusión mutua, **retener y esperar**, sin desalojo y espera circular.

Gang scheduling rompe retener y esperar: el trabajo se admite entero o se queda en la cola sin tomar nada, como pedir todos los locks de una transacción de una vez. Philly lo dice textual: los frameworks necesitan todas las GPUs a la vez, así que el planificador hace gang scheduling y además trata de respetar la localidad (§2.3).

**Bloqueo de cabeza de cola.** Con gang y FIFO estricto, un trabajo de 32 GPUs al frente de la cola, con 20 libres, espera; y los de 1, 2 o 4 GPUs de atrás también, aunque entran. Las GPUs libres quedan sin usar.

**Backfill.** Si el primero no entra, pasan los que entran. Philly lo hace: el 38,1% de las decisiones fueron fuera de orden, y para los trabajos de más de 4 GPUs, todas (§3.1.1). El análisis de Alibaba encuentra el mismo problema: alrededor del 9% de las instancias cortas pasó más de la mitad de su tiempo esperando.

**El precio: hambre.** Con backfill codicioso, cada vez que se libera una GPU la toma un trabajo chico, y el grande nunca junta las suyas. La solución clásica de HPC es **EASY backfilling** (Lifka, 1995): el primero reserva la hora en que va a entrar y sólo pasan los que terminan antes. Necesita que cada trabajo declare su duración, que en deep learning nadie sabe. Philly, además, recupera la cuota de un equipo con preempción sólo cuando el clúster pasa el 90% de uso.

**La demo** (`go run ./cmd/sim gang`): 2.000 trabajos en 256 GPUs (4 racks de 8 servidores de 8), al 90% de carga.

| Planificador | JCT prom. | Espera 1 GPU | Espera 2–8 | Espera >8 | Uso |
| --- | --- | --- | --- | --- | --- |
| De a un pod, sin gang | 29,7 h | 24,5 h | 26,2 h | 27,2 h | 72% |
| FIFO, gang | 10,0 h | 6,4 h | 6,4 h | 6,6 h | 68% |
| FIFO + backfill, gang | 3,8 h | 14 s | 8 min | 3,2 h | 74% |

JCT es el tiempo de completar un trabajo, desde que llega hasta que termina. Sin gang no llega a trabarse del todo (siempre termina algún trabajo chico), pero las GPUs retenidas por trabajos que esperan a las que les faltan suman unas 6.000 GPU·horas. Con FIFO estricto, todos esperan más de 6 horas con un 32% del clúster sin asignar. Con backfill, los chicos esperan segundos y los de más de 8 GPUs, más de 3 horas. `go run ./cmd/sim traba` muestra el deadlock mínimo: 8 GPUs y dos trabajos de 8.

## 4. Localidad y fragmentación

**Por qué importa dónde.** Dentro de un servidor, el all-reduce va por NVLink; entre servidores, por la red, y el anillo va a la velocidad del enlace más lento (Clase 16). Philly lo mide (§3.2.1): los trabajos de 16 GPUs usan sus GPUs un 43,7% en promedio si están en 2 servidores, 40,9% en 4 y **28,6% en 8** (tabla 5). Los de 8 GPUs en un servidor, 56,9%; los de 16, 34,3%. Además, los que comparten servidor con trabajos ajenos se interfieren: CPU, memoria del host, PCIe, red.

**El dilema.** En el clúster virtual 2 de Philly, el 25% de los trabajos de más de 4 GPUs esperó 10 minutos o más para arrancar, contra el 10% de los de 1 GPU. El paper separa dos causas de espera (tabla 2):

| GPUs | Por la cuota | Por fragmentación |
| --- | --- | --- |
| 2 a 4 | 40,6% | 59,4% |
| 5 a 8 | 25,8% | 74,2% |
| más de 8 | 2,1% | 97,9% |

**Fragmentación**: hay GPUs libres de sobra, pero no en un servidor o en un rack, que es lo que pide la localidad. Medido en tiempo, alrededor del 80% de la espera de todos los trabajos es por fragmentación. Con el clúster a dos tercios de uso, menos del 4,5% de los servidores está completamente vacío. Es la fragmentación externa de la memoria, con servidores en lugar de páginas, y tiene las mismas soluciones: ubicar con best-fit (el trabajo chico va al servidor más lleno donde entra) y compactar, que acá es migrar.

**Lo que hace Philly.** Si no hay lugar con buena localidad, el trabajo espera unos minutos (2 a 3), reintenta, y después de un número fijo de reintentos acepta servidores repartidos. Los trabajos de más de 8 GPUs repartidos en 16 servidores arrancan mucho antes que los que esperaron 2 o 4 (figura 4), pero corren peor.

**Lo que propone Tiresias: consolidar sólo si importa** (§2.2). La sensibilidad a la localidad depende de cómo están repartidos los parámetros en tensores, porque cada tensor viaja como un mensaje en la agregación de gradientes. En VGG19 el tensor más grande es el 71,5% del modelo; en AlexNet, el 61,0%; en ResNet-50, el 9,2% (tabla 6). Los modelos con un tensor gigante sufren al repartirse; los demás, poco. Tiresias perfila el modelo sin preguntarle al usuario y sólo consolida a los sensibles; contra ubicar al azar, la mejora en tiempo de entrenamiento llega a 1,67 veces.

**La demo** (`go run ./cmd/sim localidad`), con FIFO + backfill. La mitad de los trabajos de varias GPUs es sensible y, repartida, corre al 60% (un número ilustrativo, del orden de lo que mide Tiresias).

| Ubicación | JCT prom. | Espera >8 | … por fragmentación | Más lento | Uso útil |
| --- | --- | --- | --- | --- | --- |
| Consolidada siempre | 3,82 h | 3,6 h | 39% | 1,00× | 73% |
| Donde haya lugar | 4,00 h | 2,3 h | 0% | 1,11× | 71% |
| Relajar tras 10 min (Philly) | 4,15 h | 3,3 h | 0% | 1,12× | 71% |
| Consolidar si importa (Tiresias) | 3,82 h | 3,2 h | 15% | 1,01× | 73% |

Repartir acorta la espera pero alarga la ejecución, y como los trabajos corren horas, pierde: es la primera recomendación de Jeon et al. Consolidar sólo a los sensibles da el JCT de consolidar siempre con menos espera para los grandes. Lo honesto: los números dependen del modelo de velocidad y de la carga; con otras muestras, consolidada siempre y consolidar si importa a veces se invierten. Lo que no cambia es que repartir a todos pierde.

## 5. Sin conocer las duraciones

**SRTF y LAS.** Atender primero al que menos tiempo le falta (SRTF, *shortest remaining time first*) minimiza el tiempo de respuesta promedio, pero necesita saber cuánto le falta a cada trabajo. **LAS** (*least attained service*) atiende primero al que **menos servicio recibió**, un dato que el planificador sí tiene. Funciona por la cola pesada: un trabajo que lleva 5 minutos probablemente sea de los muchos cortos; uno que lleva 3 días, de los pocos que duran semanas. Es el clásico foreground-background de los sistemas de tiempo compartido.

**2D-LAS (Tiresias, §3.2.2).** El servicio tiene dos dimensiones: un trabajo de 32 GPUs que corrió una hora le costó al clúster lo mismo que uno de 1 GPU que corrió 32 horas.

> prioridad = − GPUs × tiempo corriendo

**Discretizar (§3.2.3).** LAS puro cambia de prioridad a cada rato y obligaría a desalojar todo el tiempo. Tiresias usa K colas con umbrales de GPU·tiempo: un trabajo empieza en la cola 1 y baja al pasar el umbral. Entre colas, prioridad estricta con preempción; dentro de cada cola, **FIFO por la hora en que arrancó por primera vez** (por llegada, un trabajo que todavía no arrancó desalojaría a uno que ya corre, sin ganar nada). Con **K = 2** alcanza: con 3 o 4 colas el JCT promedio mejora sólo un 1% (§5.4.2). En su testbed, el umbral es de 3.200 GPU·segundos. Contra el hambre, PROMOTEKNOB: si un trabajo esperó K veces lo que corrió, vuelve a la cola 1 (el *aging* de los planificadores de procesos).

**Desalojar cuesta un checkpoint.** Sacarle las GPUs a un trabajo es guardar su estado y volver a cargarlo, como los checkpoints de las clases 16 y 17. En el testbed de Tiresias, 221 preempciones costaron 13.724 segundos: **unos 62 s cada una** (§5.2.4). Gandiva lo hace mucho más barato suspendiendo en el borde de un mini-batch, cuando la memoria de la GPU es mínima (midió 77 veces de diferencia entre máximo y mínimo): menos de 100 ms para clasificación de imágenes, hasta 1 s para traducción; con turnos de un minuto, un 2% o menos.

**Resultados de Tiresias** (60 GPUs P100, 15 máquinas de 4): contra YARN-CS, el planificador de capacidad que usaba Philly, el JCT promedio mejora **5,5 veces** y el mediano 27; la espera promedio baja de más de 8.000 segundos a unos 1.000, y la mitad de los trabajos espera 13 segundos o menos; el makespan baja 1,21 veces. En simulación con la traza de Philly (tabla 5): 2,4 veces mejor que YARN-CS, 1,5 contra best-effort y 2 contra Gandiva, y **el mismo JCT promedio que SRTF**, que conoce las duraciones.

**La demo** (`go run ./cmd/sim duraciones`), misma carga, todos consolidando sólo si importa:

| Planificador | JCT prom. | Espera 1 | Espera >8 | Preempciones |
| --- | --- | --- | --- | --- |
| FIFO | 10,03 h | 6,4 h | 6,6 h | 0 |
| FIFO + backfill | 3,82 h | 14 s | 3,2 h | 0 |
| 2D-LAS, umbral 3.200 GPU·s | 4,12 h | 29 min | 24 s | 2.642 |
| 2D-LAS, umbral 300.000 GPU·s | 3,70 h | 2 s | 23 min | 569 |
| SRTF (conoce las duraciones) | 3,67 h | 0 s | 47 min | 1.386 |

Con un umbral adecuado a la carga (300.000 GPU·s, unas 80 GPU·horas), 2D-LAS le gana al backfill y queda a un 1% de SRTF. Con el umbral del testbed de Tiresias, en esta carga de trabajos de horas, casi todo baja de cola enseguida (uno de 32 GPUs a los 100 segundos) y se desaloja más de cuatro veces más: peor que backfill. El umbral se elige mirando la carga. 2D-LAS también arregla el hambre del backfill: un trabajo grande nuevo tiene prioridad alta mientras recibió poco, y espera 23 minutos en lugar de 3,2 horas.

El costo de cada preempción, con el umbral bueno: gratis 3,70 h; 1 min 3,70 h; 10 min 3,76 h; 30 min 3,91 h. Lo honesto: contra backfill la mejora es chica (3%); la grande es contra FIFO estricto. En el paper pasa lo mismo: 5,5 veces contra YARN-CS, 1,5 contra best-effort.

## 6. Compartir

**En el tiempo: Gandiva (OSDI 2018).** El entrenamiento es predecible: cada mini-batch se parece al anterior. Gandiva lo usa para tres cosas: **time-slicing** (los trabajos se turnan las GPUs de a un minuto, suspendiendo entre pasos, y un trabajo nuevo da su primera señal en minutos en lugar de esperar horas), **empaquetar** (dos trabajos chicos en una GPU si no se molestan) y **migrar** (mover un trabajo en segundos, para desfragmentar o mejorar la localidad). En un clúster de 180 GPUs, el uso promedio pasa de 50,1% a 62,8%, un 26% más; en búsqueda de hiperparámetros explora casi 10 veces más configuraciones. Busca equidad y aprovechamiento más que JCT: en la simulación de Tiresias queda 2 veces peor en JCT.

**Una GPU entre varios: MLaaS in the Wild (Alibaba PAI, NSDI 2022).** Dos meses de 2020 en **6.742 GPUs** (1.295 máquinas de 2 GPUs y 519 de 8), más de 1,2 millones de tareas y 7,5 millones de instancias. Mucho más heterogéneo que Philly: entrenamiento, inferencia, notebooks. La instancia mediana usa **0,042 GPUs** y la mediana pide 0,5. Con GPU sharing alcanza en promedio con el **50%** de las GPUs, y en la hora pico se ahorra hasta el 73%; la interferencia es rara. Y al revés: reservar las máquinas buenas para los trabajos grandes y empaquetar a los chicos en las otras bajó la espera de los grandes un 68% y la de todos un 45%. Hoy las GPUs se pueden partir en instancias aisladas (MIG) o compartir por tiempo.

**Entre equipos: cuotas.** Philly tiene una cuota por clúster virtual; lo que un equipo no usa se presta, y se recupera con preempción sólo con el clúster por encima del 90%. La espera por cuota es la otra columna de la tabla 2: para los trabajos de 2 a 4 GPUs, el 40,6% de las esperas.

En Kubernetes:

- **Kueue** hace admisión por cuotas: una ClusterQueue por equipo, y las colas de un mismo cohort se prestan lo que no usan. Para el todo o nada, **waitForPodsReady**: si no todos los pods del trabajo están listos dentro de un tiempo configurado, el trabajo se desaloja y vuelve a la cola. La documentación lo llama una implementación simple de todo o nada: no evita retener y esperar, lo corta con un timeout, como la recuperación de un deadlock.
- **Volcano** es un scheduler alternativo con gang scheduling: el grupo de pods se ubica entero o no se ubica.

**Hoy, a escala: MAST (Meta, OSDI 2024).** Decenas de regiones, millones de máquinas, decenas de miles de GPUs. Antes, cada usuario elegía a mano la región donde guardar sus datos y correr (tienen que estar juntos), y esas decisiones aisladas sobrecargaban algunas regiones mientras otras sobraban. Con MAST el trabajo va a un planificador global que decide dónde van los datos y el trabajo; sus principios son desacoplar en el tiempo, desacoplar el alcance y búsqueda exhaustiva. En la región más cargada, la demanda de GPUs de alta prioridad pasó de **2,63 veces la oferta a 0,98**.

**Entrenar y servir en la misma flota.** El serving sigue a los usuarios, con picos de día y valles de noche; las GPUs que sobran de noche pueden entrenar, si el planificador puede devolverlas a tiempo, con preempción y checkpoints. DeepSeek reduce los nodos de inferencia de noche y usa esas GPUs para otra cosa (Clase 19). Es el tema de la Clase 22.

## 7. Demo, preguntas y lecturas

**El repositorio.** `planificar-gpus` es un simulador de eventos discretos de un clúster de 256 GPUs, en Go sin dependencias. Paquetes: `cluster` (topología, ubicación con best-fit, fragmentación y velocidad según dónde quedó el trabajo), `trace` (la carga sintética) y `sched` (el simulador y los planificadores). Una corrida de 2.000 trabajos tarda milisegundos.

| Comando | Qué muestra |
| --- | --- |
| `go run ./cmd/sim carga` | la forma de la carga |
| `go run ./cmd/sim traba` | dos trabajos sin gang, trabados |
| `go run ./cmd/sim gang` | sin gang, FIFO y backfill |
| `go run ./cmd/sim localidad` | consolidar, repartir o relajar |
| `go run ./cmd/sim duraciones` | FIFO, 2D-LAS y SRTF, con el costo de preempción |
| `go test ./...` | los tests de lo que dice la clase |

Los tests verifican, con tres cargas distintas, que sin gang dos trabajos se traban; que backfill elimina el bloqueo de cabeza de cola y hace esperar a los grandes; que consolidar le gana a repartir; que consolidar siempre produce espera por fragmentación; que consolidar sólo si importa iguala el JCT con menos espera; que 2D-LAS le gana a FIFO y a backfill y queda a menos de 5% de SRTF; que un umbral bajo desaloja de más; y que una preempción cara empeora el JCT.

El simulador *Planificador de GPUs* es el mismo modelo en el navegador: la grilla de 256 GPUs se llena en vivo, la cola dice por qué espera el primero (por GPUs o por fragmentación), y se puede cambiar el orden, la ubicación, la carga y el costo de preempción. El botón Comparar corre todos los planificadores con la misma carga.

Ejercicios del README: EASY backfilling, el PROMOTEKNOB de Tiresias, cuotas por equipo con préstamo, fallar temprano con una cola de prueba, migrar para desfragmentar y compartir una GPU entre trabajos chicos.

**Preguntas de repaso.**

1. ¿Qué condición de Coffman rompe el gang scheduling? ¿Y el timeout de Kueue? *Gang rompe retener y esperar: todas las GPUs juntas o ninguna. Kueue deja retener, y si el trabajo no completa a tiempo lo desaloja: rompe "sin desalojo", como la recuperación de un deadlock.*
2. ¿Por qué con backfill codicioso un trabajo grande puede no arrancar nunca? *Cada vez que se libera lugar lo toma un chico que entra; el grande necesita muchas GPUs libres a la vez y nunca las junta. Se arregla reservando (EASY, que necesita duraciones) o con prioridades como LAS.*
3. En Philly, ¿por qué esperan sobre todo los trabajos grandes, si hay GPUs libres? *Por fragmentación: piden localidad y las libres están desparramadas en servidores a medio usar. Para los de más de 8 GPUs, el 97,9% de las esperas.*
4. ¿Por qué LAS se parece a SRTF cuando las duraciones tienen cola pesada? *Porque haber corrido mucho predice que falta mucho: atender primero al que menos recibió es atender primero, en promedio, al que menos le falta.*
5. ¿Qué pasa con 2D-LAS si el umbral entre colas es muy bajo? ¿Y si desalojar cuesta mucho? *Con un umbral bajo, casi todo baja enseguida a la segunda cola, que es FIFO, y se desaloja mucho más. Si desalojar cuesta mucho, cada preempción le suma ese tiempo al trabajo y el JCT sube.*
6. ¿Por qué Gandiva suspende los trabajos en el borde de un mini-batch? *Porque ahí la memoria de la GPU es mínima (las activaciones ya se liberaron): hay muy poco para copiar, y suspender cuesta menos de un segundo.*
7. Un equipo tiene una cuota de 64 GPUs y usa 10. ¿Qué hace Philly con las 54 restantes, y qué pasa cuando el equipo las pide? *Se las presta a otros equipos con trabajos esperando; cuando el dueño las pide, se recuperan con preempción, sólo si el clúster está por encima del 90% de uso.*
8. En Alibaba, la instancia mediana usa 0,042 GPUs. ¿Qué cambia asignar fracciones de GPU? ¿Qué riesgo trae? *En promedio alcanza con la mitad de las GPUs. El riesgo es la interferencia entre instancias que comparten una GPU, que en la traza resultó rara.*

**Lecturas.**

- Jeon et al., [Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](https://www.usenix.org/conference/atc19/presentation/jeon), USENIX ATC 2019. Es el paper de la clase.
- Gu et al., [Tiresias: A GPU Cluster Manager for Distributed Deep Learning](https://www.usenix.org/conference/nsdi19/presentation/gu), NSDI 2019.
- Xiao et al., [Gandiva: Introspective Cluster Scheduling for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/xiao), OSDI 2018.
- Weng et al., [MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters](https://www.usenix.org/conference/nsdi22/presentation/weng), NSDI 2022.
- Choudhury et al., [MAST: Global Scheduling of ML Training across Geo-Distributed Datacenters at Hyperscale](https://www.usenix.org/conference/osdi24/presentation/choudhury), OSDI 2024.
- Kueue, [Setup All-or-nothing with ready Pods](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/).

**Cierre.** En las clases 16 y 17 el trabajo tenía todas las GPUs; hoy vimos por qué conseguirlas no es trivial. Un planificador de GPUs decide tres cosas sin saber cuánto dura nada: **todo o nada** (gang, backfill y el hambre de los grandes), **dónde** (localidad contra espera, con la fragmentación en el medio) y **en qué orden** (2D-LAS, con desalojos que cuestan un checkpoint). Son los problemas clásicos de deadlock, fragmentación y SRTF, con trabajos que piden muchas GPUs a la vez y que corren distinto según dónde queden.
