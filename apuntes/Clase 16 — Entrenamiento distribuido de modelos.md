# Clase 16 — Entrenamiento distribuido de modelos

Dec 2, 2026 · @Alfonso Campodonico

Entrenar un modelo de lenguaje grande es, antes que un problema de machine learning, un problema de sistemas distribuidos: el modelo no entra en una GPU y el cómputo no termina en una vida humana. Esta clase muestra cómo se parte ese trabajo en tres dimensiones —datos, tensores y pipeline— y qué cuesta cada una en comunicación, memoria y tiempo ocioso.

**Paper:** D. Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), SC 2021.

**Lecturas complementarias:** S. Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), SC 2020; P. Patarasuk y X. Yuan, [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf), JPDC 2009.

**Repositorio:** `entrenamiento-distribuido` (Go, sin dependencias), con tres demos que se corren en clase.

Casi nada de lo que vamos a ver es nuevo para la materia. El paralelismo de datos es sharding del input, como en MapReduce. ZeRO es sharding del estado, como en la Clase 3. El all-reduce es una barrera, y una barrera arrastra el problema de los stragglers que ya vimos con las backup tasks. Lo nuevo es la escala y que el estado es enorme, mutable y está acoplado: cada paso depende de que todos hayan terminado el anterior.

## 1. Por qué entrenar es un problema distribuido

Hay dos paredes, y cada una obliga a distribuir por una razón distinta: la **memoria** y el **cómputo**.

Entrenar es repetir un ciclo muy simple millones de veces. Se toma un batch de ejemplos, se hace el **forward** (pasar el input por todas las capas hasta obtener una predicción y un error), después el **backward** (recorrer las capas al revés calculando el gradiente de cada parámetro), y al final el **paso del optimizador**, que mueve cada parámetro un poquito en contra de su gradiente. Para nosotros el modelo es una función con miles de millones de números adentro, y el entrenamiento es un loop que los actualiza.

La pared del cómputo se ve con un número del paper: entrenar GPT-3, con 175 mil millones de parámetros, llevaría unos **288 años** en una sola GPU V100. No hay forma de esperar eso, así que hay que repartir el cómputo entre miles de GPUs.

La pared de la memoria es más sutil. Uno pensaría que un modelo de 175 mil millones de parámetros en 16 bits ocupa 350 GB. Pero entrenar no es sólo guardar los pesos: también hay que guardar los gradientes y el estado del optimizador, y eso multiplica la cuenta por ocho (lo vemos en la sección 4). La GPU más grande del paper tiene 80 GB. El modelo, simplemente, no entra.

> El paper resume las dos paredes en la primera página: la capacidad de memoria de una GPU es limitada, así que un modelo grande no entra ni siquiera en un servidor con varias GPUs; y la cantidad de operaciones lleva a tiempos de entrenamiento irreales. Las dos razones piden cosas distintas: la del cómputo se resuelve con más copias trabajando en paralelo, la de la memoria con partir el modelo.

## 2. Paralelismo de datos

La idea más simple es la de MapReduce: si el input es enorme, se parte. Cada worker tiene una **copia completa del modelo** y procesa una porción distinta del batch. Hace su forward y su backward localmente y obtiene un gradiente que sólo refleja *sus* ejemplos.

El problema aparece en el paso del optimizador. Si cada worker aplicara su propio gradiente, las copias del modelo divergirían en el primer paso y tendríamos d modelos distintos en lugar de uno. Hace falta que todos apliquen **exactamente la misma actualización**, y para eso tienen que ponerse de acuerdo en el gradiente: sumar los gradientes de todos y dividir por d. Esa operación —todos aportan un vector y todos reciben la suma— se llama **all-reduce**.

Con el all-reduce, el paralelismo de datos tiene una propiedad muy fuerte: con el mismo batch global, d workers producen **el mismo modelo** que uno solo. El paper la llama *strict optimizer semantics*. Es la misma clase de garantía que daba MapReduce con mappers deterministas: la ejecución distribuida es indistinguible de una secuencial. En la demo se ve directamente: la diferencia entre los pesos entrenados con 1 y con 4 workers es del orden de 10⁻¹⁶, puro redondeo porque la suma se hace en otro orden.

Esa garantía tiene un precio: el all-reduce es una **barrera**. Ningún worker puede dar el paso siguiente hasta que llegaron los gradientes de todos, así que cada paso dura lo que tarda el worker más lento. Con miles de GPUs, siempre hay alguna más lenta: un straggler. En MapReduce lo resolvíamos con backup tasks, pero acá no se puede lanzar una copia del worker lento sin duplicar también su porción de memoria y su estado.

> El paper marca dos límites del paralelismo de datos (§1). Primero, a partir de cierto punto el batch por GPU se vuelve tan chico que la GPU se usa mal y la comunicación domina. Segundo, la cantidad máxima de GPUs es el tamaño del batch: GPT-3 se entrenó con un batch de 1536, así que el paralelismo de datos sólo admite 1536 GPUs, cuando para terminar en un tiempo razonable hacían falta unas 10.000. Y hay un tercer límite que no se resuelve agregando réplicas: **cada réplica tiene que entrar en una GPU**. El paralelismo de datos ataca la pared del cómputo, pero no la de la memoria.

## 3. Ring all-reduce

El all-reduce ingenuo es un **parameter server**: un coordinador recibe los gradientes de todos, los suma y le devuelve el resultado a cada uno. Funciona, pero el coordinador recibe y envía (n−1)·S datos, donde S es el tamaño del gradiente. Es el mismo cuello de botella que tenía el master de MapReduce, ahora en la red: su enlace transporta todo, y el costo crece linealmente con la cantidad de GPUs.

El **ring all-reduce** elimina al coordinador. Las n GPUs forman un anillo, cada una sólo le envía a su vecina de la derecha, y el gradiente se parte en n pedazos. La operación tiene dos fases de n−1 pasos cada una:

1. **Reduce-scatter.** En cada paso, cada GPU le pasa un pedazo a su vecina, que lo suma al suyo. Después de n−1 pasos, cada GPU tiene *un* pedazo con la suma de todas.
2. **All-gather.** Cada GPU hace circular su pedazo terminado. Después de otros n−1 pasos, todas tienen todos los pedazos.

&#91;embedded content: ring all-reduce con 4 GPUs · el gradiente partido en 4 pedazos (A–D)\]

Después del reduce-scatter, cada GPU tiene un solo pedazo con la suma completa (la GPU r, el pedazo r+1); el all-gather los reparte a todas.

En cada paso, cada GPU envía S/n datos, y hay 2(n−1) pasos. Lo que envía cada GPU en total es:

```latex
\text{enviado por GPU} = 2 \cdot \frac{n-1}{n} \cdot S
```

Eso tiende a 2·S **sin importar cuántas GPUs haya**, y todos los enlaces trabajan a la vez. Patarasuk y Yuan demostraron en 2009 que es el mínimo posible: no existe un all-reduce en el que cada proceso envíe menos. La demo lo mide contando floats por canal:

| GPUs | Central: envía el rank 0 | Ring: envía cada rank |
| --- | --- | --- |
| 2 | 1,00·S | 1,00·S |
| 4 | 3,00·S | 1,50·S |
| 8 | 7,00·S | 1,75·S |
| 16 | 15,00·S | 1,88·S |
| 32 | 31,00·S | 1,94·S |

El anillo no es gratis: optimiza ancho de banda, no latencia. Tiene 2(n−1) pasos secuenciales, y cada uno paga la latencia de un mensaje. Para gradientes chicos o anillos muy largos conviene un algoritmo en árbol, con 2·log n pasos; NCCL elige entre los dos según el tamaño. Para los gradientes de un modelo grande, que son de gigabytes, el ancho de banda manda y el anillo gana.

> El paper usa esta propiedad para analizar el paralelismo de datos (§3.3.1): como el costo del ring all-reduce escala con (d−1)/d = 1 − 1/d, agregar réplicas casi no aumenta la comunicación por GPU. Por eso el paralelismo de datos escala tan bien mientras el modelo entre en memoria.

## 4. Dónde se va la memoria: ZeRO

Cada parámetro cuesta **16 bytes** de memoria durante el entrenamiento, no 2. La cuenta que hace el paper de ZeRO para entrenamiento con precisión mixta y el optimizador Adam es esta:

| Qué se guarda | Precisión | Bytes por parámetro |
| --- | --- | --- |
| Pesos (los que usa el forward) | 16 bits | 2 |
| Gradientes | 16 bits | 2 |
| Copia maestra de los pesos | 32 bits | 4 |
| Momento de Adam | 32 bits | 4 |
| Varianza de Adam | 32 bits | 4 |
| **Total** |  | **16** |

Con esa cuenta, un modelo de un billón de parámetros necesita unos 16 TB sólo para su estado, sin contar las activaciones del forward. Y el paralelismo de datos no ayuda: cada réplica guarda los 16 bytes por parámetro completos. El paper de ZeRO reporta que el paralelismo de datos solo se queda sin memoria con modelos de más de 1.400 millones de parámetros en GPUs de 32 GB.

La observación de ZeRO es que esa memoria es **redundante**: las d réplicas guardan exactamente lo mismo. Es el mismo diagnóstico que llevó al sharding en la Clase 3. Si hay d copias idénticas de un estado enorme, se puede partir en d pedazos y que cada worker sea dueño de uno. ZeRO lo hace en tres etapas, cada una partiendo algo más:

| Etapa | Qué se parte entre las d réplicas | Memoria por GPU (7.500 millones de parámetros, d = 64) |
| --- | --- | --- |
| Paralelismo de datos | Nada | 120 GB |
| ZeRO-1 | Estado del optimizador | 31,4 GB |
| ZeRO-2 | + gradientes | 16,6 GB |
| ZeRO-3 | + pesos | 1,9 GB |

Con ZeRO-3 la memoria por GPU cae linealmente con d. A cambio, ningún worker tiene los pesos completos: antes de calcular cada capa tiene que pedirles su pedazo a los demás y descartarlo al terminar. Las etapas 1 y 2 no agregan comunicación respecto del paralelismo de datos; la etapa 3, sí.

> Narayanan et al. comparan su enfoque con ZeRO-3 (§5.2) y encuentran que, al duplicar la cantidad de GPUs con el mismo batch, el suyo rinde un 70% más para modelos de 175 y 530 mil millones de parámetros. La razón es la comunicación entre nodos: ZeRO-3 trae los pesos por la red en cada capa. Las dos ideas no compiten, se combinan: PyTorch implementa ZeRO-3 como FSDP, y se usa junto con las técnicas que siguen.

## 5. Paralelismo de tensores

Si el modelo no entra en una GPU, la alternativa a ZeRO es partir **el cómputo de cada capa**. Un transformer es casi entero multiplicaciones de matrices, y una multiplicación de matrices se puede repartir entre GPUs partiendo una de las matrices. Es la técnica de Megatron (Shoeybi et al., 2019), y la sección 2.3 del paper la resume con el bloque MLP de cada capa:

```latex
Y = \mathrm{GeLU}(XA) \qquad Z = \mathrm{Dropout}(YB)
```

El truco está en *cómo* se parte cada matriz. A se parte por columnas, A = \[A₁, A₂\]: cada GPU calcula su mitad de XA y le aplica GeLU por su cuenta, porque GeLU actúa elemento por elemento y no necesita la otra mitad. B se parte por filas, y cada GPU multiplica su mitad de Y por su mitad de B. Recién al final hay que sumar los dos resultados parciales: un all-reduce. Con la atención pasa lo mismo, partiendo por cabezas. El resultado son **dos all-reduce en el forward y dos en el backward por cada capa**.

Eso es mucha comunicación, y además está en el camino crítico: ninguna GPU puede seguir hasta que termina el all-reduce de la capa. El paper calcula que cada GPU comunica 8·b·s·h·(t−1)/t datos por capa y por microbatch, con b el tamaño del microbatch, s la longitud de secuencia, h la dimensión oculta y t el grado de paralelismo de tensores.

Por eso el paralelismo de tensores sólo funciona sobre enlaces muy rápidos. Dentro de un servidor DGX A100, las 8 GPUs están conectadas por **NVLink y NVSwitch**; entre servidores, cada nodo tiene 8 placas InfiniBand de 200 Gb/s, que son bastante más lentas. Un transformer con t = 8 dentro de un nodo funciona bien; con t = 16, cada uno de esos all-reduce tiene que cruzar la red entre nodos y el rendimiento se cae.

> El paper agrega un segundo problema (§3.3.2): con t grande, cada GPU multiplica matrices cada vez más chicas, y las GPUs modernas no alcanzan su pico con matrices chicas. Partir demasiado desperdicia el hardware aunque la red fuera infinita. De ahí sale el *Takeaway #1*: usar paralelismo de tensores hasta la cantidad de GPUs de un servidor, y escalar más allá con otra cosa.

## 6. Paralelismo de pipeline

La otra forma de partir el modelo es por **capas**: la GPU 0 tiene las primeras capas, la GPU 1 las siguientes, y así. Entre etapas sólo viaja la activación de una capa a la siguiente, por comunicación punto a punto: b·s·h datos por microbatch, mucho menos que el all-reduce del paralelismo de tensores. Por eso el pipeline puede cruzar la red entre servidores.

El problema es obvio en cuanto se dibuja. Con un batch entero, la GPU 1 no puede empezar hasta que la GPU 0 terminó, y mientras una trabaja las demás esperan. La solución es la de cualquier línea de montaje: partir el batch en **m microbatches** y hacerlos fluir uno detrás de otro. Aun así, al principio de cada batch el pipeline se está llenando y al final se está vaciando. Ese tiempo ocioso se llama **burbuja**.

Esta es la salida de la demo con p = 4 etapas y m = 8 microbatches. Los números son forwards, las letras son backwards (que tardan el doble) y los puntos son tiempo en que la GPU no hace nada:

```text
GPipe — p=4, m=8, tf=1, tb=2
GPU 0 │12345678.........aabbccddeeffgghh│
GPU 1 │.12345678......aabbccddeeffgghh..│
GPU 2 │..12345678...aabbccddeeffgghh....│
GPU 3 │...12345678aabbccddeeffgghh......│
tiempo=33  ideal=24  burbuja=37.5%  (fórmula (p-1)/m = 37.5%)
activaciones en memoria (pico por GPU): [8 8 8 8]

1F1B (PipeDream-Flush) — p=4, m=8, tf=1, tb=2
GPU 0 │1234......aa5bb6cc7dd8ee.ff.gg.hh│
GPU 1 │.123....aa4bb5cc6dd7ee8ff.gg.hh..│
GPU 2 │..12..aa3bb4cc5dd6ee7ff8gg.hh....│
GPU 3 │...1aa2bb3cc4dd5ee6ff7gg8hh......│
tiempo=33  ideal=24  burbuja=37.5%  (fórmula (p-1)/m = 37.5%)
activaciones en memoria (pico por GPU): [4 3 2 1]
```

El paper calcula la burbuja (§2.2.1). Llenar el pipeline cuesta p−1 forwards y vaciarlo p−1 backwards, mientras que el trabajo útil son m forwards y m backwards. El tiempo perdido, medido respecto del tiempo ideal sin burbuja, es:

```latex
\text{burbuja} = \frac{p-1}{m}
```

Con p = 4 y m = 8 da 3/8 = 37,5%, exactamente lo que mide el simulador: (33 − 24) / 24. Como fracción del tiempo total es (p−1)/(m+p−1): en el diagrama, 9 de 33 casilleros por GPU. Para que la burbuja sea chica hace falta **m ≫ p**: muchos microbatches por cada etapa.

Los dos schedules tienen la misma burbuja, pero no la misma memoria. **GPipe** hace todos los forwards y después todos los backwards, así que cada GPU guarda las activaciones de los m microbatches hasta el backward. **1F1B** (PipeDream-Flush) arranca con unos pocos forwards y después alterna uno y uno, así que una etapa nunca tiene más de p microbatches en vuelo. Con m ≫ p, que es justo lo que queremos para achicar la burbuja, 1F1B usa mucha menos memoria.

La burbuja existe por una decisión de diseño: al final de cada batch hay un **flush**, se espera a que todos los microbatches terminen y recién ahí se aplica el optimizador. Es lo que conserva la semántica estricta, porque cada microbatch ve en el backward los mismos pesos que vio en el forward.

> El paper propone un schedule **intercalado** (§2.2.2): en lugar de un bloque contiguo de capas, cada GPU tiene v pedazos chicos (la GPU 1 tiene las capas 1, 2, 9 y 10; la GPU 2, las 3, 4, 11 y 12). Cada pedazo tarda v veces menos, y la burbuja baja a (p−1)/(v·m). El costo es que la comunicación punto a punto también se multiplica por v. Con la optimización de scatter/gather de §4.1, que aprovecha las 8 placas InfiniBand de cada nodo, el schedule intercalado mejora el rendimiento en un 10% o más.

## 7. PTD-P: componer las tres dimensiones

Ninguna de las tres técnicas alcanza sola, y la contribución del paper es mostrar cómo combinarlas. Las n GPUs se organizan como una grilla de **p × t × d = n**: cada réplica del modelo está partida en p etapas de pipeline, cada etapa está partida en t GPUs con paralelismo de tensores, y hay d réplicas con paralelismo de datos.

| Dimensión | Qué se parte | Comunicación | Cuándo | Dónde vive |
| --- | --- | --- | --- | --- |
| Tensores (t) | Las matrices de cada capa | 4 all-reduce por capa | En cada microbatch | Dentro del nodo, por NVLink |
| Pipeline (p) | Las capas | Punto a punto, b·s·h | En cada microbatch | Entre nodos, por InfiniBand |
| Datos (d) | El batch | 1 all-reduce de los gradientes | Una vez por batch | Entre réplicas |

La tabla se lee como una jerarquía de costos. Lo que más comunica va por el enlace más rápido, y lo que menos comunica va por el más lento. La regla sale del hardware, no del modelo.

&#91;embedded content: PTD-P sobre el clúster · tensores dentro del nodo, pipeline entre nodos, datos entre réplicas\]

Cada fila es una réplica completa del modelo; cada columna, una etapa del pipeline. Las líneas punteadas unen las GPUs que tienen las mismas capas en distintas réplicas: son las que hacen el all-reduce de los gradientes.

El paper la resume en tres reglas prácticas (§3):

1. **Takeaway #1.** Usar paralelismo de tensores hasta la cantidad de GPUs de un servidor (8 en un DGX A100), y escalar entre servidores con pipeline.
2. **Takeaway #2.** Elegir el tamaño de modelo paralelo t·p mínimo para que el modelo entre en memoria, y usar paralelismo de datos para todo lo demás. El paralelismo de datos es el que menos comunica por cómputo realizado.
3. **Takeaway #3.** El tamaño de microbatch óptimo depende del modelo: uno más grande usa mejor la GPU, pero deja menos microbatches por batch y agranda la burbuja.

Con esas reglas, el paper entrena un modelo de **un billón de parámetros en 3.072 GPUs A100** con un rendimiento de 502 petaFLOP/s en total. Son 163 teraFLOP/s por GPU, el 52% del pico teórico de 312, y eso incluye comunicación, carga de datos y el optimizador. Estiman que el entrenamiento completo, sobre 450 mil millones de tokens, llevaría unos 84 días.

> La cifra que más dice de sistemas distribuidos está en §5.9: con el modelo de un billón de parámetros, la comunicación del pipeline usó un ancho de banda de bisección efectivo de 892 GB/s, y la del paralelismo de datos, 12,9 TB/s. El paper es explícito en que su resultado viene de optimizar software y hardware juntos: con una red más lenta entre nodos, o con una partición que comunique más, el escalado se cae. Combinaciones subóptimas de tensores y pipeline llegan a rendir la mitad, aun con enlaces rápidos.

## 8. Fallas y checkpoints

En MapReduce una falla era barata: se reintentaba la tarea que falló y el resto seguía. En un entrenamiento sincrónico, **una falla detiene a todos**. Cada paso necesita los gradientes de todas las réplicas, cada etapa del pipeline necesita a la anterior y cada capa partida necesita a sus t GPUs. Si una sola de las 3.072 GPUs se cae, ningún paso puede terminar.

La recuperación es la mitigación que el paper de MapReduce describía para el coordinador y decidía no implementar: **checkpoints periódicos**. Cada tanto se guarda el estado completo, y ante una falla todos vuelven al último checkpoint y rehacen los pasos perdidos. Con miles de GPUs, las fallas dejan de ser improbables y pasan a ser parte de la operación normal, así que acá el checkpoint no es opcional.

Para que volver atrás sea correcto hace falta la misma condición que en MapReduce: **determinismo**. Si el batch de cada paso depende sólo del número de paso, rehacer los pasos 100 a 120 da exactamente los mismos pesos que la primera vez. En la demo, un entrenamiento con una falla en el paso 120 y checkpoints cada 50 pasos termina con los mismos pesos, bit a bit, que uno sin fallas; sólo ejecutó 21 pasos de más.

Elegir cada cuánto guardar es un compromiso. Guardar seguido cuesta tiempo en cada checkpoint; guardar poco cuesta muchos pasos rehechos en cada falla. Con un costo de checkpoint C y un tiempo medio entre fallas F, la aproximación clásica de Young da el intervalo óptimo:

```latex
T_{\text{óptimo}} \approx \sqrt{2 \cdot C \cdot F}
```

> El paper da los números del problema (§5.10): el checkpoint del modelo de un billón de parámetros ocupa **13,8 TB**. La carga inicial desde los 384 nodos alcanza 1 TB/s, el máximo del sistema de archivos paralelo, y el guardado llega a 273 GB/s, el 40% del pico de escritura. A esa velocidad, cada checkpoint tarda cerca de un minuto con las 3.072 GPUs esperando. El storage vuelve a aparecer como parte del sistema de cómputo, igual que GFS en MapReduce.

La misma lógica de "todos o ninguno" aparece al **planificar** el trabajo. Un orquestador como Kubernetes, por defecto, ubica los pods de a uno. Si un entrenamiento necesita 512 pods y sólo entran 500, esos 500 quedan ocupando GPUs sin poder avanzar ni un paso. Por eso los clústeres de entrenamiento usan *gang scheduling*: se le asignan recursos al trabajo entero o no se le asigna nada.

## 9. Demo, preguntas y referencias

El repositorio simula todo lo anterior en Go: cada GPU es una goroutine, cada enlace un canal, y se cuentan los datos que pasan por cada uno. Las tres demos que se corren en clase son:

```text
go run ./cmd/demo allreduce                 # central vs anillo
go run ./cmd/demo train -straggler 2ms      # la barrera espera al más lento
go run ./cmd/demo train -fail-at 120        # una falla vuelve a todos al checkpoint
go run ./cmd/demo pipeline -p 4 -m 8        # la burbuja de GPipe y 1F1B
```

**Preguntas de repaso**

1. ¿Por qué el paralelismo de datos no permite entrenar un modelo que no entra en una GPU, por más réplicas que se agreguen?
2. Con 8 GPUs, ¿cuánto envía el rank más cargado en un all-reduce central y cuánto en uno en anillo? ¿Qué pasa con cada número al pasar a 64 GPUs?
3. ¿En qué se parece ZeRO al sharding de la Clase 3? ¿Qué comunicación agrega la etapa 3 que no tenían las etapas 1 y 2?
4. ¿Por qué el paralelismo de tensores va dentro del nodo y el de pipeline entre nodos? Justificalo con la cantidad de comunicación de cada uno.
5. Con p = 8 etapas, ¿cuántos microbatches hacen falta para que la burbuja sea menor al 10%? ¿Y con el schedule intercalado y v = 2?
6. GPipe y 1F1B tienen la misma burbuja. ¿Por qué se prefiere 1F1B?
7. En MapReduce, un worker lento se resolvía con backup tasks. ¿Por qué esa técnica no se aplica directamente al entrenamiento sincrónico?
8. ¿Qué condición tiene que cumplir el entrenamiento para que volver a un checkpoint dé el mismo resultado? ¿Con qué condición de MapReduce se corresponde?

**Referencias**

- D. Narayanan et al. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). SC 2021.
- S. Rajbhandari et al. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). SC 2020.
- P. Patarasuk y X. Yuan. [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf). JPDC 69(2), 2009.
- M. Shoeybi et al. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). 2019.
- J. Dean y S. Ghemawat. [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/es//archive/mapreduce-osdi04.pdf). OSDI 2004 (Clase 2).
