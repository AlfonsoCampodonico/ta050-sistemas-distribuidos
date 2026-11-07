# Clase 17 — Entrenamiento a escala de frontera

Nov 4, 2026 · @Alfonso Campodonico

La Clase 16 fue Megatron-LM: datos, tensores y pipeline sobre miles de GPUs. Esta clase es cómo se entrena hoy, con dos casos publicados con mucho detalle —Llama 3 de Meta y DeepSeek-V3— y lo que se sabe de entrenar en más de un datacenter. Las tres preguntas de siempre siguen siendo la guía: qué se parte, qué hay que comunicar y por qué enlace, y qué pasa cuando algo falla. La novedad es que la tercera deja de ser un caso borde.

## 1. De Megatron a hoy

Las ideas de la Clase 16 siguen siendo la base; lo que cambia es la escala y lo que esa escala obliga a hacer.

|  | Megatron-LM (2021) | Hoy |
| --- | --- | --- |
| GPUs | 3.072 A100 | 16.384 H100 (Llama 3) |
| Paralelismo | datos, tensores, pipeline | + contexto, + expertos |
| Precisión | 16 bits | FP8 en las multiplicaciones grandes (DeepSeek-V3) |
| Fallas | — | una interrupción inesperada cada tres horas (Llama 3) |
| Dónde | un clúster | varios datacenters (Gemini) |

Todas las filas son de sistemas, incluso la de precisión: es cuántos bytes se mueven por cada número.

## 2. La escala de un entrenamiento de frontera

Llama 3 405B se entrenó con hasta **16.384 H100 a la vez**, en un clúster de 24.000, con 3,8·10²⁵ operaciones de punto flotante en total. Cada GPU sostuvo entre 380 y 430 TFLOPs en BF16, entre 38% y 43% de su pico, y cada paso de entrenamiento procesa 16 millones de tokens (§3.3, tabla 4).

La cuenta de servilleta con esos números:

```latex
\frac{3{,}8 \cdot 10^{25}\ \text{FLOPs}}{16.384 \times 400 \cdot 10^{12}\ \text{FLOPs/s}} \approx 5{,}8 \cdot 10^{6}\ \text{s} \approx 67\ \text{días}
```

No todas las corridas usaron todas las GPUs todo el tiempo, pero el orden es ese: más de nueve semanas de cómputo continuo. Con una falla inesperada cada tres horas, en 67 días son más de 500, y cada una frena a todo el clúster.

El 60% de la capacidad que no se usa se va en comunicación, burbujas del pipeline y movimiento de memoria. Cada técnica de esta clase recupera un pedazo: con 50% de MFU en lugar de 40%, el mismo entrenamiento tardaría unos 54 días en lugar de 67.

## 3. Paralelismo 4D

Llama 3 combina cuatro paralelismos y los ordena de más a menos comunicación: **\[TP, CP, PP, DP\]** (§3.3.2). Cada GPU tiene una coordenada por dimensión y la primera varía más rápido, así que queda entre GPUs vecinas.

| Dimensión | Qué parte | Cuándo se comunica | Dónde queda |
| --- | --- | --- | --- |
| Tensores (TP) | cada matriz | 4 all-reduce por capa y por microbatch | dentro del servidor, NVLink |
| Contexto (CP) | la secuencia | all-gather de K y V en cada capa | dentro del pod |
| Pipeline (PP) | las capas | activaciones a la etapa vecina, por microbatch | dentro del pod |
| Datos (DP, con FSDP) | el batch | pesos y gradientes, una vez por paso | entre pods |

**La red.** El clúster de 405B usa RoCE a 400 Gb/s por GPU. 192 racks de 16 GPUs forman un pod de 3.072 GPUs con ancho de banda completo; ocho pods forman el clúster, y arriba del pod hay una sobresuscripción de 1 a 7 (§3.3.1). Por eso datos va afuera: se sincroniza una vez por paso y tolera la capa más lenta. Como las colectivas generan pocos flujos muy grandes, la librería abre 16 flujos por par de GPUs y E-ECMP los reparte por varios caminos; en el núcleo usan switches con buffers profundos en lugar de control de congestión.

**Las configuraciones** (tabla 4). El producto TP·CP·PP·DP es la cantidad de GPUs, y siempre hay 16 millones de tokens por paso:

| GPUs | TP | CP | PP | DP | Secuencia | TFLOPs/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 8.192 | 8 | 1 | 16 | 64 | 8.192 | 430 |
| 16.384 | 8 | 1 | 16 | 128 | 8.192 | 400 |
| 16.384 | 8 | 16 | 16 | 8 | 131.072 | 380 |

Con secuencias 16 veces más largas entran 16 veces menos secuencias en el paso: datos baja de 128 a 8 y contexto sube de 1 a 16.

**FSDP.** El paralelismo de datos de hoy es ZeRO implementado en PyTorch: las réplicas no guardan el estado completo. Llama 3 parte el estado del optimizador y los gradientes, y los pesos que se juntaron para el forward no se vuelven a partir hasta el backward: más memoria, un all-gather menos. TorchTitan reimplementó FSDP partiendo cada parámetro por separado sobre tensores distribuidos, con unos 7% menos de memoria por GPU (ICLR 2025).

**Contexto.** Con 128K tokens, las activaciones de una secuencia no entran en una GPU. La secuencia se parte en 2·CP pedazos y cada GPU recibe uno del principio y su simétrico del final, porque con la máscara causal los pedazos del final tienen más atención. En lugar de pasar las keys y values en anillo, Llama 3 hace un all-gather antes de cada atención: con grouped-query attention K y V son chicos, y la atención cuesta del orden de S² mientras el all-gather cuesta S.

**Pipeline.** Es el schedule intercalado de la Clase 16, con burbuja (PP−1)/(V·M), más tres ajustes: una cantidad ajustable de microbatches seguidos por etapa (el batch ya no tiene que ser múltiplo de las etapas), una capa menos en la primera y la última etapa para compensar el embedding y la salida, y mensajes asincrónicos con memoria liberada apenas deja de hacer falta. Con eso entrenaron con 8K tokens sin activation checkpointing.

En el repositorio, `go run ./cmd/demo 4d` calcula dónde cae cada grupo con esta red; con el orden al revés (`-orden dp,pp,cp,tp`), un grupo de tensores queda repartido entre pods.

## 4. Expertos: DeepSeek-V3

Un modelo denso usa todos sus parámetros para cada token. Un *mixture of experts* (MoE) reemplaza la capa feed-forward por muchos expertos y un router que elige unos pocos por token. DeepSeek-V3 tiene **671B parámetros en total y 37B activos por token**: 256 expertos ruteados más 1 compartido por capa, y cada token va a 8 ruteados. Se paga la memoria de un modelo grande y el cómputo de uno chico.

**Paralelismo de expertos (EP).** Los expertos se reparten entre GPUs: EP64 sobre 8 nodos. Cada capa MoE hace dos all-to-all: uno para mandar cada token a las GPUs de sus expertos (*dispatch*) y otro para traer los resultados (*combine*). A diferencia del all-reduce de datos, este tráfico depende de qué expertos eligió el router, y va por la red en cada capa.

**Ruteo limitado por nodo.** Dentro del nodo hay NVLink a unos 160 GB/s; entre nodos, InfiniBand a unos 50 GB/s, más de tres veces menos. DeepSeek limita cada token a expertos en **como máximo 4 nodos**: elige primero los nodos con mejor puntaje y después los expertos dentro de ellos. El token cruza InfiniBand una vez por nodo destino y se reparte por NVLink adentro, así que el costo de red queda acotado.

El demo `go run ./cmd/demo moe` lo muestra con 256 expertos en 8 nodos: con ruteo libre un token toca en promedio 5,29 nodos (4,62 remotos); con el límite, 3,95 (3,46 remotos).

**Balance sin pérdida auxiliar.** Si el router manda todo a pocos expertos, esas GPUs son el cuello de botella. En lugar de una pérdida auxiliar que compite con la del modelo, DeepSeek-V3 suma a cada experto un sesgo que solo afecta la elección, y lo sube o baja según si el experto quedó sub- o sobrecargado en el paso anterior.

**DualPipe.** Con EP la relación entre cómputo y comunicación es de alrededor de 1 a 1: si se esperan los all-to-all, la mitad del tiempo se pierde. DualPipe parte cada microbatch en atención, dispatch, MLP y combine, y superpone el cómputo de un microbatch con la comunicación de otro, en forward y backward a la vez. Además alimenta el pipeline desde los dos extremos: la burbuja baja a (PP/2 − 1)·(F&B + B − 3W), contra (PP − 1)·(F + B) de 1F1B, a cambio de guardar dos copias de los parámetros. La comunicación usa solo 20 de los 132 SMs de la H800.

**FP8.** Entrenan las multiplicaciones en FP8 con escalas por bloque: 1×128 para activaciones y 128×128 para pesos, así un valor extremo no arruina la precisión de todo el tensor. Acumulan en FP32 cada 128 elementos. El error relativo de la pérdida queda por debajo de 0,25% contra BF16.

**El costo.** 14,8 billones de tokens en 2.048 H800, 3,7 días por billón de tokens, **2,788 millones de horas GPU** (2,664 millones el preentrenamiento, más extensión de contexto y post-entrenamiento), unos 5,576 millones de dólares a 2 dólares la hora. El paper aclara que eso cubre solo la corrida oficial: no incluye investigación, ablaciones ni corridas fallidas. La lección para la materia no es el número sino de dónde salió: diseñar el modelo (MoE, ruteo por nodo) y el sistema (DualPipe, FP8) juntos, alrededor de la red que tenían.

## 5. Las fallas son el estado normal

Con miles de GPUs, que algo se rompa no es una excepción. En 54 días de preentrenamiento de Llama 3 405B hubo **466 interrupciones**: 47 planificadas y **419 inesperadas**. Cerca del 78% de las inesperadas fueron de hardware, y el 58,7% vino de GPUs (la GPU en sí, su memoria HBM3 o SRAM, su procesador de sistema, corrupción silenciosa de datos). Aun así mantuvieron más del 90% de tiempo útil, con diagnóstico automático y solo tres intervenciones manuales importantes (§3.3.4).

**La cuenta.** Si las fallas son independientes, el tiempo medio entre fallas del clúster es el de una GPU dividido por la cantidad de GPUs:

> MTBF del clúster = MTBF de una GPU / N

Con los datos de Llama 3 (419 fallas en 54 días sobre 16.384 GPUs) sale un MTBF de unas 50.677 horas por GPU, casi seis años. Con 16K GPUs eso es una falla cada 3 horas; con 131K, una cada 23 minutos.

**Cada cuánto guardar.** Cada checkpoint cuesta C segundos; cada falla hace perder lo que se hizo desde el último checkpoint más el reinicio. Guardar seguido gasta tiempo en checkpoints; guardar poco pierde mucho trabajo por falla. El intervalo que minimiza la pérdida es el de Young y Daly:

> τ\* = √(2·C·M)

donde M es el MTBF del clúster. Con C = 30 s y M = 3 h, τ\* ≈ 14 minutos.

**Checkpoints asincrónicos.** Si el checkpoint solo copia el estado a memoria del host y la escritura al almacenamiento sigue en segundo plano, C baja mucho y el intervalo óptimo se acorta. TorchTitan reporta entre 5 y 15 veces menos overhead con checkpoints asincrónicos.

El demo `go run ./cmd/demo fallas` (y `-gpus 131072` para la segunda fila) simula esto con 30 s de checkpoint sincrónico, 3 s de asincrónico y 3 minutos de reinicio, cada uno con su intervalo óptimo:

| GPUs | Intervalo (sinc. / asinc.) | Sincrónico | Asincrónico |
| --- | --- | --- | --- |
| 16.384 | 14 min / 4 min | 91,4% | 96,1% |
| 131.072 | 5 min / 2 min | 71,4% | 82,9% |

A 131K GPUs, incluso con checkpoints asincrónicos se pierde casi una quinta parte del clúster. Por eso el problema de la próxima escala no es solo el ancho de banda: es que un trabajo sincrónico, donde cada paso espera a todas las GPUs, se detiene entero cada vez que una falla. La sección siguiente ataca justamente eso.

**Para infraestructura.** Esto es lo mismo que en Kubernetes: no se evita que los nodos fallen, se diseña para que fallar sea barato. Detectar rápido (health checks de GPU, NCCL que no responde), sacar el nodo, reiniciar desde el último checkpoint con nodos de repuesto ya calientes. Llama 3 también menciona que el consumo del clúster oscila decenas de megawatts cuando todas las GPUs esperan a la vez, un problema para la red eléctrica.

## 6. Sincronizar menos: DiLoCo

Todo lo anterior supone una red muy rápida y un trabajo sincrónico: en cada paso, todas las réplicas de datos hacen un all-reduce y esperan a la más lenta. Eso obliga a tener el clúster en un solo edificio y hace que una falla frene a todos.

**DiLoCo** (Douillard et al., 2023) cambia la regla. Cada réplica da H pasos por su cuenta con su optimizador (AdamW) y recién ahí se sincronizan. Lo que se promedia no son gradientes sino el cambio de cada réplica desde los pesos compartidos, el *pseudo-gradiente*, y se aplica con un optimizador externo con momento de Nesterov. Con H = 500 y 8 réplicas igualan al entrenamiento sincrónico comunicándose **500 veces menos**. Con H = 1, optimizador interno SGD y externo SGD con tasa 1, es paralelismo de datos común. El demo usa justamente SGD adentro y promedio simple afuera, sin momento.

El demo `go run ./cmd/demo diloco` lo reproduce en chico, con 8 réplicas y 400 pasos:

| Sincronizar cada | Rondas | Floats enviados | Loss |
| --- | --- | --- | --- |
| 1 paso (datos común) | 400 | 25.600 | 1,00e-04 |
| 10 pasos | 40 | 2.560 | 1,00e-04 |
| 50 pasos | 8 | 512 | 1,00e-04 |
| 200 pasos | 2 | 128 | 1,00e-04 |

Es el mejor caso posible: el modelo es convexo y las réplicas ven datos parecidos. Con LLMs el resultado del paper es que funciona, pero no gratis: hay que ajustar H y el optimizador externo.

**Decoupled DiLoCo** (Google DeepMind, abril de 2026) lleva la idea a varios datacenters. Cada grupo de réplicas sincroniza por su lado y de forma asincrónica, así que una falla en un sitio no frena a los demás. Reportan:

- para sostener 95% de utilización entre 8 datacenters, el ancho de banda necesario baja de 198 Gbps a 3,3 Gbps (0,84 Gbps comunicando en int4);
- con fallas simuladas sobre 1,2 millones de chips, 88% de tiempo útil contra 58% del paralelismo de datos elástico (27% sin elasticidad);
- calidad comparable: en un modelo denso de 5B entrenado con 1T tokens, el promedio de evaluaciones de texto da 69,8 contra 70,1 del paralelismo de datos sin fallas, y 69,8 contra 69,7 con fallas simuladas (tabla 2).

**Por qué importa.** Si no hace falta sincronizar cada paso, el clúster deja de tener que estar en un solo lugar, y la energía, que hoy es el límite real para construir un datacenter de un gigawatt, se puede buscar en varios sitios. Es el mismo cambio que vimos en la materia con sistemas que pasan de consistencia fuerte a eventual: se acepta que las réplicas diverjan un rato a cambio de comunicarse menos y tolerar fallas.

## 7. ¿Y los laboratorios?

Lo que sabemos con detalle viene de los laboratorios que publicaron sus sistemas. De los demás hay pistas, y hay que presentarlas como pistas.

| Laboratorio | Qué publicó | Qué se sabe |
| --- | --- | --- |
| Meta | el paper de Llama 3 | 16K H100, paralelismo 4D, RoCE, fallas |
| DeepSeek | el informe de V3 | MoE, FP8, DualPipe, 2.048 H800, costo |
| Google | Gemini, DiLoCo | TPUs, varios datacenters, sincronizar menos |
| OpenAI | casi nada | GPT-4.5 en varios clústeres |
| Anthropic | acuerdos de cómputo | Trainium2 (Project Rainier) y TPUs de Google |

- **Google.** El informe técnico de Gemini (2023) dice que Gemini Ultra se entrenó con TPUs en varios datacenters, con paralelismo de datos sincrónico entre superpods. La línea DiLoCo, hasta Decoupled DiLoCo en 2026, es su trabajo publicado sobre sincronizar menos.
- **OpenAI.** Publica muy poco de su infraestructura de entrenamiento; dijo que GPT-4.5 se entrenó en varios clústeres.
- **Anthropic.** Tampoco publicó cómo entrena. Lo público son sus acuerdos de cómputo: Project Rainier de AWS, con cerca de medio millón de chips Trainium2, y un acuerdo con Google por hasta un millón de TPUs, anunciado en octubre de 2025.

**¿Es igual en todos lados?** Que no esté publicado no quiere decir que hagan algo distinto. Las restricciones físicas son las mismas: una red jerárquica donde adentro es rápido y afuera es lento, una tasa de fallas del hardware que crece con la escala, y la energía. Es razonable esperar soluciones parecidas (partir en varias dimensiones ordenadas por la red, esconder la comunicación, guardar estado seguido, sincronizar menos entre sitios), pero no lo sabemos.

**Pregunta para la clase.** ¿Qué cambia si el hardware no es una GPU de NVIDIA sino una TPU o un Trainium? Cambian los enlaces y sus velocidades (las TPUs, por ejemplo, se conectan en un toro dentro del pod), y con eso cuánto conviene cada dimensión. Las tres preguntas de la Clase 16 siguen siendo las mismas.

## 8. Demo, preguntas y lecturas

**El repositorio.** `entrenamiento-distribuido` suma cuatro demos a las de la Clase 16 (allreduce, train y pipeline):

| Comando | Qué muestra |
| --- | --- |
| `go run ./cmd/demo 4d` | dónde cae cada dimensión en la red de Llama 3 |
| `go run ./cmd/demo moe` | cuántos nodos toca un token, con y sin límite |
| `go run ./cmd/demo fallas` | fallas, intervalo de checkpoint y tiempo útil |
| `go run ./cmd/demo diloco` | sincronizar cada H pasos |

`go test ./...` verifica lo que dice la clase: el orden de Llama 3 deja tensores dentro del servidor, limitar nodos baja el tráfico, el intervalo de Young y Daly le gana a los demás y sincronizar cada 10 pasos converge igual. El simulador *Entrenamiento a escala* permite jugar con las fallas y el 4D en el navegador.

Ejercicios nuevos del README: repuestos en caliente y checkpoints en memoria para el simulador de fallas, y el balanceo de expertos con el sesgo de DeepSeek-V3 para el de MoE.

**Preguntas de repaso.**

1. ¿Por qué el orden es \[TP, CP, PP, DP\] y no al revés? *Tensores se comunica en cada capa y microbatch, en el camino crítico: va por NVLink. Datos sincroniza una vez por paso y tolera la capa sobresuscripta entre pods.*
2. Con secuencias de 128K, ¿qué dimensión crece y cuál baja? *Contexto sube de 1 a 16 y datos baja de 128 a 8: en los 16M tokens del paso entran 16 veces menos secuencias, con las mismas GPUs.*
3. ¿Por qué con expertos la comunicación es all-to-all? *Cada token va a expertos en GPUs distintas, así que cada par de GPUs intercambia datos distintos. Limitar a 4 nodos baja los envíos remotos de 4,6 a 3,5 por token, un 25%.*
4. Con 65.536 GPUs y checkpoints de 30 s, ¿cada cuánto guardar? *M ≈ 50.700 h / 65.536 ≈ 0,77 h; τ\* = √(2 · 30 s · 0,77 h) ≈ 7 minutos.*
5. ¿Por qué DiLoCo aguanta mejor las fallas? *Una falla solo detiene a su grupo, que se reincorpora cuando vuelve; con paralelismo de datos sincrónico detiene a todos.*

**Lecturas.**

- Grattafiori et al., [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), 2024, §3.3.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024, §3.
- Douillard et al., [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105), 2023.
- Liang et al., [TorchTitan](https://arxiv.org/abs/2410.06511), ICLR 2025.
- Google DeepMind, Decoupled DiLoCo, abril de 2026.
- Gemini Team, [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805), 2023.

**Próxima clase (18).** Pasamos de entrenar a servir: cómo se sirve un modelo, con el paper de vLLM.
