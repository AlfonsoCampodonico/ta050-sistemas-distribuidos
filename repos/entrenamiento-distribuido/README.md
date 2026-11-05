# Entrenamiento distribuido — demo de clase

Material práctico de las clases **16 — Entrenamiento distribuido de modelos**
y **17 — Entrenamiento a escala de frontera** de TA050 Sistemas Distribuidos I
(FIUBA). Es una simulación en Go, sin GPUs ni
red real: cada "GPU" es una goroutine y cada enlace entre GPUs es un canal.
Lo que se simula es lo que importa para la materia: **cuántos datos se mueven,
quién espera a quién y qué pasa cuando algo falla**.

Papers de las clases: Narayanan et al., *Efficient Large-Scale Language Model
Training on GPU Clusters Using Megatron-LM*, SC 2021
([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)) para la 16, y para la 17
la sección de infraestructura de *The Llama 3 Herd of Models* (2024,
[arXiv:2407.21783](https://arxiv.org/abs/2407.21783), §3.3) junto con el
*DeepSeek-V3 Technical Report* (2024, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), §3).

## Requisitos

Go 1.24 o superior. No hay dependencias externas.

```bash
go test ./...
```

## Estructura

| Paquete | Qué modela | Concepto de la clase |
|---|---|---|
| `collective` | All-reduce central (tipo parameter server) y ring all-reduce | Por qué el anillo envía 2(n-1)/n·S por GPU y el central (n-1)·S |
| `train` | Regresión lineal entrenada con paralelismo de datos | Réplicas idénticas, barrera por paso, stragglers, checkpoints |
| `pipeline` | Simulador de schedules GPipe y 1F1B | La burbuja (p-1)/m y la memoria de activaciones |
| `layout` | Dónde cae cada dimensión de paralelismo 4D en la red | El orden [TP, CP, PP, DP] de Llama 3 |
| `moe` | Ruteo de tokens a expertos repartidos en nodos | Paralelismo de expertos y node-limited routing (DeepSeek-V3) |
| `fallas` | Un entrenamiento largo con GPUs que se rompen | Fallas a escala, intervalo de checkpoint de Young/Daly, checkpoints asincrónicos |
| `localsgd` | Réplicas que se sincronizan cada H pasos | DiLoCo: entrenar con mucha menos comunicación |
| `cmd/demo` | La herramienta que se corre en clase | — |

## Demo 1 — all-reduce: central vs anillo

```bash
go run ./cmd/demo allreduce
```

Cada GPU tiene un gradiente de S floats y todas necesitan la suma. Con el
enfoque central, el rank 0 recibe y reenvía todo: su enlace transporta
(n-1)·S y se vuelve el cuello de botella. Con el anillo, cada GPU sólo le
habla a su vecina y envía 2(n-1)/n·S, que tiende a 2·S **sin importar cuántas
GPUs haya**. Ese es el límite inferior para all-reduce (Patarasuk y Yuan, 2009).

Mirá `collective/allreduce.go`: la función `Ring` tiene las dos fases
(reduce-scatter y all-gather) en una docena de líneas.

## Demo 2 — paralelismo de datos, stragglers y fallas

```bash
go run ./cmd/demo train                       # 1 worker vs 4 workers: mismo modelo
go run ./cmd/demo train -straggler 2ms        # un worker lento frena a todos
go run ./cmd/demo train -fail-at 120          # una falla hace volver a todos al checkpoint
go run ./cmd/demo train -fail-at 120 -checkpoint-every 10
```

Tres cosas para observar:

1. **Semántica estricta.** Con el mismo batch global, 4 workers producen el
   mismo modelo que 1 (hasta el redondeo). El all-reduce es lo que lo garantiza:
   todas las réplicas aplican exactamente la misma actualización. El código
   lo verifica en cada paso y falla si alguna réplica diverge.
2. **El straggler.** El all-reduce es una barrera: el paso termina cuando
   termina el worker más lento. Compará con la ejecución especulativa de
   MapReduce (Clase 2): ¿por qué acá no se puede lanzar una "backup task"?
3. **La falla.** En MapReduce una falla se resolvía reintentando una tarea.
   Acá el estado está repartido y acoplado: si un worker muere, el paso no
   termina para nadie, y todos vuelven al último checkpoint. Funciona porque
   cada paso es **determinista** — el batch depende sólo del número de paso —,
   la misma condición que pedía MapReduce.

## Demo 3 — la burbuja del pipeline

```bash
go run ./cmd/demo pipeline -p 4 -m 8
go run ./cmd/demo pipeline -p 4 -m 32
go run ./cmd/demo pipeline -p 8 -m 8
```

El modelo se parte en p etapas y el batch en m microbatches. Los números son
forwards, las letras backwards y los puntos son tiempo en que la GPU no hace
nada. El simulador mide la burbuja y coincide con la fórmula del paper,
(p-1)/m, para los dos schedules. La diferencia está en la última línea: con
GPipe cada GPU guarda las activaciones de los m microbatches; con 1F1B, a lo
sumo p.

## Clase 17 — Demo 4: dónde cae cada dimensión

```bash
go run ./cmd/demo 4d                              # Llama 3 405B, 16.384 GPUs
go run ./cmd/demo 4d -cp 16 -dp 8                 # la configuración de 128K tokens
go run ./cmd/demo 4d -orden dp,pp,cp,tp           # el orden al revés
```

La GPU de rank r tiene una coordenada por dimensión, y la primera dimensión
del orden varía más rápido. Con [TP, CP, PP, DP], los 8 ranks de un grupo de
tensores son 8 GPUs del mismo servidor (NVLink); los de pipeline caen en 16
servidores del mismo pod; los de datos cruzan pods. Es el orden de Llama 3
(§3.3.2): lo que más se comunica, adentro.

## Demo 5 — expertos y tráfico entre nodos

```bash
go run ./cmd/demo moe
go run ./cmd/demo moe -nodos 16
```

DeepSeek-V3 tiene 256 expertos por capa y cada token usa 8. Con los expertos
repartidos en 8 nodos y ruteo libre, un token visita en promedio 5,3 nodos;
cada nodo remoto es un envío por InfiniBand. Limitando a 4 nodos por token,
como hace DeepSeek-V3 (§2.1.2), baja a 3,95.

## Demo 6 — fallas y checkpoints a escala

```bash
go run ./cmd/demo fallas
go run ./cmd/demo fallas -gpus 131072
go run ./cmd/demo fallas -checkpoint 300 -restart 600
```

Con la tasa de fallas de Llama 3 —419 interrupciones inesperadas en 54 días
con 16.384 GPUs, §3.3.4— cada GPU falla una vez cada ~50.000 horas, y el
clúster entero una vez cada 3. La demo simula 54 días de cómputo con fallas
al azar y compara intervalos de checkpoint con el de Young/Daly, √(2·C·M).
Los costos de guardar y de reiniciar son ilustrativos, elegidos para que con
16.384 GPUs el tiempo útil dé más de 90%, como reporta el paper.

## Demo 7 — sincronizar menos (DiLoCo)

```bash
go run ./cmd/demo diloco
go run ./cmd/demo diloco -workers 32
```

Cada réplica da H pasos sola y recién ahí se promedian los cambios: H veces
menos comunicación. En esta regresión lineal no se pierde nada, pero es el
mejor caso posible (convexo, datos parecidos en todas las réplicas). DiLoCo
(Douillard et al., 2023) muestra que con LLMs también funciona, con 8 réplicas
y 500 veces menos comunicación.

## Ejercicios

1. **Costo del anillo con latencia.** Con el modelo α + β·bytes, el ring
   all-reduce cuesta 2(n-1)·α + 2(n-1)/n·S·β. ¿Para qué tamaños de S conviene
   un algoritmo en árbol (2·log n pasos) en lugar del anillo? Implementá
   `Tree` en `collective` y medí los pasos.
2. **All-reduce jerárquico.** Implementá un all-reduce en dos niveles: anillo
   dentro de cada "nodo" de 8 GPUs y anillo entre nodos. ¿Cuánto tráfico pasa
   por los enlaces entre nodos? Relacionalo con el *Takeaway #1* del paper.
3. **Schedule intercalado.** Agregá a `pipeline` el schedule *interleaved*
   (§2.2.2): cada GPU tiene v pedazos del modelo. Verificá que la burbuja baja
   a (p-1)/(v·m) y contá cuántos mensajes punto a punto más se envían. Llama 3
   usa este schedule.
4. **ZeRO.** En `train` todas las réplicas guardan los pesos completos.
   Modificalo para que cada worker guarde sólo 1/d del estado del optimizador
   (ZeRO etapa 1): después del all-reduce, cada uno actualiza su pedazo y se
   hace un all-gather. ¿Cuánto cambia el tráfico?
5. **Repuestos en caliente.** En `fallas`, el reinicio cuesta lo mismo siempre.
   Modelá nodos de repuesto ya cargados: si hay uno libre, el reinicio cuesta
   R/10; si no, R. ¿Cuántos repuestos hacen falta con 131.072 GPUs?
6. **Checkpoint en memoria.** Guardar en la memoria de otro nodo es mucho más
   barato que en el almacenamiento, pero se pierde si fallan los dos nodos.
   Agregá a `fallas` dos niveles de checkpoint (en memoria seguido, en disco
   cada tanto) y buscá la combinación con más tiempo útil.
7. **Balance de expertos.** En `moe` las afinidades son uniformes. Hacé que
   algunos expertos sean más populares y medí cuánto se desbalancea la carga
   por GPU. DeepSeek-V3 lo corrige con un sesgo por experto que se ajusta
   durante el entrenamiento (§2.1.2): implementalo.

## Referencias

- D. Narayanan et al. *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM*. SC 2021.
- S. Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models*. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- P. Patarasuk y X. Yuan. *Bandwidth Optimal All-reduce Algorithms for
  Clusters of Workstations*. JPDC 69(2), 2009.
- M. Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism*. 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- A. Grattafiori et al. (Meta). *The Llama 3 Herd of Models*. 2024.
  [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- DeepSeek-AI. *DeepSeek-V3 Technical Report*. 2024.
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- W. Liang et al. *TorchTitan: One-stop PyTorch native solution for production
  ready LLM pre-training*. ICLR 2025.
- A. Douillard et al. *DiLoCo: Distributed Low-Communication Training of
  Language Models*. 2023. [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)
- J. W. Young. *A first order approximation to the optimum checkpoint
  interval*. CACM 17(9), 1974. J. T. Daly. *A higher order estimate of the
  optimum checkpoint interval for restart dumps*. FGCS 22(3), 2006.
