# Planificar GPUs — demo de clase

Material práctico de la clase **21 — Repartir un clúster de GPUs** de TA050
Sistemas Distribuidos I (FIUBA). Es un simulador de eventos discretos de un
clúster de GPUs compartido, en Go: racks de servidores de 8 GPUs, una carga
sintética con la forma de las trazas publicadas y los planificadores de la
clase. No hay GPUs ni red: lo que se simula es **quién espera, dónde corre
cada trabajo y cuánto le cuesta correr ahí**.

Paper de la clase: Jeon et al., *Analysis of Large-Scale Multi-Tenant GPU
Clusters for DNN Training Workloads*, USENIX ATC 2019
([USENIX](https://www.usenix.org/conference/atc19/presentation/jeon)), y como
complemento Gu et al., *Tiresias: A GPU Cluster Manager for Distributed Deep
Learning*, NSDI 2019
([USENIX](https://www.usenix.org/conference/nsdi19/presentation/gu)).

## Requisitos

Go 1.24 o superior. No hay dependencias externas.

```bash
go test ./...
```

## Estructura

| Paquete | Qué modela | Concepto de la clase |
|---|---|---|
| `cluster` | Racks, servidores y GPUs libres; dónde ubicar un trabajo y a qué velocidad corre ahí | Localidad, fragmentación, best-fit; por qué repartir un trabajo lo hace más lento |
| `trace` | Una carga sintética: 70% de trabajos de 1 GPU, unos pocos de 16 y 32, duraciones con cola pesada | La forma de Philly: muchos chicos, pocos enormes, nadie sabe cuánto dura su trabajo |
| `sched` | El simulador y los planificadores: FIFO, backfill, "de a un pod", localidad estricta o relajada, 2D-LAS, SRTF | Gang scheduling, bloqueo de cabeza de cola, Tiresias, el costo de desalojar |
| `cmd/sim` | Las comparaciones | Las tablas de la clase |

Un trabajo tiene una cantidad de GPUs y una duración **en una ubicación
ideal**: la menor cantidad de servidores posible, en un rack. Si el
planificador lo reparte en más servidores, corre más lento: al 60% de su
velocidad si es sensible a la localidad (la mitad de los trabajos de varias
GPUs, como VGG o AlexNet en Tiresias), al 95% si no; un poco menos si además
cruza racks. El planificador nunca ve la duración, salvo SRTF, que está como
referencia de lo que se podría hacer sabiéndola.

## Demo 1 — la carga

```bash
go run ./cmd/sim carga
go run ./cmd/sim carga -semilla 7
```

El 70% de los trabajos usa una GPU y pide el 10% del GPU·tiempo; el 6% usa
más de 8 y pide el 45%. La mediana dura media hora, el p99 más de tres días, y
el 1% de los trabajos más grandes se lleva la mitad del clúster.

## Demo 2 — gang scheduling, FIFO y backfill

```bash
go run ./cmd/sim traba
go run ./cmd/sim gang
go run ./cmd/sim gang -carga 1.1
```

`traba` es el caso mínimo: 8 GPUs y dos trabajos de 8 que llegan juntos. Si
cada uno junta GPUs de a una (como un orquestador que ubica cada pod por su
cuenta), cada uno se queda con 4 y espera las del otro para siempre.

`gang` corre la carga entera con tres planificadores:

- **de a un pod**: sin gang. No llega a trabarse del todo porque siempre
  termina algún trabajo chico, pero las GPUs retenidas esperando a las que
  faltan son miles de GPU·horas tiradas, y todos esperan alrededor de un día.
- **FIFO con gang**: el primero de la cola entra entero o nadie pasa. Un
  trabajo de 32 GPUs al frente deja GPUs libres sin usar: los de 1 GPU esperan
  horas.
- **FIFO + backfill**: si el primero no entra, pasan los que entran. Los chicos
  ya no esperan, y los grandes esperan más: siempre hay un chico que les gana
  el lugar.

## Demo 3 — localidad

```bash
go run ./cmd/sim localidad
go run ./cmd/sim localidad -relajar 120
```

Cuatro formas de ubicar, todas con FIFO + backfill:

- **consolidada siempre**: los grandes esperan servidores vacíos; buena parte
  de esa espera es con GPUs libres de sobra pero repartidas (la columna
  *frag. >8*, la espera por fragmentación de la tabla 2 de Jeon et al.).
- **donde haya lugar**: arrancan antes pero corren más lento, y como corren
  horas, pierden más de lo que ganaron.
- **relajar después de esperar**: lo que hacía Philly, consolidada hasta un
  tiempo de espera y después donde haya lugar.
- **consolidar sólo si importa**: Tiresias. Los trabajos sensibles esperan
  su lugar; los demás van donde haya. El JCT de consolidar siempre con menos
  espera.

## Demo 4 — sin conocer las duraciones

```bash
go run ./cmd/sim duraciones
go run ./cmd/sim duraciones -preempcion 600
```

FIFO, FIFO + backfill, 2D-LAS con dos colas (Tiresias) con dos umbrales, y
SRTF, que conoce las duraciones. 2D-LAS le da prioridad al trabajo que menos
GPU·tiempo recibió: los nuevos pasan primero, y los que ya corrieron mucho
(que probablemente van a correr mucho más) bajan a la segunda cola. Sin saber
nada de las duraciones, queda a pocos puntos de SRTF.

El umbral importa: con el de Tiresias (3.200 GPU·s, para su testbed) en esta
carga de trabajos de horas, casi todo baja de cola enseguida y desaloja más de cuatro
veces más. Y cada preempción cuesta guardar y recargar un checkpoint: el
default es 62 s, lo que midió Tiresias (13.724 s en 221 preempciones).

## Ejercicios

1. **Reserva para el primero (EASY backfill).** El backfill de `sched` es
   codicioso: los grandes esperan horas. Implementá la variante clásica de los
   clústeres de HPC: el primero de la cola reserva la hora en que va a entrar,
   y sólo pasan por adelante los que terminan antes. ¿Qué duración usás si el
   usuario no la sabe?
2. **Anti-hambre.** Agregá el `PROMOTEKNOB` de Tiresias: si un trabajo
   esperó más de K veces lo que corrió, vuelve a la primera cola. ¿Qué pasa
   con el p99 del JCT?
3. **Cuotas por equipo.** Philly tenía 14 clústeres virtuales, uno por equipo,
   con una cuota cada uno. Agregá un equipo a cada trabajo y una cuota de GPUs
   por equipo, con préstamo de lo que otro equipo no usa y preempción para
   devolverlo.
4. **Fallar temprano.** En Philly, el 30,7% de los trabajos terminó mal o lo
   mataron, y usaron el 55% del GPU·tiempo. Marcá una fracción de trabajos como
   fallidos (fallan al rato de arrancar) y agregá una cola de prueba de 1 GPU
   por donde pasan antes de pedir todas. ¿Cuánto GPU·tiempo se ahorra?
5. **Migrar para desfragmentar.** Cuando un trabajo grande espera por
   fragmentación, mové trabajos chicos para liberar servidores enteros (Gandiva
   migra en segundos). ¿Cuánto baja la columna *frag. >8*?
6. **Compartir una GPU.** En el clúster de Alibaba la mediana de uso de una
   instancia era 0,042 GPUs. Permití que dos trabajos de 1 GPU compartan una,
   con un costo de interferencia. ¿Cuánto baja la espera de los chicos?

## Referencias

- M. Jeon et al. *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN
  Training Workloads*. USENIX ATC 2019.
  [USENIX](https://www.usenix.org/conference/atc19/presentation/jeon)
- J. Gu et al. *Tiresias: A GPU Cluster Manager for Distributed Deep
  Learning*. NSDI 2019.
  [USENIX](https://www.usenix.org/conference/nsdi19/presentation/gu)
- W. Xiao et al. *Gandiva: Introspective Cluster Scheduling for Deep
  Learning*. OSDI 2018.
  [USENIX](https://www.usenix.org/conference/osdi18/presentation/xiao)
- Q. Weng et al. *MLaaS in the Wild: Workload Analysis and Scheduling in
  Large-Scale Heterogeneous GPU Clusters*. NSDI 2022.
  [USENIX](https://www.usenix.org/conference/nsdi22/presentation/weng)
- A. Choudhury et al. *MAST: Global Scheduling of ML Training across
  Geo-Distributed Datacenters at Hyperscale*. OSDI 2024.
  [USENIX](https://www.usenix.org/conference/osdi24/presentation/choudhury)
- Kueue, [All-or-nothing con waitForPodsReady](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/).
