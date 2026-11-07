import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidekit import *
from cover import cover

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "clase-20")
d = Deck(ROOT, 20, "Muchos modelos, pocas GPUs")
D, L = DARK, LIGHT

# 1 ─ portada
d.raw("cover", cover(20, "Muchos modelos, pocas GPUs",
    "Servir cientos de modelos y miles de fine-tunes con una flota compartida, cuando casi todos están ociosos casi todo el tiempo",
    "Paper: Fu et al., <i>ServerlessLLM: Low-Latency Serverless Inference for Large Language Models</i>, OSDI 2024 · con Sheng et al., <i>S-LoRA</i>, MLSys 2024",
    """Tiempo: 0:00–0:02 (2 min).
Idea clave: en las clases 18 y 19 había un modelo con mucha demanda; hoy hay muchos modelos con poca demanda cada uno, y el problema pasa a ser dónde y cuándo cargar cada uno.
Decir: hasta ahora servimos un modelo: primero una réplica por dentro (Clase 18), después muchas réplicas detrás de un router (Clase 19). Pero una plataforma real no sirve un modelo: sirve cientos de modelos abiertos y miles de versiones ajustadas por sus clientes, y la gran mayoría recibe un pedido cada tanto.
Darle una GPU a cada uno es tirar plata: casi todas estarían ociosas. Compartir las GPUs obliga a cargar el modelo cuando llega un pedido, y un modelo son decenas de gigabytes.
El paper de hoy es ServerlessLLM (Universidad de Edimburgo, OSDI 2024): cómo hacer que ese arranque en frío tarde segundos y no minutos. Y de compañero, S-LoRA (Berkeley, MLSys 2024): cómo servir miles de fine-tunes de un mismo modelo base en una sola GPU.
Esta clase abre la serie 20 a 22: las tres son sobre cómo se reparte un clúster de GPUs.
Pasar a la siguiente: qué cambia respecto de la clase pasada."""), section="El problema: muchos modelos, carga en ráfagas, y una GPU por modelo no cierra.")

# 2 ─ mapa
d.slide("mapa", D, head("Dónde estamos", "De un modelo muy pedido a muchos modelos poco pedidos", D) +
    table(["", "Clase 19: un modelo", "Hoy: una flota"], [
        ["Cuántos modelos", "uno, con muchas réplicas", "cientos, casi todos ociosos"],
        ["Lo que se mueve", "el KV cache", "los pesos: decenas de GB"],
        ["El scheduler decide", "dónde está el prefijo", "dónde está el checkpoint"],
        ["Fine-tunes", "—", "miles de adaptadores sobre un modelo base"],
    ], D, widths=[24, 34, 42], size=32),
    """Tiempo: 0:02–0:04 (2 min).
Idea clave: la estructura es la misma que la Clase 19 —un estado grande, un scheduler que sabe dónde está—, pero el estado ahora son los pesos del modelo.
Decir: recorrer las filas. En la Clase 19 había un modelo con mucha demanda y lo que se cacheaba y se movía era el KV cache. Hoy el estado grande son los checkpoints: 13 GB un modelo de 7B, 130 GB uno de 70B. El scheduler ya no pregunta "¿dónde está este prefijo?" sino "¿dónde está este modelo, en DRAM, en SSD o en ningún lado?".
La última fila es la segunda parte de la clase: cuando los modelos son fine-tunes de uno mismo, hay un truco mucho mejor que cargarlos enteros.
Pasar a la siguiente: cómo es la carga."""
    , d.footer())

# 3 ─ carga (claro)
d.slide("carga", L, col(eyebrow("La carga · Azure Functions, citado en ServerlessLLM §2.3", L), h2("Ráfagas, y arranques en frío todo el tiempo", L), gap=16) +
    row(big("&gt; 40%", "de las funciones arranca en frío en más de un cuarto de sus pedidos", L),
        big("≈ 25%", "arranca en frío en más del 60% de sus pedidos", L, color="warn"),
        big("5&nbsp;min", "de keep-alive: aun así", L, color="blue"), gap=96) +
    p("Para un modelo de lenguaje, cada arranque en frío es traer decenas de GB a una GPU. Los proveedores reportan latencias de decenas de segundos al iniciar un modelo.", L),
    """Tiempo: 0:04–0:07 (3 min).
Idea clave: la carga de una plataforma con muchos modelos es en ráfagas: un modelo recibe nada por un rato y de golpe muchos pedidos. Con eso, los arranques en frío no son la excepción.
Decir: no hay trazas públicas de serverless para modelos de lenguaje, así que ServerlessLLM usa la de Azure Functions (Shahrad et al., "Serverless in the Wild", ATC 2020), que es la referencia de la literatura. Con un keep-alive de 5 minutos —una función queda cargada 5 minutos después de su último pedido—, más del 40% de las funciones tiene una tasa de arranques en frío mayor al 25%, y cerca de un cuarto, mayor al 60%.
Para una función de Azure, un arranque en frío es levantar un contenedor: cientos de milisegundos. Para un modelo de lenguaje es traer el checkpoint. El paper cita a proveedores, Bloomberg entre ellos, que reportan decenas de segundos para iniciar un modelo.
Preguntar a la clase: ¿por qué no dejar todos los modelos cargados siempre? (La próxima slide.)
Pasar a la siguiente: la idea de serverless.""",
    d.footer("Fu et al., ServerlessLLM, §2.3 · Shahrad et al., ATC 2020", L), gap=64)

# 4 ─ serverless
def node(title, sub, t=D, color="line", w=None, accent=False):
    c = t[color] if color in t else color
    size = f"width:{w}px;" if w else "flex:1;"
    return (f'<div style="{size}display:flex;flex-direction:column;gap:8px;padding:24px;border-radius:12px;background:{t["card"]};border:{"2px" if accent else "1px"} solid {c}">'
            f'{h3(title, t, size=28)}{p(sub, t, size=24)}</div>')
d.slide("serverless", D, head("Serverless para modelos · §2.2", "Se paga por pedido; las GPUs se comparten", D) +
    row(node("Router de pedidos", "manda cada pedido a una GPU que ya tiene el modelo, o pide arrancar uno", D),
        node("Scheduler de carga", "elige en qué servidor y GPU cargar el modelo", D, color="acc", accent=True),
        node("Almacenamiento de modelos", "los checkpoints que subió cada cliente: S3 o parecido", D), gap=24) +
    row(node("Servidor de GPU", "GPUs, mucha DRAM y SSDs; el proceso de inferencia (vLLM, Accelerate)", D, w=1000),
        node("Ejemplos", "SageMaker, Azure, KServe, Hugging Face, Together, Replicate…", D), gap=24) +
    p("Para el cliente: no paga GPUs ociosas. Para el proveedor: muchos modelos sobre las mismas GPUs.", D, size=28),
    """Tiempo: 0:07–0:10 (3 min).
Idea clave: serverless es multiplexar: muchos modelos comparten las GPUs, y el costo es que a veces hay que cargar el modelo antes de atender.
Decir: la figura 1 del paper. El cliente sube su checkpoint —el modelo abierto tal cual, uno ajustado, o uno propio— y paga por pedido o por token. Cuando llega un pedido, el router lo manda a una GPU que ya tenga ese modelo cargado; si no hay, le pide al scheduler de carga que arranque uno en alguna GPU libre, y ese servidor descarga el checkpoint, lo pasa por SSD y DRAM y lo sube a la GPU.
Ofrecen esto Amazon SageMaker, Azure, Google, Hugging Face, Together, Replicate, Fireworks, Databricks y otros. KServe es la versión abierta sobre Kubernetes.
Conectar con la materia: es la misma separación de la Clase 19 entre router y motores, con un componente nuevo: el que decide qué está cargado dónde.
Pasar a la siguiente: por qué no darle una GPU a cada modelo.""",
    d.footer("Fu et al., OSDI 2024, §2.1–2.2 y figura 1"), gap=32)

# 5 ─ multiplexación
d.slide("multiplex", D, head("Multiplexación estadística", "Las ráfagas de muchos modelos no coinciden", D) +
    row(col(card(label("Una GPU por modelo", D) + p("Sin arranques en frío. Pero cada GPU se dimensiona para el pico de su modelo, y el resto del tiempo está ociosa.", D, size=28), D, pad=36),
            card(label("Una flota compartida", D, color="acc") + p("Se dimensiona para el pico de la suma, que es mucho menor que la suma de los picos.", D, size=28), D, accent=True, pad=36), gap=24, flex=1),
        col(label("AlpaServe · OSDI 2023", D, color="blue"),
            big("10×", "más pedidos por segundo, o 6× más ráfagas, con más del 99% de los pedidos dentro del SLO", D, size=80, color="blue"),
            p("Demo: 64 modelos en 64 GPUs dedicadas, ocupadas el 4% del tiempo; en 16 compartidas, el 16%.", D, size=24, color="muted"), gap=20, width=640), gap=56, align="center"),
    """Tiempo: 0:10–0:13 (3 min).
Idea clave: si las ráfagas de los distintos modelos no coinciden, compartir las GPUs atiende la misma carga con muchas menos; es la misma razón por la que existe la nube.
Decir: el argumento clásico. Con una GPU por modelo hay que dimensionar cada una para el pico de su modelo, y como la carga es en ráfagas, el pico es mucho más alto que el promedio. Con una flota compartida alcanza con el pico de la suma: cuando un modelo tiene su ráfaga, los otros probablemente no.
AlpaServe (Li et al., OSDI 2023) lo lleva más lejos: muestra que conviene incluso partir un modelo que entra en una GPU entre varias (paralelismo de modelo), para que una ráfaga de un modelo use muchas GPUs a la vez. Reportan hasta 10 veces más pedidos por segundo, o 6 veces más ráfagas, cumpliendo el SLO en más del 99% de los pedidos.
En la demo del repositorio: 64 modelos con una GPU cada uno están ocupadas un 4% del tiempo; en una flota de 16, un 16%. Cuatro veces menos GPUs.
Pasar a la siguiente: el precio de compartir, que es cargar.""",
    d.footer("Li et al., AlpaServe, OSDI 2023 · go run ./cmd/sim arranque"), gap=40)

# 6 ─ tamaños
d.slide("tamanos", D, head("Anatomía de un arranque en frío · §2.3", "El checkpoint es enorme y está lejos", D) +
    row(col(table(["Modelo", "Checkpoint en FP16"], [
        ["LLaMA-2-70B", "130 GB"], ["DBRX", "250 GB"], ["Mixtral-8x22B", "≈ 280 GB"], ["Grok-1", "&gt; 600 GB"],
    ], D, widths=[55, 45], right=(1,), size=28), gap=16, flex=1),
        col(big("≥ 26 s", "para bajar 130 GB de S3 con una red de 5 GB/s", D, size=88, color="warn"),
            big("&lt; 100 ms", "lo que tarda generar un token", D, size=88, color="acc"), gap=40, width=640), gap=64, align="center"),
    """Tiempo: 0:13–0:16 (3 min).
Idea clave: un arranque en frío de un modelo de lenguaje cuesta cientos de veces lo que cuesta generar un token; el primer token de un pedido frío lo domina la carga.
Decir: la primera causa son los tamaños. LLaMA-2-70B en FP16 son 130 GB; DBRX, 250; Mixtral-8x22B, unos 280; Grok-1, más de 600. Bajar 130 GB de S3 o de un blob storage, con una red de 5 GB/s, que es rápida, lleva como mínimo 26 segundos.
Comparar con la Clase 18: generar un token cuesta menos de 100 ms. Un pedido que llega a un modelo frío espera cientos de tokens antes de empezar.
Preguntar a la clase: ¿de dónde se podría traer más rápido? (Del servidor mismo: la próxima slide.)
Pasar a la siguiente: los niveles de almacenamiento.""",
    d.footer("Fu et al., OSDI 2024, §2.3"), gap=40, section="Anatomía de un arranque en frío: niveles, formato y carga en pipeline.")

# 7 ─ niveles
def lvl(name, size_, speed, t=D, w=900, hi=False):
    return (f'<div style="display:flex;gap:32px;align-items:center"><div style="width:{w}px;height:64px;border-radius:10px;'
            f'background:{t["acc"] if hi else t["fill"]};display:flex;align-items:center;justify-content:space-between;padding:0 24px">'
            f'<p style="font-family:{MONO};font-size:24px;font-weight:600;color:{t["onfill"] if hi else t["ink"]}">{name}</p>'
            f'<p style="font-family:{MONO};font-size:24px;color:{t["onfill"] if hi else t["muted"]}">{size_}</p></div>{p(speed, t, size=24)}</div>')
d.slide("niveles", D, head("Usar lo que ya tiene el servidor · §3.1", "Un servidor de GPU tiene una jerarquía de almacenamiento casi ociosa", D) +
    col(lvl("GPU", "HBM", "512 GB/s desde la DRAM, sumando las 8 GPUs por PCIe 5.0", w=420, hi=True),
        lvl("DRAM", "hasta 4 TB", "≈ 60 GB/s desde NVMe en RAID 0", w=640),
        lvl("SSD", "64 TB NVMe · 192 TB SATA", "en el servidor mismo", w=860),
        lvl("remoto", "S3, blob", "≥ 26 s para 130 GB a 5 GB/s", w=1060), gap=16) +
    p("Guardar los checkpoints en DRAM y SSD del servidor: escala sola con la flota y no cuesta hardware nuevo.", D, size=28),
    """Tiempo: 0:16–0:19 (3 min).
Idea clave: el servidor de inferencia ya tiene la capacidad y el ancho de banda para guardar checkpoints localmente; los sistemas serverless no la usaban.
Decir: la observación de la que parte el paper (§3.1). Un servidor de 8 GPUs actual puede tener hasta 4 TB de DRAM, 64 TB en NVMe y 192 TB en SSDs SATA. Y en anchos de banda: con PCIe 5.0, cada GPU tiene su propio enlace con la DRAM, y sumados dan 512 GB/s; de NVMe en RAID 0 a DRAM, unos 60 GB/s. KServe o Ray Serve usaban una fracción de la memoria del host y casi nada de los SSDs.
Por qué es buena idea: es barato (el hardware ya está), escala solo (cada servidor nuevo trae su DRAM y su SSD) y va a mejorar (un Grace Hopper tiene 1 TB de memoria de CPU con 900 GB/s hasta la HBM).
El paper descarta dos alternativas: guardar sólo en DRAM no alcanza para modelos de cientos de GB, y poner servidores de almacenamiento aparte es caro: con una NIC de 100 Gb/s la descarga igual pasa los 20 segundos, y un servidor de caché de AWS con 210 GB y 200 Gb/s cuesta 16,3 dólares la hora, lo mismo que un servidor de 8 GPUs g5.48xlarge.
Conectar con la Clase 19: es la misma jerarquía que el KV cache de Mooncake, con otro contenido.
Pasar a la siguiente: aun desde el SSD local, cargar tarda.""",
    d.footer("Fu et al., OSDI 2024, §2.4 y §3.1"), gap=40)

# 8 ─ cargar (claro)
d.slide("cargar", L, head("Figura 6a · desde NVMe local de 12 GB/s", "Aun con el checkpoint en el SSD, cargar tarda decenas de segundos", L) +
    table(["Modelo", "PyTorch", "Safetensors", "ServerlessLLM"], [
        ["OPT-2.7B", "3,0 s", "1,8 s", "0,5 s"],
        ["OPT-30B", "34 s", "18,5 s", "4,5 s"],
        ["Falcon-40B", "50 s", "25 s", "6,2 s"],
        ["LLaMA-2-70B", "84 s", "48 s", "10,3 s"],
    ], L, widths=[31, 23, 23, 23], right=(1, 2, 3), size=32, bold_rows=(3,)) +
    p("Entre 3,6× y 8,2× más rápido. Safetensors sufre fallos de página (112 mil para LLaMA-2-7B en frío); PyTorch copia dos veces.", L, size=28),
    """Tiempo: 0:19–0:22 (3 min).
Idea clave: el cuello de botella no es sólo bajar el archivo: el cargador de siempre no aprovecha ni una fracción del ancho de banda del SSD.
Decir: la figura 6a, con los checkpoints en un NVMe en RAID 0 de 12 GB/s y las cachés del sistema operativo vaciadas. PyTorch tarda 84 segundos en cargar LLaMA-2-70B en 8 GPUs, aunque el archivo esté al lado. Safetensors, que usa mmap, 48. ServerlessLLM, 10,3: 8,2 y 4,7 veces más rápido. Para el modelo más chico, 6 y 3,6 veces.
¿Por qué son lentos? PyTorch lee tensor por tensor, primero a la memoria del host y después a la GPU: copia todo dos veces. Safetensors usa mmap, y en frío cada página es un fallo de página: 112 mil para LLaMA-2-7B. Y ninguno de los dos aprovecha un dispositivo rápido: en la figura 6b, sobre el RAID 0 de NVMe usan una fracción chica del ancho de banda.
Un detalle: el tiempo de ServerlessLLM depende sólo del tamaño, no del modelo: OPT-13B y LLaMA-2-13B tardan lo mismo.
Pasar a la siguiente: qué hace distinto.""",
    d.footer("Fu et al., OSDI 2024, §2.3 y §7.2, figura 6a", L), gap=40)

# 9 ─ formato
d.slide("formato", D, head("Checkpoint pensado para leer · §4.1", "Se escribe una vez y se carga miles", D) +
    row(card(label("Particiones por GPU", D, color="acc") + p("Un archivo por GPU con sólo los bytes de sus tensores, uno detrás de otro: se lee de corrido en bloques grandes.", D, size=28), D, accent=True, pad=36),
        card(label("Índice de tensores", D) + p("nombre → (GPU, offset, tamaño), alineado. La dirección de cada tensor es base + offset: no hay que interpretar nada.", D, size=28), D, pad=36),
        card(label("Cargar aparte", D) + p("Un model manager carga los bytes mientras el proceso de inferencia arma el modelo; después sólo apunta cada tensor.", D, size=28), D, pad=36), gap=32),
    """Tiempo: 0:22–0:25 (3 min).
Idea clave: los formatos de checkpoint están pensados para entrenar, donde se escribe mucho y se lee poco; en serverless es al revés, así que conviene convertir el checkpoint una vez a un formato que se lea rápido.
Decir: durante el entrenamiento se guardan checkpoints todo el tiempo y casi nunca se leen; en serverless, el cliente sube el modelo una vez y se carga miles de veces. Entonces ServerlessLLM lo convierte, al subirlo, a un formato propio con dos cosas.
Uno, particiones por GPU: el plan de paralelismo dice qué tensor va a qué GPU, así que para cada GPU se arma un archivo con sólo los bytes de sus tensores, sin metadatos en el medio. Se lee secuencial, en bloques grandes.
Dos, un índice que dice, para cada tensor, en qué GPU va, en qué offset y cuánto mide, alineado a palabras de memoria. Restaurar un tensor es base de la GPU más offset.
Y separa la carga de la inferencia: un model manager reserva la memoria en las GPUs y carga los bytes; el proceso de inferencia, mientras tanto, construye el objeto del modelo, y al final recibe las direcciones base (handles de CUDA IPC) y apunta cada tensor a su lugar.
Conectar con la materia: es la misma idea que un formato de log o de columnas: decidir el layout en disco por el patrón de acceso.
Pasar a la siguiente: cómo se mueven esos bytes.""",
    d.footer("Fu et al., OSDI 2024, §4.1 y figura 2"), gap=40)

# 10 ─ pipeline
def mult(name, x, t=D):
    return card(p(x, t, size=56, color="acc", mono=True, weight=600, lh=1) + p(name, t, size=24), t, pad=28, gap=12)
d.slide("pipeline", D, head("Cargador multinivel · §4.2 y figura 7", "Cada optimización suma, y en pipeline manda el nivel más lento", D) +
    row(mult("leer en bloques de 16 MB", "1,2×"), mult("O_DIRECT: sin caché del kernel", "2,1×"), mult("varios hilos por SSD", "2,3×"),
        mult("memoria fijada: DMA directo a la GPU", "1,4×"), mult("pipeline entre niveles", "1,5×"), gap=20) +
    p("Remoto → SSD → DRAM → GPU, por pedazos: mientras un pedazo sube a la GPU, el siguiente ya está en DRAM. El tiempo total es tamaño ÷ el ancho de banda del nivel más lento.", D, size=28) +
    p("Multiplicadas, unas 12 veces sobre leer tensor por tensor. Cuatro núcleos de CPU alcanzan.", D, size=28, color="muted"),
    """Tiempo: 0:25–0:29 (4 min).
Idea clave: con el formato correcto, el cargador puede saturar cada nivel; y en pipeline, el tiempo de carga es el del nivel más lento, no la suma.
Decir: la figura 7 parte de leer tensor por tensor y agrega una cosa por vez, midiendo cuánto sube el throughput. Leer en bloques grandes (16 MB): 1,2 veces; un tercio de los tensores mide menos de 1 MB. Leer con O_DIRECT, sin pasar por la caché de páginas del kernel: 2,1 veces. Varios hilos, porque un SSD tiene muchos canales que atienden en paralelo: 2,3. Memoria fijada (pinned), así la copia a la GPU es DMA sin pasar por la CPU: 1,4. Y el pipeline entre niveles: 1,5. Multiplicadas dan unas 12 veces; la cuenta es nuestra, las barras del paper.
Además, la DRAM se maneja como un pool de pedazos de tamaño fijo (sin fragmentación) y cada GPU usa su enlace PCIe en paralelo.
El pipeline es lo que hace que la estimación sea simple: si el checkpoint está en SSD, el tiempo es tamaño sobre ancho de banda del SSD, porque la DRAM y el PCIe son mucho más rápidos y van en paralelo. Lo vamos a usar en el scheduler.
Conectar con la Clase 17: es el mismo principio que superponer comunicación y cómputo.
Pasar a la siguiente: la demo.""",
    d.footer("Fu et al., OSDI 2024, §4.2 y §7.2"), gap=40)

# 11 ─ demo carga (claro)
d.slide("democarga", L, head("Demo · arranque, tabla 1", "Dónde está el checkpoint decide el arranque, y el cargador lo multiplica", L) +
    table(["Segundos hasta la GPU", "desde DRAM", "desde SSD", "desde remoto"], [
        ["13 GB, cargador ingenuo", "1,7", "9,0", "19,4"],
        ["13 GB, cargador en pipeline", "0,5", "2,2", "10,4"],
        ["130 GB, cargador ingenuo", "17,3", "89,6", "193,6"],
        ["130 GB, cargador en pipeline", "5,2", "21,7", "104,0"],
    ], L, widths=[40, 20, 20, 20], right=(1, 2, 3), size=30, bold_rows=(1, 3)) +
    p("Modelo del repositorio: red de 10 Gb/s, SSD de 6 GB/s, PCIe de 25 GB/s. Ingenuo: nivel por nivel, con el 30% del ancho de banda local.", L, size=24, color="muted"),
    """Tiempo: 0:29–0:31 (2 min, con demo).
Idea clave: con un cargador en pipeline, un modelo de 13 GB en DRAM está en la GPU en medio segundo; desde remoto tarda veinte veces más.
Demo: go run ./cmd/sim arranque, la primera tabla. Y en el simulador Flota de GPUs, cambiar el cargador y mirar cómo cambia el color de cada carga.
Leer la tabla: los anchos de banda son del orden del testbed del paper —red de 10 Gb/s, un NVMe, PCIe 4.0—, no medidos. El cargador ingenuo pasa por cada nivel uno después del otro y aprovecha el 30% del ancho de banda local; el de ServerlessLLM va en pipeline y satura cada nivel, así que manda el más lento.
Desde DRAM: 0,5 contra 1,7 segundos. Desde remoto, aun en pipeline, 10 segundos: la red de 10 Gb/s es el cuello. Para 130 GB, de 5 segundos a más de 3 minutos según dónde esté y cómo se lea.
La conclusión que usa el scheduler: dónde está el checkpoint importa tanto como el cargador. Si el modelo está en la DRAM de algún servidor, conviene arrancarlo ahí.
Pasar a la siguiente: dónde arrancar.""",
    d.footer("go run ./cmd/sim arranque · simulador Flota de GPUs", L, plain=True), gap=40)

# 12 ─ estimar
d.slide("estimar", D, head("Ubicación consciente de la localidad · §6", "El scheduler estima cuánto tarda en arrancar en cada servidor", D) +
    row(card(label("Cargar · §6.1", D, color="acc") + p("q + n ÷ b", D, size=56, color="ink", mono=True, weight=600, lh=1.1) +
             p("q: lo que ya está en la cola de carga del servidor (las cargas van de a una). n: tamaño. b: ancho de banda del nivel más lento del camino.", D, size=26), D, accent=True, pad=36),
        card(label("Mudar una inferencia · §6.2", D, color="blue") + p("α·(t<sub>in</sub> + t<sub>out</sub>) + β", D, size=56, color="ink", mono=True, weight=600, lh=1.1) +
             p("recalcular el KV cache de los tokens de entrada y los generados. t<sub>out</sub> se estima como duración ÷ tiempo por token; α y β se ajustan con lo medido.", D, size=26), D, pad=36), gap=32) +
    p("Error de la estimación: hasta 5 ms en GPU y 40 ms desde SSD. El scheduler corrige los anchos de banda con lo que va midiendo.", D, size=28),
    """Tiempo: 0:31–0:34 (3 min).
Idea clave: como la carga en pipeline es predecible, el scheduler puede estimar el tiempo de arranque en cada servidor con una cuenta simple y elegir el menor.
Decir: el scheduler de ServerlessLLM tiene dos estimadores. El de carga: para cada servidor, la cola de carga que ya tiene (q) más el tamaño del modelo (n) sobre el ancho de banda (b) del nivel donde está el checkpoint en ese servidor. Las cargas de un servidor van de a una a propósito: dos cargas a la vez se pelean el SSD y la estimación deja de servir.
El de migración, que vamos a ver en un momento: cuánto tarda otro servidor en recalcular el KV cache de una inferencia en curso, lineal en los tokens. Para no preguntarle a cada servidor cuántos tokens generó, lo estima con lo que sabe el router: cuánto hace que empezó y cuánto tarda un token.
Precisión: en sus mediciones el error está acotado en 5 ms en la GPU y 40 ms desde SSD; la limpieza del estado de la GPU después de una migración lo empeora, en promedio 26 ms y una vez 623 ms en 119 migraciones.
Conectar con la Clase 19: es el router consciente del prefijo de Mooncake con otra fórmula: allá cola más prefill de lo que falta, acá cola más carga de lo que no está.
Pasar a la siguiente: qué pasa si el mejor servidor está ocupado.""",
    d.footer("Fu et al., OSDI 2024, §6.1, §6.2 y §7.3"), gap=36, section="Dónde arrancar: localidad, migración en vivo y el estado del scheduler.")

# 13 ─ políticas
d.slide("politicas", D, head("Figura 3 · el servidor bueno está ocupado", "Cuatro formas de arrancar el modelo B", D) +
    p("S1: GPU libre, A en DRAM, B en SSD. &nbsp; S2: GPU corriendo A, B en DRAM.", D, size=28, color="ink", mono=True) +
    table(["Política", "Modelo B (llega)", "Modelo A (corriendo)"], [
        ["Disponibilidad", "carga desde el SSD de S1: lento", "no se entera"],
        ["Localidad", "espera a que A termine en S2", "no se entera; S1 queda ociosa"],
        ["Desalojo", "arranca ya en S2, desde DRAM", "se corta: recarga y recalcula en S1"],
        ["Migración en vivo", "arranca en S2 después de mudar A", "sigue en S1 con una pausa mínima"],
    ], D, widths=[24, 40, 36], size=28, bold_rows=(3,)),
    """Tiempo: 0:34–0:38 (4 min).
Idea clave: con inferencias largas e impredecibles, ir a donde está el checkpoint choca con que ese servidor esté ocupado; la salida es mudar la inferencia que estorba.
Decir: el ejemplo de la figura 3. Dos servidores. S1 tiene la GPU libre, A en DRAM y B en SSD. S2 está corriendo una inferencia de A y tiene B en DRAM. Llega un pedido para B.
Disponibilidad, el scheduler serverless de siempre: hay una GPU libre en S1, carga B desde el SSD. A no se entera, B tarda.
Localidad pura: B está en la DRAM de S2, así que espera a que A termine. Pero la duración de una inferencia de un LLM no se conoce de antemano: depende de cuántos tokens genere. B puede esperar mucho, y S1 queda ociosa.
Desalojo, como Shepherd: corta A en S2, arranca B desde DRAM, y vuelve a arrancar A en S1. B queda bien, pero A pierde todo lo que llevaba: recarga el modelo y recalcula.
Migración en vivo: primero carga A en S1 —está en su DRAM—, después pasa el estado de la inferencia de A a S1 y A sigue ahí; recién entonces S2 carga B desde DRAM. Las dos quedan bien.
Preguntar a la clase: ¿qué estado hay que pasar de S2 a S1 para que A siga? (La próxima slide.)
Pasar a la siguiente: cómo se muda una inferencia.""",
    d.footer("Fu et al., OSDI 2024, §5.1 y figura 3"), gap=36)

# 14 ─ migración
d.slide("migracion", D, head("Migración en vivo · §5.2", "Se mandan los tokens y el destino recalcula el KV cache", D) +
    row(col(table(["", "Mandar el KV cache", "Mandar los tokens"], [
        ["Bytes por la red", "1 a decenas de GB", "decenas a cientos de KB"],
        ["En el destino", "nada", "un prefill"],
    ], D, widths=[30, 35, 35], size=28),
        p("Recalcular el KV cache de 1.000 tokens cuesta lo mismo que generar unos 100: el destino alcanza al origen en pocas rondas.", D, size=28), gap=28, flex=1),
        card(label("Clase 19, al revés", D, color="warn") + p("Mooncake trae el KV cache porque tiene 800 Gb/s de RDMA. Acá la red es Ethernet de 10 Gb/s y el destino está libre: conviene recalcular.", D, size=28), D, width=560, pad=36), gap=48, align="center"),
    """Tiempo: 0:38–0:42 (4 min).
Idea clave: el estado de una inferencia en curso son sus tokens; el KV cache es derivado y se puede recalcular, y recalcular es un prefill, mucho más rápido que generar.
Decir: la opción obvia sería un snapshot del proceso, como en la migración de máquinas virtuales; el paper la descarta porque crear y mandar el snapshot tarda decenas de segundos o minutos, y la migración por páginas sucias no está soportada en contenedores con GPU.
La observación: el estado que importa son los tokens —el prompt y lo generado—, decenas a cientos de KB. El KV cache son de 1 a decenas de GB, pero se puede reconstruir a partir de los tokens. Y reconstruirlo es un prefill: según las mediciones que citan, recalcular el KV cache de 1.000 tokens tarda lo que generar unos 100.
Entonces se hace por rondas: el origen manda los tokens que tiene y sigue generando; el destino recalcula. Cuando termina, el origen tiene unos pocos tokens nuevos, que son una ronda mucho más corta. Cuando la diferencia es chica, el origen para, manda lo último, y el router redirige el stream al destino. Es la misma idea que la migración en vivo de máquinas virtuales por pre-copia: copiar mientras corre, en rondas que achican lo pendiente, y parar sólo para lo último.
Conectar con la Clase 19: allá Mooncake decidía entre traer y recalcular y le convenía traer, porque tiene RDMA de 800 Gb/s y el prefill de prompts de 8.000 tokens es caro. Acá la red es de 10 Gb/s y el destino está ocioso. Misma pregunta, otra respuesta: depende del ancho de banda contra el cómputo.
Pasar a la siguiente: los pasos y qué pasa si algo falla.""",
    d.footer("Fu et al., OSDI 2024, §5.2 · Qin et al., FAST 2025"), gap=40)

# 15 ─ pasos
steps = ["Cargar A en el destino (si no hay una instancia ociosa)", "El scheduler le pide al origen que migre",
         "El origen manda los tokens y sigue generando", "El destino recalcula el KV cache; se repite con lo nuevo",
         "El origen para y le devuelve todo al router, con «migrado»", "Se descarga A del origen y se carga B",
         "El router cambia el destino en su tabla y sigue el stream"]
d.slide("pasos", D, head("El protocolo · §5.3 y §5.4", "Siete pasos, y qué pasa si alguien se cae", D) +
    row(f'<ol style="flex:1.3;font-size:30px;line-height:1.5;color:{D["ink"]}">' + "".join(f"<li>{s}</li>" for s in steps) + "</ol>",
        col(card(label("Falla el origen", D, color="warn") + p("Antes de migrar: se aborta y se descarga el destino. Durante: el destino tira el KV cache.", D, size=24), D, pad=28),
            card(label("Falla el destino", D, color="warn") + p("Cargando: se cancela. Recalculando: el origen sigue como si nada.", D, size=24), D, pad=28),
            card(label("Termina antes", D) + p("Si A termina en el medio, se avisa al scheduler y el destino deja de recalcular.", D, size=24), D, pad=28), gap=16, flex=1), gap=48),
    """Tiempo: 0:42–0:45 (3 min).
Idea clave: la migración es un protocolo distribuido entre router, scheduler, origen y destino, y está diseñado para que cualquier falla deje a la inferencia corriendo en algún lado.
Decir: recorrer los pasos de la figura 4. Lo importante es el paso 5: el origen no le devuelve los tokens al destino directamente sino al router, con una marca de "migrado"; el router, que es el que tiene la conexión con el cliente, actualiza su tabla de rutas y sigue mandando los tokens desde el destino (paso 7). El cliente no se entera.
Las fallas (§5.4): mientras el destino carga o recalcula, el origen sigue generando. Entonces si falla el destino, el origen simplemente continúa. Si falla el origen antes de migrar, se aborta y se descarga el modelo del destino; durante la migración, el destino descarta el KV cache a medio hacer. Y si la inferencia termina sola en el medio, el origen avisa y el destino deja de recalcular.
Conectar con la materia: es el patrón de las transacciones distribuidas de la primera parte de la materia: hasta el paso 5 nada es irreversible, y el punto de compromiso es uno solo.
Pasar a la siguiente: dónde guarda el scheduler lo que sabe.""",
    d.footer("Fu et al., OSDI 2024, §5.3, §5.4 y figura 4"), gap=40)

# 16 ─ estado
d.slide("estado", D, head("El estado del scheduler · §6 y §6.3", "Qué hay cargado dónde vive en un almacén de claves confiable", D) +
    row(card(label("Lo que guarda", D, color="acc") + p("Por servidor: qué GPUs están ocupadas, qué checkpoints hay en DRAM y en SSD, y su cola de carga.", D, size=28), D, accent=True, pad=36),
        card(label("Cuándo lo escribe", D) + p("Al asignar una carga, marca la GPU. Cuando el servidor confirma, actualiza el almacenamiento y recién ahí avisa al router.", D, size=28), D, pad=36),
        card(label("Si el scheduler se cae", D) + p("El nuevo lee el estado del almacén (etcd, ZooKeeper) y lo sincroniza con los servidores.", D, size=28), D, pad=36), gap=32) +
    p("Atiende miles de cargas por segundo en un servidor; repartir el scheduler queda como trabajo futuro.", D, size=28, color="muted"),
    """Tiempo: 0:45–0:48 (3 min).
Idea clave: el mapa de qué modelo está en qué nivel de qué servidor es estado compartido, y el scheduler lo guarda en un almacén consistente para sobrevivir a sus propias caídas.
Decir: el scheduler toma decisiones con un mapa: qué GPU está ocupada, qué checkpoint está en la DRAM o el SSD de cada servidor, qué cargas tiene encoladas cada uno. Ese mapa lo guarda en un almacén de claves confiable, etcd o ZooKeeper.
El orden de escritura importa: al asignar una carga, primero marca la GPU como ocupada en el almacén; cuando el servidor confirma que terminó, actualiza qué hay en su almacenamiento; y recién después le avisa al router que ya puede mandar pedidos ahí. Así el router nunca rutea a un modelo que todavía no está.
Si el scheduler se cae, el que lo reemplaza lee el último estado y lo sincroniza con los servidores.
Conectar con la materia: ZooKeeper y etcd son las clases de consenso. Y el mapa es una caché del estado real de los servidores: puede estar desactualizado (un checkpoint desalojado de la DRAM), y entonces la estimación falla; el scheduler corrige con lo que le devuelven los servidores. Es el mismo problema de información de carga vieja que vimos con el router de la Clase 19.
Pasar a la siguiente: la demo de ubicación.""",
    d.footer("Fu et al., OSDI 2024, §6 y §6.3"), gap=40)

# 17 ─ demo ubicación (claro)
d.slide("demoubica", L, head("Demo · arranque, tabla 2", "La localidad vale más que la migración; desalojar es peor que nada", L) +
    table(["Política", "Frío", "Cargas de DRAM", "de remoto", "Demora media", "p99"], [
        ["Al azar, cargador ingenuo", "62%", "33%", "34%", "24,5 s", "188 s"],
        ["Al azar", "53%", "31%", "35%", "7,5 s", "80 s"],
        ["Consciente de la localidad", "47%", "70%", "7%", "5,5 s", "83 s"],
        ["Localidad + desalojo (49)", "50%", "68%", "7%", "6,3 s", "102 s"],
        ["Localidad + migración (6)", "48%", "69%", "7%", "5,5 s", "83 s"],
    ], L, widths=[35, 10, 17, 13, 15, 10], right=(1, 2, 3, 4, 5), size=28, bold_rows=(2,)) +
    p("64 modelos, 3.996 pedidos en una hora, en ráfagas; 4 servidores de 4 GPUs. Demora: lo que tardó de más cada pedido.", L, size=24, color="muted"),
    """Tiempo: 0:48–0:52 (4 min, con demo).
Idea clave: lo que más pesa es el cargador; después, arrancar donde está el checkpoint. Con GPUs de sobra la migración casi no se activa, y desalojar empeora.
Demo: go run ./cmd/sim arranque, la segunda tabla. En el simulador Flota de GPUs, cambiar la política y mirar el color de las cargas: con "al azar" hay muchas desde remoto; con localidad, casi todas desde DRAM.
Leer la tabla: 64 modelos con popularidad de ley de potencia, una hora de pedidos en ráfagas, 16 GPUs, cada instancia atiende hasta 8 pedidos a la vez. El cargador ingenuo es lo peor de lejos: 24,5 s de demora media. Con el cargador en pipeline y al azar, 7,5 s. Eligiendo el servidor con el checkpoint más cerca, las cargas desde DRAM pasan de 31% a 70%, las de remoto de 35% a 7%, y la demora media baja a 5,5 s.
La cola (p99) la dominan las ráfagas: cuando un modelo recibe decenas de pedidos de golpe, las 16 GPUs no alcanzan, con cualquier política.
Desalojar (49 veces) empeora la media y la cola: cada inferencia cortada pierde lo que llevaba. La migración casi no se usa (6 veces) porque casi siempre hay una GPU libre en un servidor con el modelo en DRAM. Con 2 GPUs por servidor (-gpus 2), migra 27 veces y desaloja 114, y la diferencia entre las dos crece: 9,4 contra 14,6 s.
Lo honesto: en el paper la migración pesa más porque sus inferencias son más largas (ShareGPT) y las GPUs están llenas.
Pasar a la siguiente: los resultados del paper.""",
    d.footer("go run ./cmd/sim arranque · simulador Flota de GPUs", L, plain=True), gap=36)

# 18 ─ resultados (claro)
d.slide("resultados", L, col(eyebrow("Resultados · OSDI 2024, figura 10", L), h2("De minutos a segundos, con las mismas GPUs", L), gap=16) +
    row(big("0,8 s", "arranque medio de OPT-6.7B (GSM8K), contra 12,1 s de Ray Serve y 8,2 s con caché en SSD", L),
        big("28×", "con OPT-30B: 7,5 s contra 213 s y 199,2 s", L, color="blue"),
        big("89%", "de los pedidos de OPT-30B atendidos en menos de 300 s; Ray Serve con caché, 26%", L, color="warn"), gap=64) +
    p("KServe: 128 s hasta el primer token, 114 de ellos bajando el checkpoint por 1 Gb/s. En total, entre 10 y 200 veces menos latencia.", L, size=28),
    """Tiempo: 0:52–0:55 (3 min).
Idea clave: juntando cargador, localidad y migración, el arranque pasa de minutos a segundos en el mismo clúster.
Decir: el clúster son 4 servidores con 4 A40 cada uno, 512 GB de DRAM, un NVMe de 2 TB y Ethernet de 10 Gb/s; la carga, la traza de Azure con ráfagas. Los competidores: Ray Serve extendido para serverless, Ray Serve con caché LRU en SSD, y KServe. Todos con keep-alive igual a su tiempo de carga y un pedido a la vez por instancia.
Con OPT-6.7B y preguntas de GSM8K, ServerlessLLM arranca en 0,8 s en promedio; Ray Serve en 12,1 y con caché en 8,2. Con una red 10 veces más rápida, Ray Serve bajaría a 3,8 s: todavía 4,7 veces más lento. Con OPT-30B, 7,5 contra 213 y 199,2 segundos. Y con un timeout de 300 s, atiende el 89% de los pedidos contra 26%.
KServe: 128 s hasta el primer token, de los cuales 114 son bajar el checkpoint de S3 por una red de 1 Gb/s; con las mismas mejoras que Ray Serve, 28 s. Sólo ServerlessLLM baja de un segundo.
Y en el scheduler (figura 8), con carga alta, el p99 de ServerlessLLM es 1,27 veces mejor que el de Shepherd* (que desaloja) y 1,95 veces mejor que el serverless al azar.
Pasar a la siguiente: cuánto dejar cargado un modelo.""",
    d.footer("Fu et al., OSDI 2024, §7.3 y §7.4", L), gap=56)

# 19 ─ keep-alive
d.slide("keepalive", D, head("Keep-alive", "Cuánto tiempo dejar cargado un modelo que nadie usa", D) +
    row(col(p("Es un TTL: más largo, menos arranques en frío y más GPUs ocupadas sin trabajo. El dilema clásico de serverless.", D, size=28),
            p("Si arrancar desde DRAM cuesta medio segundo, el TTL deja de importar: se puede liberar la GPU enseguida.", D, size=28, color="ink"),
            p("En la evaluación del paper, el keep-alive de cada sistema es igual a su tiempo de carga.", D, size=24, color="muted"), gap=24, width=640),
        col(table(["Keep-alive", "Clásico: frío", "demora", "Con caché: frío", "demora"], [
            ["0 s", "86%", "60 s", "68%", "5,6 s"], ["10 s", "79%", "56 s", "60%", "5,5 s"],
            ["60 s", "61%", "47 s", "42%", "5,4 s"], ["600 s", "58%", "47 s", "35%", "5,4 s"],
        ], D, widths=[22, 20, 17, 24, 17], right=(1, 2, 3, 4), size=26),
            p("Clásico: descarga cada vez, cargador ingenuo, al azar · go run ./cmd/sim arranque", D, size=22, color="muted"), gap=16, flex=1), gap=48, align="center"),
    """Tiempo: 0:55–0:59 (4 min, con demo).
Idea clave: el keep-alive es el TTL de un caché de modelos en GPU; su valor depende de cuánto cuesta un fallo, y si los checkpoints están cerca, el fallo es barato.
Decir: después de atender el último pedido, ¿cuánto tiempo dejamos el modelo en la GPU? Si lo sacamos enseguida, el próximo pedido arranca en frío. Si lo dejamos mucho, la GPU está ocupada sin hacer nada, y en la nube eso es plata. Es el dilema que estudia "Serverless in the Wild" con la traza de Azure, donde proponen un keep-alive adaptativo por aplicación en lugar de uno fijo.
La demo, tabla 3: a la izquierda un serverless clásico que descarga el checkpoint cada vez. Pasar de 0 a 600 segundos de keep-alive baja los arranques en frío de 86% a 58% y la demora media de 60 a 47 s; a cambio, más de la mitad del tiempo de GPU es modelos cargados sin pedidos. A la derecha, con DRAM y SSD como caché y el cargador en pipeline: los arranques en frío bajan igual, pero la demora casi no cambia, porque un arranque desde DRAM cuesta medio segundo.
Conclusión: con el caché multinivel un fallo es barato, así que un keep-alive largo deja de pagar: con 600 s la GPU queda ociosa un 82% del tiempo para ganar 0,2 s. Se puede liberar la GPU enseguida y dejar el modelo en la DRAM, que es mucho más barata.
Conectar con la materia: en la flota de la demo, un modelo ocioso se desaloja si hace falta su GPU, como una entrada de un caché LRU. TTL y LRU juntos, igual que en un caché web.
Pasar a la siguiente: la segunda parte, los fine-tunes.""",
    d.footer("Shahrad et al., ATC 2020 · Fu et al., OSDI 2024, §7.4"), gap=40)

# 20 ─ lora
d.slide("lora", D, head("Muchos fine-tunes · LoRA", "Un fine-tune es el mismo modelo más una corrección chica", D) +
    row(col(p("W′ = W + A·B", D, size=72, color="acc", mono=True, weight=600, lh=1),
            p("W es h × d. A es h × r y B es r × d, con r mucho menor: 8, 16, 64. Se entrenan sólo A y B, en las proyecciones de query, key, value y salida.", D, size=28), gap=24, flex=1),
        col(big("10.000×", "menos parámetros para entrenar, con calidad comparable", D, size=80),
            big("cientos de miles", "de adaptadores LoRA publicados", D, size=56, color="blue"), gap=32, width=640), gap=56, align="center"),
    """Tiempo: 0:59–1:02 (3 min).
Idea clave: si los modelos de la flota son fine-tunes de un mismo modelo base, no hace falta cargar modelos enteros: comparten casi todos los pesos.
Decir: hasta acá cada modelo era distinto. Pero en una plataforma de fine-tuning, la mayoría son versiones de unos pocos modelos base, ajustadas con LoRA (Hu et al., 2021): en lugar de cambiar la matriz W, se le suma un producto A·B de rango r chico. Para una matriz de 4096 × 4096 y rango 8, son 65 mil parámetros en lugar de 16 millones.
El paper de S-LoRA lo resume: LoRA reduce los parámetros entrenables en órdenes de magnitud —10.000 veces, por ejemplo— con calidad comparable a ajustar todo, y la comunidad publicó cientos de miles de adaptadores. Se aplica, típicamente, a las proyecciones de query, key, value y salida de la atención.
Un adaptador de un modelo de 7B pesa decenas de MB; el modelo base, 13 GB.
Preguntar a la clase: si tengo 1.000 adaptadores del mismo modelo base, ¿cuántas copias del modelo base necesito en la GPU? (Una sola, si se hace bien.)
Pasar a la siguiente: cómo servirlos.""",
    d.footer("Sheng et al., S-LoRA, MLSys 2024, §2 · Hu et al., LoRA, 2021"), gap=40, section="Muchos fine-tunes: LoRA, lote mixto y paginación unificada.")

# 21 ─ fusionar vs mixto
d.slide("mixto", D, head("Fusionar o calcular aparte · S-LoRA §4.1", "x·W para todo el lote, x·A·B para cada uno", D) +
    row(card(label("Fusionar: W + A·B", D, color="warn") + p("Cero costo por token. Pero un lote es de un solo adaptador, y cambiar es restar uno y sumar otro con la GPU parada. Con una copia por adaptador, en una A100 de 80 GB entran menos de 5.", D, size=28), D, pad=36),
        card(label("Calcular aparte: x·W + x·A·B", D, color="acc") + p("x·W, que es lo caro, en un solo lote con todos los pedidos. x·A·B, que es chico, con kernels que juntan adaptadores de distinto rango sin relleno.", D, size=28), D, accent=True, pad=36), gap=32) +
    p("Lo que se gana en batching de x·W es mucho más que lo que cuesta x·A·B. Punica: 12× más throughput, sumando unos 2 ms por token.", D, size=28),
    """Tiempo: 1:02–1:05 (3 min).
Idea clave: si no se fusiona el adaptador, todos los pedidos comparten la parte cara del cálculo y entran en el mismo lote, sea cual sea su adaptador.
Decir: el paper de LoRA propone fusionar: sumar A·B a W y servir el modelo resultante, que no cuesta nada extra por token. Para muchos adaptadores, sumar y restar sobre la marcha. El problema: el lote sólo puede tener pedidos de un adaptador. Y la alternativa ingenua, una copia fusionada por adaptador, es lo que S-LoRA llama vLLM-packed: en una A100 de 80 GB entran menos de 5.
S-LoRA hace lo contrario: no fusiona. Calcula x·W para todo el lote con una multiplicación grande (GEMM), y x·A·B aparte para cada pedido con su adaptador. Es más cuenta, pero x·A·B es mucho más chica que x·W (el rango es chico), y lo que se gana en batching de la parte cara es mucho más.
El detalle técnico son los kernels: los adaptadores tienen distinto rango y los pedidos distinto largo, así que un batched GEMM normal necesita rellenar con ceros. Usan kernels propios que juntan los pesos desde memoria no contigua: MBGMM para prefill, MBGMV para decode; este último, una extensión del kernel de Punica.
Punica (Chen et al., 2023), trabajo concurrente con la misma idea, reporta 12 veces más throughput que los sistemas de entonces, sumando unos 2 ms de latencia por token. Su kernel se llama SGMV.
Conectar con la Clase 18: es batching otra vez. El decode está limitado por leer los pesos, así que cuantos más pedidos compartan una lectura de W, mejor.
Pasar a la siguiente: dónde viven los adaptadores.""",
    d.footer("Sheng et al., MLSys 2024, §4 y §5.3 · Chen et al., Punica, 2023"), gap=36)

# 22 ─ paginación unificada
def pool(t=D):
    cells = ("kkkkaakkkkkkbbbbkkk.kkaakk..kkkkcccckkkk..kkkkkkaakk...kk" * 2)[:100]
    col_ = {"k": t["blue"], "a": t["acc"], "b": t["acc"], "c": t["warn"], ".": t["fill"]}
    out = "".join(f'<div style="width:40px;height:40px;border-radius:6px;background:{col_[c]}"></div>' for c in cells)
    return f'<div style="display:grid;grid-template-columns:repeat(20,40px);gap:6px">{out}</div>'
d.slide("paginas", D, head("Unified Paging · S-LoRA §5", "Adaptadores y KV cache comparten un mismo pool de páginas", D) +
    row(col(pool(), row(chip("KV cache: (S, H)", D, color="blue"), chip("adaptador: (R, H)", D, color="acc"), chip("libre", D, color="fill", fg=D["ink"]), gap=12), gap=20, width=920),
        col(p("Un KV cache de S tokens ocupa S vectores de tamaño H; un adaptador de rango R, R vectores de tamaño H. Mismo H: misma página.", D, size=26),
            p("Todos los adaptadores viven en la memoria del host; se traen a la GPU los del lote, y mientras corre uno se adelantan los del próximo.", D, size=26), gap=24, flex=1), gap=48, align="center"),
    """Tiempo: 1:05–1:09 (4 min).
Idea clave: los adaptadores en la GPU se comportan como el KV cache —aparecen y desaparecen, tienen tamaños distintos—, así que se paginan juntos, como en la Clase 18.
Decir: S-LoRA guarda todos los adaptadores en la memoria del host, que es grande, y trae a la GPU sólo los que usa el lote actual. Como el lote está acotado por la memoria de la GPU, los adaptadores activos también.
Pero traer y sacar adaptadores de distinto tamaño fragmenta la memoria, igual que el KV cache antes de PagedAttention. La observación: un KV cache de una capa para S tokens es una matriz S × H; un peso LoRA de rango R es R × H. Comparten H. Entonces una página es un vector de H, el KV cache de S tokens ocupa S páginas y un adaptador de rango R ocupa R páginas, mezclados en un mismo pool que usa toda la memoria que no son los pesos base. Lo llaman Unified Paging: PagedAttention generalizado.
El dibujo: azul, KV cache de pedidos; verde, adaptadores; naranja, uno de rango más alto. Todos intercalados, sin huecos grandes.
Y para que traer adaptadores no frene, mientras corre una iteración miran la cola de espera y adelantan los adaptadores que va a necesitar el próximo lote.
Conectar con la materia: es memoria virtual otra vez, y la memoria de la GPU funciona como caché de la del host. Los adaptadores populares se quedan; los demás se desalojan cuando hace falta lugar para KV cache.
Pasar a la siguiente: los resultados.""",
    d.footer("Sheng et al., MLSys 2024, §5.1 y §5.2"), gap=40)

# 23 ─ resultados S-LoRA (claro)
d.slide("slorares", L, head("S-LoRA · tabla 3, Llama-7B en una A100 de 80 GB", "Miles de adaptadores con el throughput de cinco", L) +
    row(col(table(["Adaptadores", "S-LoRA", "vLLM-packed", "PEFT"], [
        ["5", "8,05", "2,04", "0,88"], ["100", "7,99", "sin memoria", "0,25"], ["1.000", "7,64", "sin memoria", "—"], ["2.000", "7,61", "sin memoria", "—"],
    ], L, widths=[25, 22, 30, 23], right=(1, 2, 3), size=30, bold_rows=(3,)),
        p("Pedidos por segundo, adaptadores de rango 8 (configuración S1).", L, size=24, color="muted"), gap=16, flex=1),
        col(big("4×", "más throughput que vLLM con una copia por adaptador", L, size=80),
            big("30×", "más que Hugging Face PEFT", L, size=80, color="blue"), gap=40, width=520), gap=64, align="center"),
    """Tiempo: 1:09–1:12 (3 min).
Idea clave: con lote mixto y paginación unificada, pasar de 5 a 2.000 adaptadores casi no cambia el throughput.
Decir: la tabla 3, con Llama-7B en una A100 de 80 GB y adaptadores de rango 8. Con 5 adaptadores, S-LoRA atiende 8,05 pedidos por segundo; vLLM-packed, una copia fusionada por adaptador, 2,04, porque no puede juntar pedidos de distintos adaptadores; PEFT, la librería de Hugging Face, que cambia de adaptador entre lotes, 0,88.
Con 100 adaptadores vLLM-packed ya no entra en memoria y PEFT cae a 0,25. S-LoRA sigue en 7,99, y con 2.000 adaptadores, 7,61. Ni siquiera evaluaron PEFT con más de 100.
¿Por qué casi no baja? Porque el costo extra depende de los adaptadores que hay en un lote, no de los que existen, y los de un lote están acotados por el tamaño del lote. Pasados unos 100 adaptadores (figura 5), el throughput deja de bajar.
Y un dato honesto de la ablación (figura 9): con un solo adaptador, fusionar es mejor que calcular aparte; desde tres, gana calcular aparte.
Pasar a la siguiente: la demo.""",
    d.footer("Sheng et al., MLSys 2024, §7.2, tabla 3 y figura 9", L), gap=40)

# 24 ─ demo lora (claro)
d.slide("demolora", L, head("Demo · lora", "Fusionar gana con un adaptador; desde dos, el lote mixto", L) +
    row(col(table(["Adaptadores", "Fusionado", "Lote mixto", "Por lote"], [
        ["1", "21,0", "18,9", "1"], ["2", "15,7", "18,9", "2"], ["5", "15,6", "18,8", "5"],
        ["100", "10,4", "18,0", "47"], ["1.000", "3,3", "17,2", "78"], ["2.000", "2,2", "16,9", "85"],
    ], L, widths=[28, 24, 24, 24], right=(1, 2, 3), size=28),
        p("Capacidad en pedidos por segundo, con carga de sobra.", L, size=24, color="muted"), gap=16, flex=1),
        col(card(label("Con 200 adaptadores, a 10 pedidos/s", L) +
                 p("Fusionado: atiende 3 por segundo y la cola crece sin límite. Lote mixto: los 10, con TTFT p99 de 44 ms.", L, size=26) +
                 p("Con caché en el pool: 200 cargas de adaptadores en vez de 1.545.", L, size=26), L, pad=36), gap=16, width=620), gap=48, align="center"),
    """Tiempo: 1:12–1:15 (3 min, con demo).
Idea clave: la demo reproduce la forma de S-LoRA: el fusionado sólo gana con un adaptador, y el lote mixto casi no pierde capacidad aunque haya miles.
Demo: go run ./cmd/sim lora.
Leer la tabla de la izquierda: un modelo de 7B, adaptadores de rangos 64 a 8, con la carga de S-LoRA (popularidad en ley de potencia, prompts y respuestas de 8 a 512 tokens) y más pedidos de los que se pueden atender, para medir capacidad. Con un adaptador, fusionado atiende 21 pedidos por segundo y el lote mixto 18,9: el lote mixto paga x·A·B en cada token. Con dos ya gana el mixto, y la distancia crece: con 2.000 adaptadores el fusionado atiende 2,2 por segundo, porque arma lotes de un solo adaptador y cambia todo el tiempo; el mixto, 16,9.
La última columna explica por qué: en un lote mixto entran a lo sumo unos 85 adaptadores distintos, haya 100 o 2.000.
A la derecha, con 200 adaptadores a 10 pedidos por segundo: el fusionado no da abasto. Y dejar los adaptadores en el pool mientras no haga falta el lugar los trae una sola vez: 200 cargas contra 1.545. Pero traer un adaptador de 7B desde el host cuesta menos de un milisegundo, así que la latencia casi no cambia: el caché importa cuando el adaptador está más lejos.
Pasar a la siguiente: la lente de sistemas distribuidos.""",
    d.footer("go run ./cmd/sim lora", L, plain=True), gap=40)

# 25 ─ lente
d.slide("lente", D, head("La lente de la materia", "Todo lo de hoy son ideas viejas con un estado nuevo", D) +
    table(["Hoy", "Ya lo vimos como"], [
        ["Checkpoints en DRAM, SSD y remoto", "jerarquía de caché; LRU por nivel"],
        ["Keep-alive", "TTL de un caché; el costo de un fallo decide el TTL"],
        ["Arrancar donde está el checkpoint", "llevar el cómputo a los datos, como MapReduce"],
        ["El mapa de qué está dónde", "estado en etcd o ZooKeeper: consenso"],
        ["Migración por rondas", "migración en vivo de VMs por pre-copia"],
        ["Rechazar si no llega al SLO", "control de admisión (S-LoRA, Mooncake)"],
        ["Flota compartida", "multiplexación estadística"],
    ], D, widths=[44, 56], size=28),
    """Tiempo: 1:15–1:19 (4 min).
Idea clave: ninguna de las técnicas de hoy es nueva; lo nuevo es el tamaño del estado (decenas de GB) y la duración impredecible de cada inferencia.
Decir: recorrer la tabla. Los checkpoints en niveles son una jerarquía de caché como la de un procesador o una CDN, cada nivel con su LRU. El keep-alive es un TTL, y el TTL correcto depende de cuánto cuesta un fallo: si el fallo es barato (DRAM), TTL corto. Arrancar el modelo donde ya está el checkpoint es lo mismo que el scheduler de MapReduce: mandar el cómputo a donde están los datos.
El mapa de qué está dónde es estado compartido que se guarda con consenso. La migración en rondas es la pre-copia de las máquinas virtuales: copiar mientras corre, en rondas que achican lo pendiente, y parar al final para lo último. Con un truco: en lugar de copiar el estado derivado (KV cache), se copia el que lo genera (los tokens) y se recalcula.
S-LoRA también tiene control de admisión: su "early abort" descarta los pedidos que ya no van a llegar al SLO de 6 segundos al primer token, como el rechazo temprano de Mooncake. Y la flota compartida es multiplexación estadística, la idea detrás de la conmutación de paquetes y de la nube.
Preguntar a la clase: ¿qué consistencia necesita el mapa del scheduler? (Si cree que un checkpoint está en DRAM y no está, la carga tarda más pero es correcta: alcanza con consistencia eventual para la eficiencia; lo que no puede pasar es que el router mande pedidos a una GPU que todavía no tiene el modelo, y por eso ese aviso va después de la confirmación.)
Pasar a la siguiente: qué llevarse.""",
    d.footer("Fu et al., OSDI 2024 · Sheng et al., MLSys 2024, §4.3"), gap=40, section="Cierre: la lente de la materia, el repositorio y preguntas.")

# 26 ─ takeaways
d.slide("takeaways", D, head("Para llevarse", "Compartir GPUs es un problema de caché y de ubicación", D) +
    row(card(label("Acercar", D, color="acc") + p("Los checkpoints en DRAM y SSD del servidor, con un formato y un cargador que saturan cada nivel.", D, size=28), D),
        card(label("Ubicar", D, color="acc") + p("Arrancar donde el checkpoint está más cerca, y si ese servidor está ocupado, mudar —no cortar— la inferencia.", D, size=28), D),
        card(label("Compartir", D, color="acc") + p("Los fine-tunes, en el mismo lote: x·W una vez para todos, x·A·B para cada uno, todo en un pool paginado.", D, size=28), D), gap=32),
    """Tiempo: 1:19–1:22 (3 min).
Idea clave: servir muchos modelos en pocas GPUs se reduce a tres decisiones: dónde guardar los pesos, dónde arrancar cada modelo y qué compartir.
Decir: tres verbos. Acercar: el arranque en frío lo dominan la distancia y el cargador; con el checkpoint en la DRAM del servidor y un cargador en pipeline, un modelo de 13 GB está en la GPU en medio segundo, y un keep-alive largo deja de pagar. Ubicar: el scheduler estima el arranque en cada servidor (cola más tamaño sobre ancho de banda) y elige; cuando el mejor está ocupado, migra la inferencia mandando los tokens y recalculando el KV cache, en lugar de cortarla. Compartir: si los modelos son fine-tunes de uno mismo, no se cargan enteros: se batchea la parte común y se pagina todo junto.
Conectar con la serie: la Clase 21 sigue con cómo repartir un clúster de GPUs entre trabajos distintos, y la 22 con entrenar con lo que se sirve.
Pasar a la siguiente: el repositorio.""",
    d.footer())

# 27 ─ demo
cmds = [("go run ./cmd/sim arranque", "cargas, ubicación y keep-alive"), ("go run ./cmd/sim arranque -gpus 2", "con pocas GPUs: migrar o desalojar"),
        ("go run ./cmd/sim lora", "fusionado contra lote mixto"), ("go test ./flota", "los tests de lo que dijimos")]
lines = "".join(f'<div style="display:flex;gap:48px">{p(c, D, size=32, color="ink", mono=True, width=760)}{p(w, D, size=32, color="muted")}</div>' for c, w in cmds)
d.slide("demo", D, head("El repositorio", "servir-tokens, ahora con una flota", D) +
    f'<div style="display:flex;flex-direction:column;gap:20px;background:#0A0C0F;border:1px solid {D["line"]};border-radius:16px;padding:48px">{lines}</div>' +
    p("Paquete nuevo: flota (cargas por nivel, keep-alive, ubicación y migración; adaptadores LoRA). Y el simulador Flota de GPUs.", D, size=28),
    """Tiempo: 1:22–1:26 (4 min).
Idea clave: el paquete flota simula una plataforma serverless en tiempo virtual, con eventos: llegadas en ráfagas, cargas desde cada nivel, instancias con keep-alive, migración y desalojo; y aparte, un servidor con muchos adaptadores LoRA.
Decir: los tests verifican las afirmaciones de la clase: cada nivel es más lento que el anterior y el cargador en pipeline es más de 3 veces más rápido desde SSD; arrancar donde está el checkpoint carga más desde DRAM y baja la demora; migrar deja la inferencia parada milisegundos mientras que desalojar le hace perder lo que llevaba y sube la demora; mandar los tokens es miles de veces menos que mandar el KV cache; con descarga cada vez el keep-alive baja mucho la demora y con caché local casi nada; una GPU por modelo usa cuatro veces más GPUs mucho menos ocupadas; fusionar gana sólo con un adaptador y el lote mixto escala hasta 2.000.
El simulador Flota de GPUs muestra las 16 GPUs en vivo, con el color de cada modelo y de dónde salió cada carga; se puede cambiar la política, el keep-alive y el cargador.
Los ejercicios del README: ubicación inicial según popularidad, keep-alive adaptativo y prefetch de adaptadores desde disco.
Pasar a la siguiente: preguntas.""",
    d.footer("go test ./..."))

# 28 ─ preguntas
qs = ["¿Por qué con un cargador en pipeline el tiempo de carga es el del nivel más lento y no la suma?",
      "¿Qué estima el scheduler de ServerlessLLM para elegir servidor? ¿De dónde saca cada término?",
      "¿Por qué ServerlessLLM manda los tokens y Mooncake el KV cache?",
      "Si un arranque desde DRAM cuesta medio segundo, ¿qué pasa con el keep-alive óptimo?",
      "¿Por qué el throughput de S-LoRA casi no cae de 100 a 2.000 adaptadores?"]
d.slide("preguntas", D, head("Para repasar", "Preguntas y lecturas", D) +
    row(f'<ol style="flex:1.4;font-size:28px;line-height:1.4;color:{D["ink"]}">' + "".join(f"<li>{q}</li>" for q in qs) + "</ol>",
        card(h3("Lecturas", D) + p("<b>Fu et al.</b>, ServerlessLLM, OSDI 2024.", D, size=24, color="ink") +
             p("Sheng et al., S-LoRA, MLSys 2024. Chen et al., Punica, 2023.", D, size=24) +
             p("Li et al., AlpaServe, OSDI 2023. Shahrad et al., Serverless in the Wild, ATC 2020.", D, size=24), D, flex=1, pad=40, gap=20), gap=64),
    """Tiempo: 1:26–1:30 (4 min).
Idea clave: cerrar con las preguntas de repaso. Las respuestas cortas, para tener a mano:
1. Porque los niveles trabajan a la vez sobre pedazos distintos: mientras un pedazo sube a la GPU, el siguiente pasa del SSD a la DRAM. El flujo lo limita el caño más angosto, como cualquier pipeline.
2. q + n/b: la cola de carga del servidor (las cargas van de a una, así que es la suma de las estimaciones anteriores), el tamaño del checkpoint y el ancho de banda del nivel donde está en ese servidor, corregido con lo que va midiendo. Para migrar, a·(tokens de entrada + generados) + b.
3. Porque depende de la red contra el cómputo. Mooncake tiene RDMA de 800 Gb/s y prompts larguísimos, así que traer es más barato que recalcular; ServerlessLLM tiene Ethernet de 10 Gb/s, el KV cache son gigabytes y el destino está libre para recalcular, que es un prefill.
4. Que importa mucho menos: un fallo es barato, así que conviene liberar la GPU enseguida y dejar el modelo en DRAM. El TTL óptimo crece con el costo del fallo.
5. Porque el costo extra depende de los adaptadores distintos de un lote, que están acotados por el tamaño del lote, no de cuántos adaptadores existen. Y todos viven en el host; en la GPU sólo los activos.
Lecturas: ServerlessLLM es el paper de la clase y S-LoRA el compañero. Complementarias: Punica, AlpaServe y la caracterización de Azure. Los apuntes tienen más preguntas.""",
    d.footer())

d.write_index("Clase 20 — Muchos modelos, pocas GPUs")
print(len(d.order), "slides")
