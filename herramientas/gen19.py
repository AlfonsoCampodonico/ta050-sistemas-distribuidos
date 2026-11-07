import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidekit import *
from cover import cover

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "clase-19")
d = Deck(ROOT, 19, "Del motor al clúster")
D, L = DARK, LIGHT

# 1 ─ portada
d.raw("cover", cover(19, "Del motor al clúster",
    "Cómo se sirve un modelo hoy: el KV cache compartido entre pedidos, un router que sabe dónde está cada prefijo, y el prefill separado del decode",
    "Paper: Qin et al., <i>Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot</i>, FAST 2025",
    """Tiempo: 0:00–0:02 (2 min).
Idea clave: la clase pasada fue una réplica por dentro; hoy es un servicio con muchas réplicas, y el KV cache pasa de ser un detalle de cada réplica al centro del diseño.
Decir: en la Clase 18 vimos vLLM: cómo una GPU arma lotes y administra el KV cache de los pedidos en curso. Desde 2024 casi todo lo nuevo en serving pasa afuera de esa réplica: guardar el KV cache entre pedidos para no recalcular, rutear cada pedido a donde ya está su prefijo, y separar el prefill del decode en máquinas distintas.
El paper de hoy es Mooncake, el sistema que sirve a Kimi, el chatbot de Moonshot AI. Ganó el premio al mejor paper en FAST 2025, una conferencia de almacenamiento: la idea central es cambiar cómputo por almacenamiento.
Es la última de las cuatro clases: 16 y 17 sobre entrenar, 18 y 19 sobre servir.
Pasar a la siguiente: qué cambia entre una réplica y un servicio."""), section="De una réplica a un servicio: la forma de la carga y cuánto se repite.")

# 2 ─ mapa
d.slide("mapa", D, head("Dónde estamos", "De una réplica a un servicio", D) +
    table(["", "Clase 18: una réplica", "Hoy: un clúster"], [
        ["KV cache", "de cada pedido, mientras dura", "compartido entre pedidos, en varios niveles"],
        ["Router", "da igual", "sabe dónde está cada prefijo"],
        ["Prefill y decode", "en la misma iteración", "en pedazos, o en máquinas distintas"],
        ["El modelo", "denso, en una réplica", "expertos repartidos en muchas GPUs"],
    ], D, widths=[22, 36, 42], size=32),
    """Tiempo: 0:02–0:04 (2 min).
Idea clave: el motor de la Clase 18 sigue adentro de cada réplica; lo nuevo es todo lo que lo rodea.
Decir: recorrer las cuatro filas, que son las cuatro partes de la clase. Uno: el KV cache deja de tirarse cuando termina el pedido; se guarda y se reusa, en la GPU, en la memoria de las CPUs del clúster o en disco. Dos: con eso, el router ya no puede repartir en ronda: tiene que mandar cada pedido a donde está su prefijo, sin cargar de más a nadie. Tres: el prefill y el decode tienen necesidades opuestas, y se separan, en el tiempo o en el espacio. Cuatro: los modelos de frontera son de expertos, y servirlos es un problema de paralelismo como el de la Clase 17.
Pasar a la siguiente: cómo es la carga que hay que servir.""",
    d.footer())

# 3 ─ carga (claro)
d.slide("carga", L, col(eyebrow("La carga de un chatbot · Mooncake §4.2", L), h2("Prompts largos, respuestas cortas, y casi todo repetido", L), gap=16) +
    row(big("7.590", "tokens de prompt, en promedio, en la traza de Kimi", L), big("182", "tokens de respuesta", L, color="blue"), big("≈ 42×", "más tokens de entrada que de salida", L, color="warn"), gap=96) +
    p("Cada turno de una conversación manda la conversación entera otra vez: el prompt de un turno es el principio del siguiente.", L),
    """Tiempo: 0:04–0:08 (4 min).
Idea clave: en un chatbot real, el prefill domina: los prompts son decenas de veces más largos que las respuestas, y casi todo el prompt ya se vio en el turno anterior.
Decir: Mooncake publicó una traza de una hora de pedidos reales de Kimi. El prompt promedio tiene 7.590 tokens y la respuesta 182: hay unas 42 veces más tokens de entrada que de salida. Comparar con la Clase 18, donde pensábamos en prompts de cientos de tokens.
¿Por qué tan largos? Porque el modelo no tiene memoria entre pedidos: cada turno de una conversación manda el prompt de sistema, toda la historia y el mensaje nuevo. Más documentos, herramientas, contexto de un agente. Y cada turno es el prefijo del siguiente.
La carga del repositorio imita eso: conversaciones de varios turnos con un prompt de sistema de 2.000 tokens por app; el prompt promedio da unos 3.900 tokens y la respuesta 178.
Preguntar a la clase: si el turno 5 de una conversación ya calculó el KV cache de los turnos 1 a 4, ¿qué habría que hacer con él? (Guardarlo, y que el turno 6 caiga donde está.)
Pasar a la siguiente: cuánto se repite en producción.""",
    d.footer("Qin et al., Mooncake, §4.2 (arXiv) · go run ./cmd/sim router", L), gap=64)

# 4 ─ reuso
d.slide("reuso", D, head("Cuánto se repite", "La mitad del prompt ya se calculó antes", D) +
    row(card(label("DeepSeek · un día de febrero de 2025", D) + p("56,3%", D, size=96, color="acc", mono=True, weight=600, lh=1) +
             p("de 608 mil millones de tokens de entrada salieron del KV cache guardado en disco", D, size=28), D, accent=True, pad=40),
        card(label("Kimi · la traza de Mooncake", D) + p("50%", D, size=96, color="warn", mono=True, weight=600, lh=1) +
             p("del KV cache se podría reusar con capacidad infinita; más de la mitad de los bloques no se vuelve a usar nunca", D, size=28), D, pad=40), gap=32),
    """Tiempo: 0:08–0:11 (4 min).
Idea clave: en producción, más o menos la mitad de los tokens de entrada no hace falta calcularlos: ya se calcularon para un pedido anterior.
Decir: DeepSeek publicó las estadísticas de un día de su servicio, del 27 al 28 de febrero de 2025: 608 mil millones de tokens de entrada, de los cuales 342 mil millones, el 56,3%, salieron de su caché de KV en disco. Más de la mitad del prefill evitado.
Mooncake analiza la traza de Kimi: con capacidad infinita, se podría reusar como mucho un 50% del KV cache. Y la popularidad es muy despareja: más de la mitad de los bloques no se vuelve a usar, y algunos —los prompts de sistema, documentos populares— se usan decenas de miles de veces. Con 1.000 bloques de caché la tasa de aciertos es 30%; con 50.000, 50%.
Conectar con la materia: es la Clase 12, consistencia en caches, con dos diferencias amables: el contenido es inmutable (el KV cache de un prefijo dado es siempre el mismo) y fallar es barato (se recalcula). Lo difícil es dónde ponerlo y cómo encontrarlo.
Pasar a la siguiente: cómo se identifica un prefijo.""",
    d.footer("DeepSeek, Open Infra Index, 2025 · Qin et al., Mooncake, §4 y §9"), gap=48)

# 5 ─ hash
def chainrow(labels, colors, t=D):
    out = []
    for i, (lab, c) in enumerate(zip(labels, colors)):
        out.append(f'<div style="width:220px;height:64px;border-radius:10px;background:{c};display:flex;align-items:center;justify-content:center">'
                   f'<p style="font-family:{MONO};font-size:24px;font-weight:600;color:{t["onfill"]}">{lab}</p></div>')
        if i < len(labels) - 1:
            out.append(f'<x-shape kind="arrow-right" style="width:40px;height:20px;background:{t["muted"]}"></x-shape>')
    return f'<div style="display:flex;align-items:center;gap:12px">{"".join(out)}</div>'
A, B, C = D["acc"], D["blue"], D["warn"]
d.slide("hash", D, head("Caché de prefijos · Mooncake §4.1", "Cada bloque se identifica por su contenido y por todo lo anterior", D) +
    col(label("Conversación 1, turno 3", D), chainrow(["h₀ sistema", "h₁ sistema", "h₂ conv. 1", "h₃ conv. 1"], [A, A, B, B]), gap=12) +
    col(label("Conversación 2, turno 1: misma app", D), chainrow(["h₀ sistema", "h₁ sistema", "h₂ conv. 2", "—"], [A, A, C, D["fill"]]), gap=12) +
    p("hᵢ = hash(hᵢ₋₁, tokens del bloque i). Dos prompts comparten el bloque i sólo si comparten todo lo anterior: se buscan en orden y se frena en el primero que falta.", D, size=28),
    """Tiempo: 0:11–0:15 (4 min).
Idea clave: el KV cache de un token depende de todos los tokens anteriores, así que un bloque sólo se puede reusar si todo su prefijo es igual; un hash encadenado captura exactamente eso.
Decir: el prompt se parte en bloques de tokens (Mooncake usa 512; el repositorio, 64). El identificador de cada bloque es el hash de sus tokens y del identificador del bloque anterior. Si dos prompts tienen el mismo identificador en el bloque 3, tienen los mismos tokens en los bloques 0 a 3.
Recorrer el dibujo: dos conversaciones de la misma app comparten el prompt de sistema, los bloques 0 y 1. Desde el bloque 2 cada una tiene su historia, así que los hashes divergen, y ya nunca vuelven a coincidir aunque después aparezca el mismo texto.
Para buscar: se recorren los hashes del prompt nuevo en orden y se frena en el primero que no está. Es lo mismo que recorrer un árbol de prefijos; SGLang lo implementa literalmente como un radix tree (RadixAttention). Para desalojar, LRU: los bloques del prompt de sistema se tocan todo el tiempo y quedan, los de conversaciones abandonadas se van.
Conectar con la Clase 18: es el mismo truco que los bloques de PagedAttention, pero entre pedidos distintos en lugar de entre muestras de un mismo pedido.
Pasar a la siguiente: dónde se guarda.""",
    d.footer("Qin et al., FAST 2025, §4.1 · Zheng et al., SGLang, 2024"), gap=36, section="El caché de prefijos y dónde vive.")

# 6 ─ niveles
def lvl(name, size_, speed, t=D, w=900, hi=False):
    return (f'<div style="display:flex;gap:32px;align-items:center"><div style="width:{w}px;height:64px;border-radius:10px;'
            f'background:{t["acc"] if hi else t["fill"]};display:flex;align-items:center;justify-content:space-between;padding:0 24px">'
            f'<p style="font-family:{MONO};font-size:24px;font-weight:600;color:{t["onfill"] if hi else t["ink"]}">{name}</p>'
            f'<p style="font-family:{MONO};font-size:24px;color:{t["onfill"] if hi else t["muted"]}">{size_}</p></div>{p(speed, t, size=24)}</div>')
d.slide("niveles", D, head("Cambiar cómputo por almacenamiento", "El KV cache se guarda en niveles, como cualquier caché", D) +
    col(lvl("HBM de la GPU", "decenas de GB", "lo que está usando el lote", w=420, hi=True),
        lvl("DRAM de las CPUs del clúster", "TB", "el pool de Mooncake, por RDMA", w=640),
        lvl("SSD", "decenas de TB", "el caché en disco de DeepSeek", w=860),
        lvl("almacenamiento remoto", "sin límite práctico", "para prefijos fríos", w=1060), gap=16) +
    p("Traer un token ya calculado es mucho más barato que recalcularlo: en el repositorio, diez veces.", D, size=28),
    """Tiempo: 0:15–0:18 (3 min).
Idea clave: como recalcular es caro y guardar es barato, el KV cache se guarda en una jerarquía de almacenamiento, del más rápido y chico al más lento y grande.
Decir: el título del paper es literal: "Trading more storage for less computation". La memoria de la GPU alcanza para el lote en curso. Pero las máquinas del clúster tienen además mucha DRAM de CPU y SSDs que en un servidor de inferencia están casi ociosos. Mooncake los junta en un pool de KV cache distribuido y lo mueve con RDMA entre máquinas. DeepSeek guarda el KV cache en disco, en su sistema de archivos distribuido. Y los frameworks actuales, como Dynamo de NVIDIA o LMCache, organizan los niveles explícitamente, hasta almacenamiento remoto.
Los tamaños son órdenes de magnitud, no números de un sistema en particular.
La cuenta que lo justifica: con el modelo de costo del repositorio, calcular un token de prefill cuesta 0,05 ms y traerlo del pool 0,005 ms. Mientras traer sea más barato que recalcular, conviene.
Preguntar a la clase: ¿qué tiene de fácil este caché comparado con un caché de una base de datos? (El contenido es inmutable: no hay invalidación. Lo único es la capacidad.)
Pasar a la siguiente: el router.""",
    d.footer("Qin et al., FAST 2025, §3 · DeepSeek, 2025"), gap=40)

# 7 ─ router
def rcard(name, how, t=D, accent=False):
    return card(h3(name, t, size=32, color="acc" if accent else "ink") + p(how, t, size=24), t, accent=accent, pad=32, gap=12)
d.slide("router", D, head("El router", "Cuatro formas de elegir réplica", D) +
    row(rcard("En ronda", "Una tras otra. No sabe nada."),
        rcard("Menor carga", "La que tiene menos trabajo encolado."),
        rcard("Por sesión", "Siempre la misma para una conversación: hash de la sesión."),
        rcard("Consciente del prefijo", "Estima el tiempo al primer token en cada una y elige el menor.", accent=True), gap=24) +
    card(p("TTFT estimado = espera por lo encolado + prefill de lo que <b>no</b> está en su caché", D, size=32, color="ink", mono=False), D, pad=32, width=1596) +
    p("Si ninguna réplica llega al SLO, Mooncake rechaza el pedido antes de calcular nada: HTTP 429.", D, size=28),
    """Tiempo: 0:18–0:22 (4 min).
Idea clave: con caché de prefijos, elegir réplica es un compromiso entre ir a donde está el prefijo y no sobrecargar a nadie.
Decir: las cuatro políticas de la demo. En ronda reparte parejo pero cada turno de una conversación cae en otra réplica, así que sólo aprovecha el prompt de sistema. Menor carga balancea bien y tampoco sabe dónde está cada conversación. Por sesión, sticky, manda siempre a la misma réplica por un hash de la sesión: aprovecha el caché pero no mira la carga.
La cuarta es la del Conductor de Mooncake (§6.1). Para cada instancia de prefill calcula cuánto del prompt tiene en caché, estima cuánto va a esperar por lo que ya tiene encolado y cuánto va a tardar el prefill de lo que le falta; la suma es el tiempo al primer token estimado, y elige la menor. Si el prefijo está en otra máquina, suma el tiempo de traerlo. Y si ninguna cumple el SLO, devuelve 429 Too Many Requests antes de gastar cómputo. En el repositorio, ese rechazo es el ejercicio 5.
Conectar con Dynamo (Clase 8): elegir réplica con información de dónde está el dato es routing por contenido; el hash de sesión es hashing módulo N: si se agrega o se cae una réplica, casi todas las sesiones cambian de réplica y pierden su caché; consistent hashing movería solo una fracción 1/N.
Pasar a la siguiente: los números.""",
    d.footer("Qin et al., FAST 2025, §6.1 y algoritmo 1"), gap=36, section="El router: cómo elegir réplica con caché de prefijos.")

# 8 ─ demo router (claro)
d.slide("demorouter", L, head("Demo · router", "El router consciente del prefijo tiene el caché de sticky y la cola de menor carga", L) +
    table(["Router", "Caché", "Prefill (M tokens)", "TTFT p50", "TTFT p99", "Desbalance"], [
        ["En ronda, sin caché", "0%", "18,3", "19 s", "63 s", "1,02"],
        ["En ronda", "56%", "7,8", "119 ms", "790 ms", "1,02"],
        ["Menor carga", "57%", "7,7", "123 ms", "512 ms", "1,02"],
        ["Por sesión", "73%", "4,9", "82 ms", "4,1 s", "1,24"],
        ["Consciente del prefijo", "73%", "4,8", "79 ms", "456 ms", "1,07"],
        ["… + pool compartido", "88%", "2,1", "66 ms", "142 ms", "1,04"],
    ], L, widths=[30, 11, 18, 13, 13, 15], right=(1, 2, 3, 4, 5), size=28, bold_rows=(4,), hi_rows=(5,)),
    """Tiempo: 0:22–0:26 (4 min, con demo).
Idea clave: sin caché el clúster no da abasto; con caché, el router decide cuánto se aprovecha y cómo queda la cola.
Demo: go run ./cmd/sim router (tarda un par de segundos) y en el simulador Clúster de inferencia, cambiar el router con el caché prendido: con "En ronda" cada réplica tiene un poco de todas las conversaciones; con "Consciente del prefijo" cada app y cada conversación se queda en una réplica.
Leer la tabla: 1.000 conversaciones, 4.539 pedidos, cuatro réplicas. Sin caché hay que calcular 18 millones de tokens de prefill y el clúster se satura: el TTFT mediano es de 19 segundos. Con caché y en ronda, 56%: el prompt de sistema y algún turno que cae de casualidad en la misma réplica. Por sesión y consciente del prefijo llegan a 73%, y el prefill baja a 4,8 millones.
La diferencia entre esos dos está en la cola: sticky no mira la carga, y una réplica con conversaciones largas se satura (desbalance 1,24, p99 de 4 segundos). El consciente del prefijo balancea: p99 de 456 ms.
La última fila agrega el pool compartido: cualquier réplica puede traer un prefijo calculado en otra, a un décimo del costo. 88% de caché y p99 de 142 ms.
Pasar a la siguiente: qué pasa con poco caché.""",
    d.footer("go run ./cmd/sim router · simulador Clúster de inferencia", L, plain=True), gap=40)

# 9 ─ sticky
d.slide("sticky", D, head("Con poco caché", "Cuando la GPU recuerda poco, elegir bien vale más", D) +
    row(col(table(["Router", "Caché", "TTFT p99"], [
        ["En ronda", "40%", "7,2 s"], ["Por sesión", "44%", "26 s"], ["Consciente del prefijo", "52%", "3,0 s"], ["… + pool compartido", "88%", "155 ms"],
    ], D, widths=[50, 20, 30], right=(1, 2), size=28, bold_rows=(2,)), p("Un cuarto del caché por réplica: go run ./cmd/sim router -cache 256", D, size=24, color="muted"), gap=16, flex=1),
        card(label("En producción", D, color="warn") + p("El routing de Claude es sticky: los mensajes siguientes de una conversación vuelven al mismo servidor. En 2025, un error de routing mandaba pedidos cortos a servidores de contexto de 1M; un cambio de balanceo lo multiplicó, y por ser sticky los afectados lo siguieron estando.", D, size=28), D, width=560, pad=36), gap=48, align="center"),
    """Tiempo: 0:26–0:28 (2 min).
Idea clave: con poco caché por réplica, mandar cada pedido a donde está su prefijo sin sobrecargar a nadie es la diferencia entre 3 y 26 segundos de cola.
Decir: la misma demo con un cuarto del caché por réplica. En ronda baja a 40%. Por sesión tiene más caché que en ronda, 44%, pero algunas réplicas se saturan y el p99 llega a 26 segundos. El consciente del prefijo llega a 52% con un p99 de 3 segundos. Y el pool compartido, que no depende del caché de cada GPU, se queda en 88%.
El recuadro es un caso real: el postmortem que publicó Anthropic en septiembre de 2025. Cuenta que su routing es sticky, y que un error de routing mandaba algunos pedidos de contexto corto a servidores configurados para el contexto de un millón de tokens, y un cambio rutinario de balanceo de carga del 29 de agosto multiplicó esa fracción. Como el routing es sticky, una vez que un pedido caía en el servidor equivocado, los siguientes de esa conversación también. Además cuenta que sirven a Claude en Trainium, GPUs de NVIDIA y TPUs, con pools separados por largo de contexto.
Lección de sistemas: el router no sólo decide la latencia; decide en qué hardware y configuración cae cada pedido.
Pasar a la siguiente: el router en Kubernetes.""",
    d.footer("Anthropic, «A postmortem of three recent issues», 2025"), gap=40)

# 10 ─ k8s
d.slide("k8s", D, head("En Kubernetes", "El router es un gateway que sabe de KV cache", D) +
    row(card(label("Gateway API Inference Extension", D) + p("Extiende el Gateway de Kubernetes para elegir el pod de inferencia por el estado del modelo, no sólo por conexiones.", D, size=28), D, pad=36),
        card(label("llm-d · CNCF Sandbox, marzo de 2026", D, color="acc") + p("Sobre vLLM y esa extensión: routing consciente del prefijo, prefill y decode separados, KV cache en varios niveles y autoscaling según el tráfico.", D, size=28), D, accent=True, pad=36), gap=32) +
    p("Lo que vimos hasta acá, como recursos de Kubernetes: el balanceador de la Clase 18 se convierte en el scheduler de la Clase 19.", D, size=28),
    """Tiempo: 0:28–0:31 (3 min).
Idea clave: todo lo de esta clase ya existe empaquetado sobre Kubernetes; el router consciente del prefijo es un gateway.
Decir: un balanceador HTTP común elige el backend por cantidad de conexiones o en ronda, que vimos que es lo peor. La Gateway API Inference Extension de Kubernetes deja que el gateway elija el pod de inferencia mirando el estado del modelo: la cola, el uso de KV cache, qué prefijos tiene.
llm-d es un proyecto que entró al sandbox de la CNCF en marzo de 2026. Está armado sobre vLLM y sobre esa extensión, y trae exactamente las piezas de hoy: un scheduler de inferencia que rutea por prefijo, prefill y decode como servicios separados que escalan por su cuenta, offloading del KV cache en varios niveles y autoscaling por tráfico y hardware. NVIDIA Dynamo hace algo parecido y se integra con la misma extensión.
Conectar con la materia: el Endpoint de Kubernetes pasa a ser un objeto con estado, y el scheduler necesita información fresca de cada réplica: el mismo problema de consistencia de información de carga que en cualquier balanceador distribuido.
Pasar a la siguiente: la otra mitad de la clase, prefill y decode.""",
    d.footer("CNCF, llm-d · NVIDIA Dynamo, Gateway API Inference Extension"), gap=40)

# 11 ─ interferencia
def iters(t=D):
    widths = [44] * 6 + [480] + [44] * 6
    out = []
    for i, w in enumerate(widths):
        big_ = w > 100
        out.append(f'<div style="width:{w}px;height:72px;border-radius:6px;background:{t["warn"] if big_ else t["acc"]};display:flex;align-items:center;justify-content:center">'
                   + (f'<p style="font-family:{MONO};font-size:24px;font-weight:600;color:{t["onfill"]}">prefill de 8.000 tokens: 420 ms</p>' if big_ else "") + '</div>')
    return f'<div style="display:flex;gap:6px">{"".join(out)}</div>'
d.slide("interferencia", D, head("Prefill y decode, juntos", "Un prompt largo frena a todos los decodes de su lote", D) +
    col(label("Iteraciones de una réplica, a escala de tiempo", D), iters(), p("cada verde: una iteración de decode, ~30 ms: un token para cada secuencia del lote", D, size=24), gap=16) +
    p("Con el modelo de costo de la Clase 18: 20 ms + 0,05 ms × 8.000 tokens. Durante esa iteración, ninguna de las otras conversaciones recibe un token: el tiempo entre tokens salta de 30 a 420 ms.", D),
    """Tiempo: 0:31–0:35 (4 min).
Idea clave: con batching continuo, prefill y decode comparten iteraciones, y un prefill largo hace que todos los decodes de ese lote esperen.
Decir: en la Clase 18 vimos que Orca mete en la misma iteración secuencias en prefill y en decode. Con prompts de cientos de tokens no importaba. Con prompts de 8.000, el prefill de uno solo cuesta, con el modelo de costo del repositorio, 20 ms de leer los pesos más 0,05 ms por token: 420 ms.
El dibujo está a escala: iteraciones de decode de unos 30 ms, cada una produce un token para cada secuencia del lote, y en el medio una iteración de 420 ms. Cada usuario que estaba leyendo su respuesta ve que el texto se detiene casi medio segundo.
Las dos métricas en tensión: el prefill es lo que ve el usuario nuevo (tiempo al primer token, TTFT), el decode lo que ve el que ya está leyendo (tiempo entre tokens, TBT o TPOT). Optimizar una a costa de la otra es fácil; las dos a la vez, no.
Preguntar a la clase: ¿cómo lo resolverían? Dejar que propongan "partir el prefill" y "separar las máquinas": son las dos próximas slides.
Pasar a la siguiente: partir el prefill.""",
    d.footer("modelo de costo de servir-tokens · Agrawal et al., OSDI 2024"), gap=40, section="Prefill y decode: interferencia, prefill en pedazos y desagregación.")

# 12 ─ sarathi
def iters2(t=D):
    out = []
    for i in range(14):
        chunk = 4 <= i < 9
        out.append(f'<div style="width:{60 if chunk else 44}px;height:72px;border-radius:6px;background:{t["acc"]};display:flex;flex-direction:column;justify-content:flex-end;overflow:hidden">'
                   + (f'<div style="height:44px;background:{t["warn"]}"></div>' if chunk else "") + '</div>')
    return f'<div style="display:flex;gap:6px">{"".join(out)}</div>'
d.slide("sarathi", D, head("Sarathi-Serve · OSDI 2024", "Partir el prefill en pedazos que entran en cada iteración", D) +
    col(label("El mismo prompt, de a 512 tokens por iteración", D), iters2(), p("naranja: un pedazo de prefill; verde: los decodes, que siguen en cada iteración", D, size=24), gap=16) +
    row(big("2,6×", "más capacidad que vLLM con Mistral-7B en una A100", D, size=72), big("5,6×", "con Falcon-180B y paralelismo de pipeline", D, size=72), gap=96),
    """Tiempo: 0:35–0:38 (3 min).
Idea clave: si el prefill se hace en pedazos de tamaño fijo, cada iteración tiene un costo acotado y los decodes nunca esperan mucho.
Decir: Sarathi-Serve (Agrawal et al., OSDI 2024) propone chunked prefill: el prompt se procesa de a pedazos de, por ejemplo, 512 tokens por iteración, y en cada iteración entran también los decodes del lote. Lo llaman stall-free scheduling: los decodes no se detienen nunca. El dibujo: las iteraciones con prefill son un poco más largas, pero ninguna explota.
El costo: el prompt largo tarda más en terminar su prefill, porque se reparte en varias iteraciones; su TTFT sube un poco. A cambio, el TBT de todos los demás queda acotado.
Los resultados del paper, contra vLLM y con límites de latencia en la cola: hasta 2,6 veces más capacidad con Mistral-7B en una A100, hasta 3,7 con Yi-34B en dos A100, y hasta 5,6 con Falcon-180B usando paralelismo de pipeline, donde además las iteraciones parejas reducen las burbujas.
Hoy chunked prefill viene prendido por defecto en los servidores de inferencia principales.
Pasar a la siguiente: la otra solución, separar las máquinas.""",
    d.footer("Agrawal et al., OSDI 2024"), gap=40)

# 13 ─ distserve
d.slide("distserve", D, head("DistServe · OSDI 2024", "Prefill y decode, en máquinas distintas", D) +
    row(card(label("Pool de prefill", D, color="warn") + p("Calcula el prompt y el primer token. Limitado por el cómputo: se optimiza para el TTFT.", D, size=28), D, pad=36),
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:8px;justify-content:center"><x-shape kind="arrow-right" style="width:96px;height:40px;background:{D["muted"]}"></x-shape>{p("KV cache", D, size=24, mono=True)}</div>',
        card(label("Pool de decode", D, color="acc") + p("Genera el resto, un token por iteración, sin prefills que lo frenen: se optimiza para el TBT.", D, size=28), D, pad=36), gap=24, align="center") +
    row(big("7,4×", "más pedidos dentro del SLO", D, size=72), big("12,6×", "o un SLO 12,6 veces más estricto", D, size=72, color="blue"), gap=96) +
    p("Goodput: pedidos por segundo que cumplen el SLO de TTFT y el de TBT a la vez.", D, size=28),
    """Tiempo: 0:38–0:42 (4 min).
Idea clave: si prefill y decode tienen cuellos de botella y SLOs distintos, se pueden poner en grupos de GPUs distintos y dimensionar cada uno por separado.
Decir: DistServe (Zhong et al., OSDI 2024) separa las dos fases. Un pool de GPUs sólo hace prefill: calcula el prompt, produce el primer token y manda el KV cache por la red. Otro pool sólo hace decode: recibe el KV cache y genera el resto, sin interferencia.
Tres ventajas: no hay interferencia; cada pool puede usar el paralelismo que le conviene (por ejemplo, más paralelismo de tensores en prefill para bajar el TTFT); y cada pool escala con su propia demanda.
La métrica que proponen es goodput: no throughput, sino cuántos pedidos por segundo cumplen a la vez el SLO de tiempo al primer token y el de tiempo entre tokens. Con eso, reportan hasta 7,4 veces más pedidos o un SLO 12,6 veces más estricto que los sistemas de entonces.
El retrospectivo que publicaron sus autores a fines de 2025 dice que en 2025 la desagregación pasó a ser lo estándar en los frameworks de producción; ojo, lo dicen sus propios autores.
Pasar a la siguiente: cuánto cuesta mover el KV cache.""",
    d.footer("Zhong et al., OSDI 2024 · Hao AI Lab, 2025"), gap=36)

# 14 ─ transferencia
d.slide("transfer", D, head("El precio de separar", "Mover el KV cache cuesta mucho menos que calcularlo", D) +
    row(col(table(["", "Llama 3 70B, prompt de 8.000 tokens"], [
        ["KV cache", "320 KiB × 8.000 ≈ 2,6 GB"],
        ["Mandarlo a 800 Gb/s", "≈ 26 ms"],
        ["Calcularlo (prefill)", "≈ 400 ms en el modelo de costo"],
    ], D, widths=[40, 60], size=28), gap=16, flex=1),
        card(p("Mooncake lo manda <b>capa por capa</b> mientras el prefill sigue: cuando termina la última capa, casi todo ya llegó.", D, size=28, color="ink"), D, accent=True, width=520, pad=36), gap=48, align="center"),
    """Tiempo: 0:42–0:44 (2 min).
Idea clave: separar prefill y decode exige mover el KV cache por la red, y la cuenta muestra que se puede pagar.
Decir: con los números de la Clase 18: Llama 3 70B ocupa 320 KiB de KV cache por token; un prompt de 8.000 tokens son unos 2,6 GB. Las máquinas del testbed de Mooncake tienen hasta 800 Gb/s de RDMA entre nodos, unos 100 GB/s: 26 ms. Calcular ese prefill, con nuestro modelo de costo, son unos 400 ms. Mover cuesta del orden de un 6% de calcular.
Y no hay que esperar al final: Mooncake manda el KV cache de cada capa apenas se termina de calcular, superpuesto con el cálculo de las capas siguientes (§3, paso 3 del flujo de un pedido).
Conectar con la Clase 17: es el mismo truco que DualPipe, esconder la comunicación detrás del cómputo.
Pasar a la siguiente: la demo, donde se ve que no todo es tan simple.""",
    d.footer("Qin et al., FAST 2025, §3 y §8.1"))

# 15 ─ demo desagregado (claro)
d.slide("demodesag", L, head("Demo · desagregado", "Con 4 réplicas, partir el prefill gana; separar depende del ratio", L) +
    table(["Configuración", "TTFT p50", "TTFT p99", "TBT p99", "SLO", "Goodput"], [
        ["Colocado", "79 ms", "456 ms", "194 ms", "25%", "1,97"],
        ["Colocado, prefill de a 512", "119 ms", "881 ms", "60 ms", "100%", "7,90"],
        ["Separado, 1 prefill + 3 decode", "22 s", "87 s", "29 ms", "25%", "1,97"],
        ["Separado, 2 + 2", "163 ms", "1,4 s", "57 ms", "62%", "4,87"],
        ["Separado, 3 + 1", "102 ms", "511 ms", "244 ms", "8%", "0,60"],
    ], L, widths=[34, 13, 13, 13, 11, 16], right=(1, 2, 3, 4, 5), size=28, bold_rows=(1,)) +
    p("SLO: primer token en menos de 2 s y nunca más de 100 ms entre dos tokens. Goodput en pedidos por segundo.", L, size=24, color="muted"),
    """Tiempo: 0:44–0:48 (4 min, con demo).
Idea clave: separar elimina la interferencia, pero obliga a elegir cuántas GPUs van a cada lado; con pocas réplicas, partir el prefill da casi todo el beneficio sin ese problema.
Demo: go run ./cmd/sim desagregado, y en el simulador Clúster de inferencia, "Prefill / decode separados" moviendo la cantidad de réplicas de prefill, o el botón Comparar.
Leer la tabla, la misma carga que la demo del router y cuatro réplicas. Colocado sin más: el TBT p99 es de 194 ms por los prefills largos; sólo un 25% de los pedidos cumple el SLO. Con prefill de a 512: TBT de 60 ms y todos cumplen. Separado con 1 prefill y 3 decode: el decode va perfecto (29 ms) pero una sola réplica de prefill no da abasto y la cola explota. Con 3 y 1, al revés: el decode se satura, también se queda sin KV cache, y el salto entre el primer token y el segundo pasa el SLO. Con 2 y 2, 62%.
Lo honesto: con este modelo de costo y 4 réplicas, el prefill en pedazos gana. Lo que este modelo no captura es la ventaja principal de DistServe a escala: cada pool con su propio paralelismo y su propio hardware, y escalando por separado con la demanda. Desagregar es dimensionar dos servicios en lugar de uno.
Pasar a la siguiente: la conclusión.""",
    d.footer("go run ./cmd/sim desagregado · simulador Clúster de inferencia", L, plain=True), gap=40)

# 16 ─ ratio
d.slide("ratio", D, head("Desagregar es dimensionar dos servicios", "El ratio entre prefill y decode tiene que seguir a la carga", D) +
    row(card(label("Prompts más largos", D, color="warn") + p("Más prefill. Con agentes y documentos, la carga se corre hacia la entrada.", D, size=28), D, pad=36),
        card(label("Respuestas más largas", D, color="acc") + p("Más decode. Los modelos de razonamiento generan miles de tokens antes de contestar.", D, size=28), D, pad=36),
        card(label("La solución", D) + p("Pools que escalan por separado. Es lo que hacen llm-d y Dynamo, con el autoscaler mirando cada cola.", D, size=28), D, pad=36), gap=32),
    """Tiempo: 0:48–0:50 (2 min).
Idea clave: la desagregación cambia un problema de interferencia por uno de dimensionamiento, y ese dimensionamiento tiene que ser dinámico.
Decir: lo que mostró la demo: un ratio fijo que no acompaña a la carga desperdicia un lado y satura el otro. Y la carga cambia: con agentes, herramientas y documentos, los prompts crecen; con modelos de razonamiento, que generan miles de tokens de pensamiento antes de la respuesta, crece el decode.
La respuesta de la industria es tratar a prefill y decode como dos servicios con autoscaling propio. llm-d y Dynamo lo hacen sobre Kubernetes, cada uno con su propio deployment. En la evaluación con la carga real de Kimi, Mooncake usa 10 instancias de prefill y 10 de decode contra 20 de vLLM colocadas.
Conectar con lo que sabe la clase de Kubernetes: es un HorizontalPodAutoscaler por pool, pero la métrica no es CPU, es la cola de prefill o el uso de KV cache.
Pasar a la siguiente: el sistema completo de Mooncake.""",
    d.footer("Qin et al., Mooncake, §8.1.3 (arXiv) · CNCF, llm-d"))

# 17 ─ mooncake
def box(title, sub, t=D, color="line", w=None, accent=False):
    c = t[color] if color in t else color
    size = f"width:{w}px;" if w else "flex:1;"
    return (f'<div style="{size}display:flex;flex-direction:column;gap:8px;padding:24px;border-radius:12px;background:{t["card"]};border:{"2px" if accent else "1px"} solid {c}">'
            f'{h3(title, t, size=28)}{p(sub, t, size=24)}</div>')
d.slide("mooncake", D, head("Mooncake · §3", "El KV cache en el centro", D) +
    box("Conductor", "el scheduler global: elige prefill y decode para cada pedido según dónde está el KV cache y la carga", D, color="acc", w=1612, accent=True) +
    row(box("Pool de prefill", "reusa el prefijo y calcula el resto", D, color="warn"),
        box("Pool de KV cache", "DRAM y SSD de las máquinas de GPU, en bloques", D),
        box("Pool de decode", "batching continuo con lo que llega", D, color="acc"), gap=24) +
    p("1 · traer el prefijo del pool → 2 · prefill de lo que falta → 3 · mandar el KV cache capa por capa (RDMA) → 4 · decode", D, size=28, color="ink", mono=False),
    """Tiempo: 0:50–0:54 (4 min).
Idea clave: Mooncake junta las dos ideas de hoy, caché de prefijos compartido y desagregación, alrededor de un pool de KV cache distribuido.
Decir: la arquitectura (figura 1 y §3). El Conductor es el scheduler global: para cada pedido elige una instancia de prefill y una de decode mirando dónde está el KV cache y cuánta carga tiene cada una. El pool de prefill y el de decode son grupos de máquinas de GPU separados. El pool de KV cache no son máquinas aparte: es la DRAM y los SSDs de las mismas máquinas de GPU, que en un servidor de inferencia están casi ociosos, organizados en bloques. Un componente en cada nodo, el Messenger, mueve el KV cache con GPUDirect RDMA.
El flujo de un pedido, los cuatro pasos de abajo: la instancia de prefill elegida trae del pool el KV cache del prefijo que ya existe; calcula sólo el resto (si es largo, en pedazos y en pipeline); manda el KV cache resultante al nodo de decode capa por capa, mientras calcula; y el pedido entra al batching continuo del decode. El decode tiene su propio control: si al llegar ve que no cumpliría el SLO de TBT, todavía lo puede rechazar.
Pasar a la siguiente: qué hacen con los prompts muy largos.""",
    d.footer("Qin et al., FAST 2025, §3 y figuras 1 y 4"), gap=32, section="Mooncake: el KV cache en el centro, contexto largo, sobrecarga y resultados.")

# 18 ─ CPP
d.slide("cpp", D, head("Contexto largo · §5.1", "Partir el prompt entre varias máquinas de prefill", D) +
    row(card(label("Pipeline por pedazos (Mooncake)", D, color="acc") + p("Cada pedazo del prompt va a otra máquina del grupo, en pipeline. Sólo se comunica en el borde entre etapas.", D, size=28), D, accent=True, pad=36),
        card(label("Tensores entre nodos", D) + p("Dos all-reduce por capa por la red, como vimos en la Clase 16: demasiado.", D, size=28), D, pad=36),
        card(label("Paralelismo de secuencia", D) + p("Comunicación entre nodos al menos una vez por capa.", D, size=28), D, pad=36), gap=32),
    """Tiempo: 0:54–0:56 (2 min).
Idea clave: para prompts de cientos de miles de tokens, el prefill de un solo pedido se reparte entre varias máquinas, y la forma de hacerlo se elige por cuánto comunica, como en la Clase 17.
Decir: un prefill de un prompt enorme en una sola máquina tarda demasiado y rompe el SLO de TTFT. Mooncake agrupa X nodos de prefill y parte el prompt en pedazos de a lo sumo prefill_chunk tokens; cada pedazo corre en otra etapa del grupo a la vez, en pipeline (chunked pipeline parallelism). La comunicación entre nodos es sólo en los bordes de las etapas.
El paper compara con las alternativas: tensores entre nodos haría dos all-reduce por capa por RDMA, y paralelismo de secuencia comunicaría entre nodos al menos una vez por capa. Es el mismo razonamiento de la Clase 17 sobre qué dimensión va por qué enlace: el pipeline es lo que menos comunica, así que es lo que cruza la red.
Pasar a la siguiente: qué pasa cuando no alcanza.""",
    d.footer("Qin et al., FAST 2025, §5.1"))

# 19 ─ sobrecarga
d.slide("sobrecarga", D, head("Sobrecarga · §7", "Rechazar temprano, y prediciendo", D) +
    row(col(p("Mooncake está pensado para cuando la demanda supera la capacidad. Si un pedido no va a cumplir el SLO, conviene rechazarlo <b>antes</b> de gastar su prefill.", D, size=28),
            p("Pero rechazar mirando la carga de ahora produce oscilaciones: prefill y decode se cargan en contrafase. La versión final predice la carga del decode.", D, size=28), gap=24, flex=1),
        col(table(["Política", "Rechazados"], [["Sin rechazo temprano", "4.183"], ["Rechazo temprano", "3.771"], ["Con predicción", "3.589"]], D, widths=[62, 38], right=(1,), size=28, bold_rows=(2,)),
            p("23.000 pedidos de la traza a doble velocidad, 8 prefill + 8 decode", D, size=24, color="muted"), gap=16, width=620), gap=56, align="center"),
    """Tiempo: 0:56–1:00 (4 min).
Idea clave: con sobrecarga, la decisión no es cómo atender a todos sino a quién no atender, y hacerlo temprano ahorra cómputo, pero la realimentación puede oscilar.
Decir: casi todos los papers de serving suponen que todos los pedidos se atienden. Mooncake no: Kimi tenía más demanda que GPUs. Si un pedido va a terminar fuera del SLO, todo el cómputo que se le dedicó se pierde. El peor caso es hacer el prefill y que después el decode lo rechace.
Rechazo temprano: al llegar el pedido, el Conductor mira la carga del prefill y la del decode y, si no da, lo rechaza antes de calcular. Pero aparece un problema de control clásico: el decode se va a cargar recién cuando termine el prefill, así que el rechazo mira una carga vieja. Resultado: el prefill y el decode oscilan en contrafase (figura 9). La solución es predecir la carga del decode en lugar de mirarla.
La tabla es la 3 de la versión de arXiv: con la traza a doble velocidad en 8 + 8 máquinas, los rechazos bajan de 4.183 a 3.589.
Conectar con la materia: es control con realimentación con demora, el mismo problema que el control de congestión de TCP. Y es el ejercicio 5 del repositorio.
Pasar a la siguiente: los resultados.""",
    d.footer("Qin et al., Mooncake, §7 y tabla 3 (arXiv)"), gap=40)

# 20 ─ resultados (claro)
d.slide("resultados", L, col(eyebrow("Resultados · FAST 2025, mejor paper", L), h2("Más del doble de pedidos atendidos en producción", L), gap=16) +
    row(big("+115%", "pedidos de Kimi atendidos en clústeres A800", L), big("+107%", "en clústeres H800", L), big("59–498%", "más capacidad efectiva en experimentos, cumpliendo el SLO", L, color="blue"), gap=80) +
    p("Miles de nodos, más de 100 mil millones de tokens por día. La ganancia es mayor con contextos largos.", L, size=28),
    """Tiempo: 1:00–1:03 (3 min).
Idea clave: el caché de prefijos compartido y la desagregación, juntos, duplican lo que el mismo hardware puede atender dentro del SLO.
Decir: los números del resumen de la versión de FAST. En el despliegue real de Kimi, Mooncake atiende un 115% más de pedidos en los clústeres de A800 y un 107% más en los de H800 que el sistema anterior. En experimentos, la capacidad efectiva —pedidos atendidos cumpliendo el SLO— sube entre 59% y 498% según el escenario, más con contextos largos. Corre en miles de nodos y procesa más de 100 mil millones de tokens por día.
Sobre las versiones: los resultados de producción y de capacidad son de la versión de FAST; la sección de sobrecarga, la traza y las estadísticas de caché están sólo en la versión extendida de arXiv (v4), y las citamos con su numeración.
El dato de contexto: es un paper de una conferencia de almacenamiento, no de machine learning, y ganó el premio al mejor paper. El KV cache es un problema de almacenamiento.
Pasar a la siguiente: cómo sirve DeepSeek un modelo de expertos.""",
    d.footer("Qin et al., FAST 2025", L), gap=64)

# 21 ─ moe serving
d.slide("moeserve", D, head("DeepSeek-V3 en producción", "Para servir expertos, paralelismo de expertos enorme", D) +
    table(["", "Prefill", "Decode"], [
        ["Unidad de despliegue", "4 nodos, 32 GPUs", "18 nodos, 144 GPUs"],
        ["Expertos ruteados por GPU", "9", "2"],
        ["Superposición", "dos microbatches alternados", "pipeline de 5 etapas"],
        ["Balanceo", "tokens de entrada por GPU", "KV cache y pedidos por GPU"],
    ], D, widths=[34, 33, 33], size=28) +
    p("Con 8 de 256 expertos activos, cada experto ve pocos tokens: hace falta un lote enorme repartido en muchas GPUs para que cada una tenga trabajo.", D, size=28),
    """Tiempo: 1:03–1:07 (4 min).
Idea clave: servir un modelo de expertos es un problema de paralelismo como el de la Clase 17, y DeepSeek lo resuelve con prefill y decode separados y paralelismo de expertos de 144 GPUs en el decode.
Decir: DeepSeek publicó en febrero de 2025 su sistema de inferencia para V3 y R1. Prefill y decode van separados. En prefill, cada unidad son 4 nodos (32 GPUs) con paralelismo de expertos de 32; en decode, 18 nodos (144 GPUs) con paralelismo de expertos de 144.
¿Por qué tan grande? Cada token activa 8 de 256 expertos. Para que la multiplicación de cada experto sea eficiente, cada experto necesita muchos tokens, o sea un lote global enorme; y repartir los expertos en más GPUs hace que cada una lea menos pesos, que en decode es lo que limita (Clase 18). En decode cada GPU tiene 2 expertos ruteados y uno compartido.
Igual que DualPipe en el entrenamiento, superponen comunicación con cómputo: en prefill, dos microbatches que se alternan; en decode, la atención partida en un pipeline de 5 etapas. Y tres balanceadores: el de prefill iguala la atención y los tokens por GPU, el de decode iguala KV cache y pedidos, y el de expertos iguala la carga del dispatch.
Pasar a la siguiente: lo que eso atiende en un día.""",
    d.footer("DeepSeek, Open Infra Index, día 6, 2025"), gap=40, section="La frontera: expertos, decodificación especulativa, KV más chico y los laboratorios.")

# 22 ─ un día de DeepSeek (claro)
d.slide("deepseekday", L, col(eyebrow("Un día de DeepSeek · 27 y 28 de febrero de 2025", L), h2("608 mil millones de tokens de entrada en 24 horas", L), gap=16) +
    row(big("227", "nodos de 8 H800 en promedio (226,75)", L), big("73,7 k", "tokens de entrada por segundo por nodo de prefill", L, color="blue"),
        big("14,8 k", "tokens de salida por segundo por nodo de decode", L, color="blue"), big("20–22", "tokens por segundo para cada usuario", L, color="warn"), gap=64) +
    p("Margen teórico de 545% si todo se cobrara a precio de R1; el informe aclara que el ingreso real es mucho menor.", L, size=28),
    """Tiempo: 1:07–1:10 (3 min).
Idea clave: con todo esto junto, un servicio de frontera atiende cientos de miles de millones de tokens por día con unos pocos cientos de nodos.
Decir: las estadísticas que publicó DeepSeek para un día. En promedio usaron 226,75 nodos de 8 H800, con un pico de 278 (bajan de noche y usan las GPUs para investigación y entrenamiento). Entraron 608 mil millones de tokens, el 56,3% del caché en disco, y salieron 168 mil millones. Cada nodo de prefill procesó unos 73.700 tokens de entrada por segundo, contando los aciertos de caché; cada nodo de decode generó unos 14.800 tokens por segundo. Y cada usuario ve unos 20 a 22 tokens por segundo.
La cuenta de servilleta: 14.800 tokens por segundo por nodo, a 21 por usuario, son unos 700 usuarios generando a la vez en cada nodo de decode. Es el lote de la Clase 18, a escala.
El margen: si todos los tokens se cobraran a la tarifa de R1, el ingreso diario sería de 562.027 dólares contra 87.072 de costo, un margen teórico de 545%. El mismo informe aclara que el ingreso real es mucho menor: V3 es más barato, la web y la app son gratis, y hay descuentos nocturnos.
Pasar a la siguiente: otra forma de atacar el decode.""",
    d.footer("DeepSeek, Open Infra Index, día 6, 2025", L), gap=64)

# 23 ─ especulativo
d.slide("especulativo", D, head("Decodificación especulativa", "Un modelo chico propone, el grande verifica de a varios", D) +
    row(col(p("Verificar k + 1 tokens en una pasada cuesta casi lo mismo que generar uno: el decode está limitado por leer los pesos.", D, size=28),
            p("(1 − α^(k+1)) / (1 − α)", D, size=44, color="acc", mono=True, weight=600),
            p("tokens por pasada del modelo grande, si acepta cada token propuesto con probabilidad α (Leviathan et al., 2023)", D, size=24), gap=20, width=680),
        col(table(["k", "α = 0,6", "α = 0,8", "α = 0,9"], [["2", "1,78×", "2,22×", "2,46×"], ["4", "1,92×", "2,80×", "3,41×"], ["8", "1,77×", "3,09×", "4,38×"]],
                  D, widths=[16, 28, 28, 28], right=(1, 2, 3), size=28),
            p("borrador al 5% del costo · go run ./cmd/sim especulativo", D, size=24, color="muted"), gap=16, flex=1), gap=56, align="center"),
    """Tiempo: 1:10–1:14 (4 min).
Idea clave: como el decode está limitado por la memoria, verificar varios tokens propuestos cuesta lo mismo que generar uno, y eso permite generar más de un token por pasada del modelo grande.
Decir: la idea de Leviathan et al. (ICML 2023): un modelo chico y barato propone k tokens, uno detrás de otro. El modelo grande corre una sola pasada sobre los k y verifica cuáles habría elegido él; acepta los primeros que coinciden y en el primero que no, pone el suyo. El resultado tiene exactamente la misma distribución que si generara el grande solo (con decodificación greedy, el mismo texto).
¿Por qué conviene? Por lo de la Clase 18: una pasada de decode cuesta casi lo mismo con 1 token que con k+1, porque lo caro es leer los pesos. Si cada token propuesto se acepta con probabilidad α, se obtienen en promedio (1 − α^(k+1))/(1 − α) tokens por pasada.
La tabla es de la demo, con un borrador que cuesta 5% de una pasada del grande: con α = 0,8 y 4 tokens propuestos, 2,8 veces más rápido. Con α bajo, proponer 8 es peor que proponer 4: se tira casi todo.
Hoy: el borrador suele ser una cabeza chica entrenada sobre el mismo modelo (EAGLE), y DeepSeek-V3 entrena un módulo de predicción de varios tokens (MTP) que se puede usar para esto.
Pasar a la siguiente: achicar el KV cache.""",
    d.footer("Leviathan et al., ICML 2023 · DeepSeek-AI, 2024, §2.2"), gap=40)

# 24 ─ kv más chico
d.slide("kvchico", D, head("Achicar el KV cache", "Los modelos nuevos se diseñan para servirse", D) +
    row(card(label("Grouped-query attention", D) + p("Varias cabezas de query comparten keys y values. Llama 3 70B: 8 cabezas de KV, 320 KiB por token (Clase 18).", D, size=28), D, pad=36),
        card(label("Multi-head latent attention", D, color="acc") + p("Keys y values comprimidos juntos en un vector chico. DeepSeek-V2: 93,3% menos KV cache que DeepSeek 67B.", D, size=28), D, accent=True, pad=36),
        card(label("Cuantizar", D) + p("El KV cache en 8 bits o menos: la mitad de memoria y de bytes por la red.", D, size=28), D, pad=36), gap=32) +
    p("Menos KV cache por token es más secuencias en el lote, más prefijos en el caché y menos bytes para mandar de prefill a decode.", D, size=28),
    """Tiempo: 1:14–1:17 (3 min).
Idea clave: como el KV cache decide casi todo, los modelos de frontera cambiaron su arquitectura para que ocupe menos.
Decir: tres técnicas. Grouped-query attention, que ya vimos: Llama 3 70B tiene 64 cabezas de query pero sólo 8 de keys y values, y por eso ocupa 320 KiB por token en lugar de 2,5 MiB. Multi-head latent attention, de DeepSeek: en lugar de guardar keys y values por cabeza, guarda un vector latente comprimido y los reconstruye al calcular; el paper de DeepSeek-V2 reporta, comparando con su modelo denso de 67B, un 93,3% menos de KV cache. Y cuantizar el KV cache, a 8 bits o menos.
Cada byte menos por token se multiplica tres veces en lo de hoy: entran más secuencias en el lote (Clase 18), caben más prefijos en cada nivel del caché, y hay menos para transferir de prefill a decode.
Preguntar a la clase: ¿qué relación tiene esto con FP8 en el entrenamiento de la Clase 17? (Es el mismo argumento: menos bytes por número es menos memoria y menos red.)
Pasar a la siguiente: qué hacen los laboratorios.""",
    d.footer("DeepSeek-AI, DeepSeek-V2, 2024 · Grattafiori et al., 2024"), gap=40)

# 25 ─ labs
d.slide("labs", D, head("¿Y los laboratorios?", "Lo que está publicado sobre cómo sirven", D) +
    table(["Quién", "Qué se sabe"], [
        ["DeepSeek", "todo el sistema: prefill y decode separados, expertos en 144 GPUs, caché en disco"],
        ["Moonshot (Kimi)", "Mooncake: pool de KV cache y desagregación"],
        ["Anthropic", "Trainium, GPUs y TPUs; pools por largo de contexto; routing sticky"],
        ["Google, OpenAI", "poco sobre su serving"],
        ["Código abierto", "vLLM, SGLang, TensorRT-LLM, Dynamo, llm-d: todo lo de hoy"],
    ], D, widths=[24, 76], size=28),
    """Tiempo: 1:17–1:21 (4 min).
Idea clave: lo que sabemos con detalle viene de DeepSeek, Moonshot y los frameworks abiertos; del resto hay pistas, y las mismas restricciones físicas.
Decir: recorrer la tabla. DeepSeek publicó su sistema de inferencia completo, con números de un día de producción. Moonshot publicó Mooncake. De Anthropic, lo que hay es el postmortem de septiembre de 2025: sirven Claude en Trainium de AWS, GPUs de NVIDIA y TPUs de Google, con pools de servidores separados según el largo de contexto y routing sticky. Google y OpenAI publicaron poco sobre cómo sirven.
Y la buena noticia para la materia: todo lo de hoy está en código abierto. vLLM y SGLang tienen caché de prefijos, prefill en pedazos y desagregación; Dynamo de NVIDIA y llm-d los orquestan en un clúster con routing por prefijo.
Remarcar, como en la Clase 17: que no esté publicado no quiere decir que lo hagan distinto. El KV cache de un token sigue ocupando lo mismo para todos, y el decode sigue limitado por la memoria.
Pasar a la siguiente: qué llevarse.""",
    d.footer("DeepSeek, 2025 · Qin et al., 2025 · Anthropic, 2025"), gap=40)

# 26 ─ takeaways
d.slide("takeaways", D, head("Para llevarse", "El KV cache dejó de ser un detalle de una réplica", D) +
    row(card(label("Guardarlo", D, color="acc") + p("Entre pedidos, en niveles: HBM, DRAM del clúster, disco. Traer es más barato que recalcular.", D, size=28), D),
        card(label("Encontrarlo", D, color="acc") + p("El router manda cada pedido a donde está su prefijo, sin cargar de más a ninguna réplica.", D, size=28), D),
        card(label("Separarlo", D, color="acc") + p("Prefill y decode en pedazos o en máquinas distintas, con el ratio siguiendo a la carga.", D, size=28), D), gap=32),
    """Tiempo: 1:21–1:24 (3 min).
Idea clave: cerrar las cuatro clases: entrenar y servir son problemas de sistemas distribuidos, y en serving el estado que importa es el KV cache.
Decir: tres verbos. Guardar: el KV cache se conserva entre pedidos, en una jerarquía de almacenamiento, porque traerlo cuesta un décimo de recalcularlo y la mitad de cada prompt ya se vio. Encontrar: el router pasa de balanceador a scheduler con información de dónde está cada prefijo; mal hecho, satura réplicas (sticky) o tira el caché (ronda). Separar: prefill y decode tienen necesidades opuestas; se separan en el tiempo (pedazos) o en el espacio (desagregación), y en el segundo caso hay que dimensionar dos servicios.
Conectar con las cuatro clases: en la 16 y la 17 el estado era el modelo y se partía en dimensiones; en la 18 y la 19 el estado es el KV cache, y se pagina, se comparte, se cachea y se transfiere. Las mismas preguntas: qué se parte, qué se comunica, qué pasa cuando falla.
Pasar a la siguiente: el repositorio.""",
    d.footer())

# 27 ─ demo
cmds = [("go run ./cmd/sim router", "el router y el caché de prefijos"), ("go run ./cmd/sim desagregado", "juntos, en pedazos o separados"),
        ("go run ./cmd/sim especulativo", "decodificación especulativa"), ("go test ./...", "los tests de lo que dijimos")]
lines = "".join(f'<div style="display:flex;gap:48px">{p(c, D, size=32, color="ink", mono=True, width=760)}{p(w, D, size=32, color="muted")}</div>' for c, w in cmds)
d.slide("demo", D, head("El repositorio", "servir-tokens, ahora con un clúster", D) +
    f'<div style="display:flex;flex-direction:column;gap:20px;background:#0A0C0F;border:1px solid {D["line"]};border-radius:16px;padding:48px">{lines}</div>' +
    p("Paquetes nuevos: prefix (hashes encadenados y LRU) y cluster (routers y desagregación). Y el simulador Clúster de inferencia.", D, size=28),
    """Tiempo: 1:24–1:27 (3 min).
Idea clave: el mismo motor de la Clase 18, ahora con caché de prefijos, prefill en pedazos y varias réplicas detrás de un router.
Decir: el paquete prefix implementa el caché con hashes encadenados y LRU; el motor lo usa al admitir un pedido para saber cuánto prefill puede saltear, y el paquete cluster junta cuatro motores detrás de un router, colocados o desagregados, en tiempo virtual.
Los tests verifican las afirmaciones de la clase: el router consciente del prefijo reusa más y tiene mejor p99 que el de ronda; sticky tiene peor cola; el pool compartido sube el reuso; el prefill en pedazos deja el TBT bajo el SLO; y el desagregado entrega todos los tokens con menor TBT que el colocado.
Los ejercicios del README: compartir bloques físicos entre secuencias con el mismo prefijo, rechazo temprano, replicación de hot spots de Mooncake y un ratio de prefill y decode adaptativo.
Pasar a la siguiente: preguntas.""",
    d.footer("go test ./..."))

# 28 ─ preguntas
qs = ["¿Por qué un bloque sólo se puede reusar si es igual todo lo anterior? ¿Cómo lo garantiza el hash encadenado?",
      "¿Qué tiene el router consciente del prefijo que no tienen el de ronda ni el sticky?",
      "Un prompt de 8.000 tokens en un lote de decodes: ¿qué ve el usuario que ya estaba leyendo? ¿Cómo lo evita Sarathi?",
      "¿Por qué con prefill y decode separados el ratio tiene que seguir a la carga?",
      "¿Por qué verificar k tokens propuestos cuesta casi lo mismo que generar uno?"]
d.slide("preguntas", D, head("Para repasar", "Preguntas y lecturas", D) +
    row(f'<ol style="flex:1.4;font-size:28px;line-height:1.4;color:{D["ink"]}">' + "".join(f"<li>{q}</li>" for q in qs) + "</ol>",
        card(h3("Lecturas", D) + p("<b>Qin et al.</b>, Mooncake, FAST 2025.", D, size=24, color="ink") +
             p("Zhong et al., DistServe, OSDI 2024. Agrawal et al., Sarathi-Serve, OSDI 2024.", D, size=24) +
             p("DeepSeek, Inference System Overview, 2025.", D, size=24), D, flex=1, pad=40, gap=20), gap=64),
    """Tiempo: 1:27–1:30 (3 min).
Idea clave: cerrar con las preguntas de repaso. Las respuestas cortas, para tener a mano:
1. Porque las keys y values de un token dependen de todos los tokens anteriores. El hash del bloque i incluye el del bloque i−1, así que si coincide en i, coincide todo el prefijo.
2. Estima el tiempo al primer token en cada réplica combinando lo que tiene en caché con su cola: va a donde está el prefijo salvo que esa réplica esté cargada. Ronda ignora el caché; sticky ignora la carga.
3. Que el texto se detiene unos 400 ms: todos los decodes del lote esperan esa iteración. Sarathi parte el prefill en pedazos de tamaño fijo por iteración, así que ninguna iteración se alarga tanto.
4. Porque cada pool tiene una capacidad fija: si hay más prefill del que su pool puede hacer, la cola de TTFT explota aunque el decode esté libre, y al revés. La mezcla de prompts largos y respuestas largas cambia.
5. Porque el decode está limitado por leer los pesos de la memoria, no por el cómputo: una pasada sobre k+1 tokens lee los mismos pesos una sola vez.
Lecturas: Mooncake es el paper de la clase. Complementarias: DistServe, Sarathi-Serve y el resumen del sistema de inferencia de DeepSeek. Los apuntes tienen más preguntas.""",
    d.footer())

d.write_index("Clase 19 — Del motor al clúster")
print(len(d.order), "slides")
