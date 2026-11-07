import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidekit import *
from cover import cover

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "clase-21")
d = Deck(ROOT, 21, "Repartir un clúster de GPUs")
D, L = DARK, LIGHT
SIM = "simulador Planificador de GPUs"


def cells(colors, t=D, size=40, gap=6, labels=None):
    """Una fila de GPUs: un cuadrado por GPU, del color dado (None = libre)."""
    out = []
    for i, c in enumerate(colors):
        # None = GPU libre: un cuadrado vacío con borde verde punteado.
        look = f"border:2px dashed {t['acc']}" if c is None else f"background:{t[c] if c in t else c}"
        lab = labels[i] if labels else ""
        out.append(f'<div style="width:{size}px;height:{size}px;border-radius:6px;{look};display:flex;align-items:center;justify-content:center">'
                   + (f'<p style="font-family:{MONO};font-size:24px;font-weight:600;color:{t["onfill"]}">{lab}</p>' if lab else "") + '</div>')
    return f'<div style="display:flex;gap:{gap}px">{"".join(out)}</div>'


def server(colors, t=D, size=40, name=None):
    inner = cells(colors, t, size)
    lab = p(name, t, size=24, color="muted", mono=True) if name else ""
    return (f'<div style="display:flex;flex-direction:column;gap:10px;padding:16px;border-radius:12px;border:1px solid {t["line"]};background:{t["card"]}">'
            f'{lab}{inner}</div>')


# 1 ─ portada
d.raw("cover", cover(21, "Repartir un clúster de GPUs",
    "Muchos equipos, un solo clúster: quién usa las GPUs, cuándo y dónde, sin saber cuánto va a durar nada",
    "Paper: Jeon et al., <i>Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads</i>, USENIX ATC 2019 · con Tiresias (NSDI 2019), Gandiva (OSDI 2018) y MLaaS in the Wild (NSDI 2022)",
    """Tiempo: 0:00–0:02 (2 min).
Idea clave: en las clases 16 y 17 el entrenamiento ya tenía sus 16.000 GPUs; hoy la pregunta es la anterior: cómo decide un clúster compartido quién las usa.
Decir: en la Clase 16 vimos cómo se parte un modelo entre GPUs, y en la 17 cómo entrena Llama 3 en 16.384 GPUs. Siempre supusimos que las GPUs estaban ahí, todas para nosotros. En la realidad, un clúster de GPUs lo comparten cientos de usuarios y decenas de equipos, con trabajos de una GPU y trabajos de cientos, y alguien tiene que decidir quién corre, cuándo y en qué GPUs.
El paper de hoy es el análisis de Philly, el clúster de entrenamiento de Microsoft, publicado en ATC 2019: es la primera traza grande de un clúster de deep learning, y casi todos los planificadores que vinieron después la usan. Lo acompañan Tiresias, que propone cómo planificar sin saber cuánto dura cada trabajo, Gandiva, que reparte las GPUs en el tiempo, y el análisis del clúster de Alibaba, de 2022.
Es una clase de planificación: los problemas son los clásicos de sistemas operativos (deadlock, bloqueo de cabeza de cola, SRTF, hambre), con dos vueltas nuevas: los trabajos piden muchas GPUs a la vez, y dónde corren cambia cuánto tardan.
Pasar a la siguiente: el mapa de la clase."""), section="La carga: quién pide GPUs, cuántas y por cuánto tiempo.")

# 2 ─ mapa
d.slide("mapa", D, head("Dónde estamos", "Del trabajo con todas las GPUs al clúster que las reparte", D) +
    table(["Parte", "La pregunta", "Lo clásico"], [
        ["La carga", "¿quién pide GPUs, cuántas, por cuánto?", "trazas de clústeres"],
        ["Todo o nada", "¿qué pasa si un trabajo recibe sólo parte de sus GPUs?", "deadlock, bloqueo de cabeza de cola"],
        ["Localidad", "¿esperar el lugar bueno o arrancar en uno peor?", "fragmentación"],
        ["Sin duraciones", "¿a quién atender primero si nadie sabe cuánto tarda?", "SRTF, LAS, preempción"],
        ["Compartir", "¿y entre equipos, o una GPU entre varios?", "time-sharing, cuotas"],
    ], D, widths=[20, 50, 30], size=30),
    """Tiempo: 0:02–0:04 (2 min).
Idea clave: son cinco preguntas de planificación, y cada una tiene un pariente clásico de la materia o de sistemas operativos.
Decir: recorrer las filas. Primero la carga: sin entender qué piden los usuarios no se puede diseñar nada. Después, que un trabajo distribuido necesita todas sus GPUs a la vez: si recibe la mitad, no puede dar un paso, y aparecen el deadlock y el bloqueo de cabeza de cola. Tercero, dónde: las GPUs de un servidor se hablan por NVLink y las de servidores distintos por la red, así que la misma cantidad de GPUs no vale lo mismo en cualquier lado. Cuarto, el orden: lo óptimo para el tiempo de respuesta es atender primero al que menos le falta, y nadie sabe cuánto le falta. Y por último compartir: entre equipos, y entre trabajos dentro de una misma GPU.
Conectar con la Clase 20: ahí el recurso escaso eran las GPUs para servir muchos modelos; hoy son las GPUs para entrenar.
Pasar a la siguiente: el clúster del paper.""",
    d.footer())

# 3 ─ philly (claro)
d.slide("philly", L, col(eyebrow("Philly · Microsoft · Jeon et al., ATC 2019, §2", L), h2("Dos meses y medio de un clúster de entrenamiento compartido", L), gap=16) +
    row(big("96.260", "trabajos, de octubre a diciembre de 2017", L), big("75", "días de logs", L, color="blue"),
        big("14", "clústeres virtuales, uno por equipo, con su cuota", L, color="warn"), big("52%", "uso promedio de las GPUs asignadas", L), gap=72) +
    p("Cientos de servidores, miles de GPUs del mismo modelo, en dos tipos de servidor: de 2 y de 8 GPUs. Planificador: YARN con gang scheduling y localidad.", L, size=28),
    """Tiempo: 0:04–0:08 (4 min).
Idea clave: Philly es un clúster de verdad, compartido por muchos equipos, y aun con casi todas las GPUs asignadas, las que están asignadas trabajan la mitad del tiempo.
Decir: Microsoft juntó 75 días de logs de su clúster de entrenamiento, de octubre a diciembre de 2017: 96.260 trabajos de 14 clústeres virtuales. Un clúster virtual es la porción de un equipo: tiene una cuota de GPUs, y si otro equipo no usa la suya, se la pueden prestar. El paper no da el número exacto de GPUs: dice cientos de servidores y miles de GPUs, en dos tipos de servidor, de 2 y de 8 GPUs.
El planificador es YARN, el de Hadoop, extendido para GPUs: hace gang scheduling y trata de respetar la localidad. Lo vamos a ver en detalle.
El número que sorprende: casi todas las GPUs están asignadas, pero las asignadas se usan en promedio sólo un 52%. El paper encuentra dos razones, y las dos son de esta clase: los trabajos repartidos en muchos servidores, que pasan tiempo esperando la red, y la interferencia entre trabajos que comparten un servidor.
Contexto: en 2017 el entrenamiento grande eran visión y traducción en pocas GPUs. La escala cambió; los problemas no.
Pasar a la siguiente: qué piden los usuarios.""",
    d.footer("Jeon et al., ATC 2019, §1 y §2.4", L), gap=64)

# 4 ─ carga
d.slide("carga", D, head("La forma de la carga", "Muchísimos trabajos chicos, unos pocos enormes", D) +
    row(col(label("Philly, 2017", D),
            p("Más del <b>82%</b> de los trabajos usaba una sola GPU.", D, size=28),
            p("Duración mediana: <b>26 minutos</b>. El p90: 25 horas.", D, size=28),
            p("El <b>0,5%</b> tardaba más de una semana, y los que piden más GPUs duran más.", D, size=28), gap=20, flex=1),
        col(label("La carga del repositorio", D),
            table(["GPUs", "Trabajos", "GPU·tiempo"], [["1", "70%", "10%"], ["2 a 8", "24%", "46%"], ["más de 8", "6%", "45%"]], D, widths=[40, 30, 30], right=(1, 2), size=28),
            p("El 1% de los trabajos que más piden se lleva la mitad del clúster.", D, size=24), gap=16, width=720), gap=72, align="start"),
    """Tiempo: 0:08–0:12 (4 min).
Idea clave: casi todos los trabajos son chicos, pero casi todo el GPU·tiempo lo piden unos pocos trabajos grandes y largos; el planificador tiene que servir bien a los dos.
Decir: lo que se sabe de Philly. El análisis de Alibaba de 2022 resume que más del 82% de los trabajos de Philly corrían en una sola GPU. La duración mediana fue de 26 minutos, pero el p90 fue de 25 horas (también de ese análisis), y Jeon et al. reportan que el 0,5% de los trabajos tardó más de una semana. Y los que piden más GPUs tienden a durar más.
Esto se llama cola pesada: la mediana dice poco. Comparar con big data: en Hadoop o Spark, las tareas duran de milisegundos a pocas horas.
A la derecha, la carga sintética del repositorio, armada con esa forma (los porcentajes son ilustrativos, no de Philly): el 70% de los trabajos usa una GPU y pide sólo el 10% del GPU·tiempo; el 6% usa más de 8 y pide el 45%. El 1% de los trabajos más grandes se lleva la mitad. GPU·tiempo es GPUs por duración: lo que un trabajo le cuesta al clúster.
Preguntar a la clase: si optimizan el tiempo de espera promedio por trabajo, ¿a quién favorecen? (A los chicos, que son casi todos.) ¿Y si optimizan el uso del clúster? (A los grandes, que son casi todo el GPU·tiempo.)
Pasar a la siguiente: lo otro que no se sabe de un trabajo, si va a terminar.""",
    d.footer("Weng et al., NSDI 2022, §2 y §3.2 · Jeon et al., §2.4 · go run ./cmd/sim carga"), gap=40)

# 5 ─ fallas
d.slide("fallas", D, head("Nadie sabe cuánto va a durar su trabajo", "Ni siquiera si va a terminar", D) +
    row(col(table(["Cómo terminó", "Trabajos", "GPU·tiempo"], [
        ["Bien", "69,3%", "44,5%"], ["Lo mató el usuario", "13,5%", "37,7%"], ["Falló", "17,2%", "17,8%"]],
        D, widths=[46, 27, 27], right=(1, 2), size=28, bold_rows=(1,)), p("Jeon et al., tabla 6", D, size=24, color="muted"), gap=12, flex=1),
        col(card(label("Por qué fallan", D, color="warn") +
                 p("Sobre todo errores del usuario: falta de memoria en la CPU, entradas incorrectas, errores en el código. Las de infraestructura son el 6,2% de los intentos, pero el 36% del tiempo hasta fallar.", D, size=28), D, pad=36),
            p("Lección del paper: atrapar los errores antes de pedir un clúster grande.", D, size=28, color="ink"), gap=24, width=760), gap=56, align="center"),
    """Tiempo: 0:12–0:16 (4 min).
Idea clave: el planificador no conoce la duración de los trabajos, y casi un tercio ni siquiera termina bien; eso descarta las políticas que necesitan saber cuánto falta.
Decir: la tabla 6 del paper. De los 96.260 trabajos, el 69,3% terminó bien y usó el 44,5% del GPU·tiempo. El 13,5% lo mató el usuario y el 17,2% falló. Juntos son el 30,7% de los trabajos y se llevaron alrededor del 55% del GPU·tiempo. Que el usuario lo mate no es necesariamente malo: muchos entrenan hasta ver que la curva no mejora, o hacen búsqueda de hiperparámetros y matan las configuraciones malas.
¿Por qué fallan? Mayormente errores del usuario: la causa más frecuente es quedarse sin memoria en la CPU, después entradas incorrectas y errores semánticos en el código. Dos fallas de infraestructura, error al guardar el checkpoint y falla del runtime de MPI, son sólo el 6,2% de los intentos, pero aparecen tarde y suman el 36% del tiempo hasta fallar. Y los trabajos de más de 4 GPUs reintentan más y fallan más.
La tercera recomendación del paper: muchas fallas deberían atraparse temprano, antes de que el trabajo pida un clúster grande. Por ejemplo, correr primero un rato en una GPU.
Conectar con la Clase 17: allá las fallas eran del hardware, una cada tres horas en 16.000 GPUs. En un clúster compartido, la mayoría las causa el usuario.
Pasar a la siguiente: el primer problema de planificar, todo o nada.""",
    d.footer("Jeon et al., ATC 2019, §4, tablas 6 y 7"), gap=48)

# 6 ─ gang
A, B = "acc", "warn"
d.slide("gang", D, head("Todo o nada · Clase 16", "Un trabajo distribuido necesita todas sus GPUs a la vez", D) +
    row(col(label("De a un pod: 8 GPUs, dos trabajos de 8", D),
            cells([A] * 4 + [B] * 4, D, size=72, gap=10, labels=["A"] * 4 + ["B"] * 4),
            p("A retiene 4 y espera las 4 de B. B retiene 4 y espera las 4 de A.", D, size=28, color="ink"),
            p("Exclusión mutua, retener y esperar, sin desalojo, espera circular: las cuatro condiciones de Coffman.", D, size=28), gap=20, flex=1),
        card(label("Gang scheduling", D, color="acc") + p("Se piden todas las GPUs juntas o ninguna: rompe <b>retener y esperar</b>. Es la misma idea que pedir todos los locks de una vez.", D, size=28) +
             p("Philly: «el planificador tiene que hacer gang scheduling y respetar la localidad».", D, size=24, color="muted"), D, accent=True, width=600, pad=40), gap=64, align="center"),
    """Tiempo: 0:16–0:20 (4 min).
Idea clave: si un trabajo sincrónico recibe sólo parte de sus GPUs, no avanza; si dos lo hacen a la vez, se traban para siempre.
Decir: en la Clase 16 lo vimos del lado del entrenamiento: el all-reduce es una barrera, y con 500 de 512 GPUs el trabajo no da ni un paso. Hoy, del lado del planificador. Un orquestador como Kubernetes, por defecto, ubica cada pod por su cuenta. Si llegan dos trabajos de 8 GPUs a un clúster de 8, cada pod que se ubica toma una GPU, y puede terminar así: A con 4 y B con 4. Ninguno puede arrancar, ninguno suelta lo que tiene. Deadlock.
Repasar las cuatro condiciones de Coffman (1971): exclusión mutua (una GPU es de un trabajo), retener y esperar (cada uno retiene lo que tiene mientras espera lo demás), sin desalojo (nadie le saca las GPUs) y espera circular. Basta con romper una.
Gang scheduling rompe retener y esperar: el trabajo se admite entero o se queda en la cola sin tomar nada. Es lo mismo que pedir todos los locks de una transacción de una vez en lugar de a medida que se necesitan. Philly lo dice textual: los frameworks necesitan todas las GPUs a la vez, así que el planificador hace gang scheduling y además trata de respetar la localidad.
Demo: go run ./cmd/sim traba es exactamente este dibujo.
Pasar a la siguiente: qué pasa en un clúster entero.""",
    d.footer("Jeon et al., §2.3 · Coffman et al., 1971 · go run ./cmd/sim traba"), gap=40, section="Todo o nada: gang scheduling, bloqueo de cabeza de cola y backfill.")

# 7 ─ demo gang (claro)
d.slide("demogang", L, head("Demo · gang", "Sin gang se desperdicia; con FIFO estricto, también", L) +
    table(["Planificador", "JCT prom.", "Espera 1 GPU", "Espera 2–8", "Espera >8", "Uso"], [
        ["De a un pod, sin gang", "29,7 h", "24,5 h", "26,2 h", "27,2 h", "72%"],
        ["FIFO, gang", "10,0 h", "6,4 h", "6,4 h", "6,6 h", "68%"],
        ["FIFO + backfill, gang", "3,8 h", "14 s", "8 min", "3,2 h", "74%"],
    ], L, widths=[30, 14, 15, 14, 14, 13], right=(1, 2, 3, 4, 5), size=30, bold_rows=(2,)) +
    p("2.000 trabajos en 256 GPUs (4 racks de 8 servidores de 8), al 90% de carga. Sin gang hubo 6.000 GPU·horas retenidas por trabajos que no podían arrancar.", L, size=24, color="muted"),
    """Tiempo: 0:20–0:24 (4 min, con demo).
Idea clave: sin gang, las GPUs retenidas sin uso multiplican la espera de todos; con gang y FIFO estricto, el problema pasa a ser la cola.
Demo: go run ./cmd/sim gang, y en el simulador, poner "De a un pod" y mirar las GPUs rayadas: retenidas por trabajos que todavía no tienen todas.
Leer la tabla: la misma carga con tres planificadores. De a un pod no llega a trabarse del todo (siempre termina algún trabajo chico y libera algo), pero las GPUs tomadas por trabajos que esperan a las que les faltan suman unas 6.000 GPU·horas sin hacer nada, y todos esperan alrededor de un día.
FIFO con gang ya no retiene nada, pero la espera promedio sigue siendo de más de 6 horas para todos, incluso para los de 1 GPU. ¿Por qué? Porque el uso es de 68%: hay GPUs libres y nadie las usa. La próxima slide lo explica.
FIFO con backfill: el JCT (tiempo de completar un trabajo, desde que llega hasta que termina) baja de 10 a 3,8 horas, los de 1 GPU esperan segundos. Pero mirar la última columna de espera: los de más de 8 GPUs siguen esperando más de 3 horas.
Pasar a la siguiente: por qué FIFO deja GPUs libres.""",
    d.footer("go run ./cmd/sim gang · " + SIM, L, plain=True), gap=40)

# 8 ─ cabeza de cola
q = [("32", "warn"), ("1", "blue"), ("1", "blue"), ("4", "acc"), ("1", "blue"), ("2", "acc")]
qhtml = "".join(f'<div style="height:72px;padding:0 24px;border-radius:10px;background:{D[c]};display:flex;align-items:center;{"outline:3px solid " + D["ink"] + ";outline-offset:3px;" if i == 0 else ""}">'
                f'<p style="font-family:{MONO};font-size:32px;font-weight:600;color:{D["onfill"]}">{n}</p></div>' for i, (n, c) in enumerate(q))
d.slide("cabeza", D, head("Bloqueo de cabeza de cola", "El primero no entra, y nadie le pasa por adelante", D) +
    col(label("La cola, en orden de llegada: GPUs que pide cada trabajo", D), f'<div style="display:flex;gap:16px;align-items:center">{qhtml}</div>', gap=16) +
    col(label("El clúster: 20 GPUs libres (punteadas), todas sin usar", D), cells(["muted"] * 36 + [None] * 20, D, size=22, gap=4), gap=16) +
    row(card(label("FIFO", D) + p("El de 32 espera, y los de 1 también: hay lugar para ellos y no lo usan.", D, size=28), D, pad=32),
        card(label("Backfill", D, color="acc") + p("Si el primero no entra, pasan los que entran. En Philly, el 38,1% de las decisiones fue fuera de orden.", D, size=28), D, accent=True, pad=32), gap=32),
    """Tiempo: 0:24–0:27 (3 min).
Idea clave: con gang scheduling y FIFO estricto, un trabajo grande al frente de la cola deja GPUs libres que los chicos de atrás podrían usar; backfill los deja pasar.
Decir: la cola tiene al frente un trabajo de 32 GPUs, y quedan 20 libres. Con FIFO estricto, el de 32 espera a que se liberen 12 más, y los de 1, 1, 4, 1 y 2 de atrás también esperan, aunque entran de sobra. Es el bloqueo de cabeza de cola (head-of-line blocking), el mismo de las colas de red o de un supermercado con una sola caja.
Backfill: si el primero no entra, se prueba con los siguientes y entran los que caben. Philly lo hace: el 38,1% de las decisiones de su planificador fueron fuera de orden, y todos los trabajos de más de 4 GPUs vieron pasar a otros adelante; en el 85% de esos casos, sin alargar su espera. El análisis de Alibaba de 2022 encuentra el mismo problema: alrededor del 9% de las instancias cortas pasó más de la mitad de su tiempo esperando.
Pasar a la siguiente: el precio de dejar pasar.""",
    d.footer("Jeon et al., §3.1.1 · Weng et al., NSDI 2022, §1"), gap=40)

# 9 ─ hambre
d.slide("hambre", D, head("El precio de dejar pasar", "Con backfill, los grandes pueden esperar para siempre", D) +
    row(col(p("Mientras el de 32 espera que se liberen 12 GPUs, cada vez que se libera una entra un trabajo de 1. El de 32 nunca junta las suyas.", D, size=30),
            p("En la demo: los de 1 GPU esperan <b>14 segundos</b>; los de más de 8, <b>3,2 horas</b>.", D, size=30, color="ink"), gap=28, flex=1),
        col(card(label("EASY backfilling · HPC, 1995", D, color="acc") + p("El primero reserva la hora en que va a entrar; sólo pasan los que terminan antes. Necesita saber cuánto dura cada trabajo.", D, size=28), D, accent=True, pad=36),
            card(label("Philly", D) + p("Backfill, y preempción entre equipos sólo cuando el clúster pasa el 90% de uso.", D, size=28), D, pad=36), gap=24, width=680), gap=56, align="center"),
    """Tiempo: 0:27–0:30 (3 min).
Idea clave: el backfill codicioso cambia el bloqueo de los chicos por el hambre de los grandes; la solución clásica, reservar, necesita conocer las duraciones.
Decir: el problema del backfill codicioso. El de 32 necesita que se liberen 12 GPUs a la vez. Pero cada vez que termina un trabajo de 1 GPU, en lugar de quedar libre esa GPU para el de 32, la toma otro de 1 que estaba atrás. El de 32 puede esperar indefinidamente: hambre (starvation). En la demo: 14 segundos para los de 1 GPU, más de 3 horas para los de más de 8.
La respuesta clásica viene de los clústeres de HPC: EASY backfilling (Lifka, 1995, en Argonne). El primero de la cola reserva la hora más temprana en que va a entrar, calculada con las duraciones de los que corren, y sólo pasan los que terminan antes de esa hora. Problema: necesita que cada trabajo diga cuánto dura, y en HPC el usuario la declara. En deep learning, como vimos, nadie la sabe. Es el ejercicio 1 del repositorio.
Philly hace backfill y, entre clústeres virtuales, preempción para devolver la cuota a su dueño, pero sólo cuando el clúster pasa el 90% de uso.
Pasar a la siguiente: el segundo problema, dónde.""",
    d.footer("Jeon et al., §2.3 y §3.1.1 · Lifka, 1995 · go run ./cmd/sim gang"), gap=40)

# 10 ─ localidad: por qué
d.slide("localidad", D, head("Dónde corre importa · Clase 16", "Repartido en más servidores, el mismo trabajo usa peor las GPUs", D) +
    row(col(label("Trabajos de 16 GPUs en Philly", D),
            table(["Repartido en", "Uso promedio de las GPUs"], [["2 servidores", "43,7%"], ["4 servidores", "40,9%"], ["8 servidores", "28,6%"]], D, widths=[50, 50], right=(1,), size=30),
            p("Jeon et al., tabla 5", D, size=24, color="muted"), gap=16, flex=1),
        col(card(p("Dentro de un servidor, el all-reduce va por NVLink. Entre servidores, por la red: cada paso espera a la conexión más lenta.", D, size=28, color="ink"), D, pad=36),
            p("Y los que comparten servidor con trabajos ajenos se interfieren: CPU, memoria, PCIe, red.", D, size=28), gap=24, width=700), gap=64, align="center"),
    """Tiempo: 0:30–0:34 (4 min).
Idea clave: la misma cantidad de GPUs no vale lo mismo en cualquier lado; un trabajo distribuido repartido en más servidores pasa más tiempo comunicándose.
Decir: en la Clase 16 calculamos el ring all-reduce: cada GPU manda unas 2 veces el tamaño del gradiente por paso, y el anillo va a la velocidad del enlace más lento. Dentro de un servidor, NVLink; entre servidores, la red, que es un orden de magnitud más lenta. Por eso el paralelismo de tensores va dentro del servidor.
Lo que mide Philly, en la tabla 5: trabajos de 16 GPUs. Repartidos en 2 servidores, sus GPUs se usan en promedio un 43,7%; en 4, un 40,9%; en 8 servidores, un 28,6%. El resto del tiempo las GPUs esperan. Y comparando tamaños: los trabajos de 8 GPUs en un servidor usan sus GPUs un 56,9% en promedio; los de 16, un 34,3%.
Además está la interferencia: si un servidor tiene GPUs de dos trabajos distintos, se pelean por la CPU, la memoria del host, el bus PCIe y la placa de red. La segunda recomendación del paper es aislar los trabajos en servidores dedicados.
Pasar a la siguiente: entonces, ¿esperar el lugar bueno?""",
    d.footer("Jeon et al., §3.2.1, tabla 5"), gap=40, section="Localidad: esperar el lugar bueno o arrancar en uno peor, y la fragmentación.")

# 11 ─ esperar o repartir
d.slide("esperar", D, head("El dilema de la localidad", "Esperar el lugar bueno o arrancar en uno peor", D) +
    row(col(label("Philly · clúster virtual 2", D), big("25%", "de los trabajos de más de 4 GPUs esperó 10 minutos o más", D, size=88),
            big("10%", "de los de 1 GPU", D, size=88, color="blue"), gap=32, width=560),
        col(label("Por qué esperan · tabla 2", D),
            table(["GPUs", "Por la cuota", "Por fragmentación"], [["2 a 4", "40,6%", "59,4%"], ["5 a 8", "25,8%", "74,2%"], ["más de 8", "2,1%", "97,9%"]], D, widths=[34, 33, 33], right=(1, 2), size=28),
            p("Fragmentación: hay GPUs libres, pero no juntas.", D, size=28, color="ink"), gap=16, flex=1), gap=72, align="center"),
    """Tiempo: 0:34–0:38 (4 min).
Idea clave: los trabajos grandes esperan más, y casi siempre no porque falten GPUs, sino porque las libres no están juntas.
Decir: en el clúster virtual 2 de Philly, el 25% de los trabajos de más de 4 GPUs esperó al menos 10 minutos para arrancar; de los de 1 GPU, sólo el 10%.
El paper separa dos causas de espera. Por la cuota (fair-share): el equipo ya usa todo lo que le toca. Por fragmentación: hay GPUs libres de sobra, pero no en un servidor o en un rack, que es lo que pide la localidad. La tabla 2 cuenta las ocurrencias: para los trabajos de 5 a 8 GPUs, el 74,2% de las esperas son por fragmentación; para los de más de 8, el 97,9%. Medido en tiempo, alrededor del 80% de la espera de todos los trabajos es por fragmentación.
Y cuánto: cuando el clúster está a dos tercios de uso, menos del 4,5% de los servidores está completamente vacío.
Cómo lo maneja Philly: si no hay lugar con buena localidad, el trabajo espera unos minutos (2 a 3), reintenta, y después de un número fijo de reintentos relaja la localidad y acepta servidores repartidos. La figura 4 muestra el efecto: los de más de 8 GPUs repartidos en 16 servidores arrancan mucho antes que los que esperaron 2 o 4. Pero, como vimos, corren peor.
Pasar a la siguiente: un dibujo de la fragmentación.""",
    d.footer("Jeon et al., §2.3 y §3.1.1, tabla 2"), gap=40)

# 12 ─ fragmentación
U, F = "muted", None
d.slide("fragmentacion", D, head("Fragmentación", "Ocho GPUs libres y un trabajo de 8 que no entra", D) +
    row(server([U] * 6 + [F] * 2, D, 52, "servidor 1 · 2 libres"), server([U] * 6 + [F] * 2, D, 52, "servidor 2 · 2 libres"), gap=40) +
    row(server([U] * 6 + [F] * 2, D, 52, "servidor 3 · 2 libres"), server([U] * 6 + [F] * 2, D, 52, "servidor 4 · 2 libres"), gap=40) +
    row(p("Es la fragmentación externa de la memoria, con servidores en lugar de páginas. Best-fit ayuda: el trabajo chico va al servidor que menos lugar libre deja.", D, size=28, width=1000),
        card(p("Probalo: «Consolidada siempre» en el simulador, y mirá la cola.", D, size=24, color="ink"), D, accent=True, pad=24, width=560), gap=56, align="center"),
    """Tiempo: 0:38–0:41 (3 min).
Idea clave: con los trabajos chicos desparramados, el clúster tiene GPUs libres de sobra y ningún lugar para uno grande.
Decir: cuatro servidores de 8 GPUs, cada uno con 6 ocupadas. Hay 8 libres: un trabajo de 8 GPUs entraría por cantidad. Pero si se exige que vaya en un servidor, no entra en ninguno. Para el planificador, la cola del de 8 es por fragmentación.
Es la fragmentación externa de la Clase de memoria de sistemas operativos: hay memoria libre suficiente pero no contigua. Y las mismas soluciones: ubicar con best-fit (el trabajo de 1 GPU va al servidor más lleno donde entra, para no romper uno vacío), y compactar, que acá es migrar trabajos, lo que hace Gandiva.
Demo: en el simulador, con "Consolidada siempre" y carga alta, la cola dice cuándo el primero espera por fragmentación: "pide 8 GPUs y hay 20 libres, pero no en un mismo servidor". Los servidores vacíos tienen borde punteado: mirar cuán pocos hay.
Pasar a la siguiente: qué propone Tiresias para la localidad.""",
    d.footer(SIM + " · Jeon et al., §3.1.1"), gap=28)

# 13 ─ tiresias consolidar
d.slide("consolidar", D, head("Tiresias · NSDI 2019 · §2.2", "Consolidar sólo a los trabajos a los que les importa", D) +
    row(col(table(["Modelo", "Su tensor más grande, sobre el total"], [["VGG19", "71,5%"], ["AlexNet", "61,0%"], ["ResNet-50", "9,2%"]], D, widths=[40, 60], right=(1,), size=30),
            p("Gu et al., tabla 6", D, size=24, color="muted"), gap=16, flex=1),
        col(p("Los modelos con un tensor enorme mandan un mensaje enorme en cada paso: son los que sufren si se reparten.", D, size=28),
            p("Tiresias los detecta mirando el modelo, sin preguntarle al usuario, y sólo a ellos los consolida. Hasta <b>1,67×</b> contra ubicar al azar.", D, size=28, color="ink"), gap=24, width=760), gap=64, align="center"),
    """Tiempo: 0:41–0:44 (3 min).
Idea clave: no todos los trabajos sufren igual al repartirse; si se puede saber cuáles sufren, sólo esos tienen que esperar el lugar bueno.
Decir: Tiresias, de la Universidad de Michigan con Microsoft, usa la misma traza de Philly. Su observación: la sensibilidad a la localidad depende de cómo están repartidos los parámetros del modelo en tensores. En VGG19, el tensor más grande es el 71,5% del modelo; en AlexNet, el 61%; en ResNet-50, el 9,2%. Cada tensor viaja como un mensaje en la agregación de gradientes, así que un modelo con un tensor gigante manda un mensaje gigante, y en una red lenta ese mensaje domina. Los modelos con muchos tensores chicos superponen mejor comunicación y cómputo.
Tiresias perfila el modelo cuando arranca, sin preguntarle nada al usuario, y sólo consolida a los sensibles; el resto va donde haya lugar. Contra ubicar al azar, la mejora en tiempo de entrenamiento llega a 1,67 veces.
Conectar con la Clase 16: es la cuenta del all-reduce, pero la constante depende del modelo.
Pasar a la siguiente: la demo de localidad.""",
    d.footer("Gu et al., NSDI 2019, §2.2, §5.2.3 y tabla 6"), gap=40)

# 14 ─ demo localidad (claro)
d.slide("demolocalidad", L, head("Demo · localidad", "Como corren horas, esperar el lugar bueno conviene", L) +
    table(["Ubicación", "JCT prom.", "Espera >8", "… por fragmentación", "Más lento", "Uso útil"], [
        ["Consolidada siempre", "3,82 h", "3,6 h", "39%", "1,00×", "73%"],
        ["Donde haya lugar", "4,00 h", "2,3 h", "0%", "1,11×", "71%"],
        ["Relajar tras 10 min (Philly)", "4,15 h", "3,3 h", "0%", "1,12×", "71%"],
        ["Consolidar si importa (Tiresias)", "3,82 h", "3,2 h", "15%", "1,01×", "73%"],
    ], L, widths=[33, 13, 13, 17, 12, 12], right=(1, 2, 3, 4, 5), size=28, bold_rows=(3,)) +
    p("FIFO + backfill. Más lento: tiempo corriendo sobre el ideal, trabajos de varias GPUs. La mitad es sensible: repartida, corre al 60%.", L, size=24, color="muted"),
    """Tiempo: 0:44–0:48 (4 min, con demo).
Idea clave: repartir acorta la espera pero alarga la ejecución, y con trabajos de horas pierde; consolidar sólo a los sensibles da el JCT de consolidar siempre con menos espera.
Demo: go run ./cmd/sim localidad, y en el simulador, cambiar la ubicación con FIFO + backfill a carga 100%.
Leer la tabla. Consolidada siempre: los trabajos de más de 8 esperan 3,6 horas, y el 39% de esa espera es con GPUs libres de sobra pero repartidas: fragmentación, como la tabla 2 de Philly. Donde haya lugar: esperan 2,3 horas, nada por fragmentación, pero corren en promedio 11% más lento, y el JCT promedio empeora. Relajar después de 10 minutos, lo que hace Philly: en este modelo, lo peor de los dos. Consolidar sólo si importa: el mismo JCT que consolidar siempre, y los grandes esperan menos.
Es la primera recomendación de Jeon et al.: como los trabajos corren mucho, el planificador debería cambiar espera por localidad.
Lo honesto: los números dependen del modelo de velocidad (60% repartido, ilustrativo) y de la carga; en el simulador, con otra muestra, consolidada siempre y consolidar si importa a veces se invierten. Lo que no cambia es que repartir a todos pierde.
Pasar a la siguiente: el tercer problema, el orden.""",
    d.footer("go run ./cmd/sim localidad · " + SIM, L, plain=True), gap=40)

# 15 ─ srtf vs las
d.slide("las", D, head("Sin conocer las duraciones", "Si no sabés cuánto le falta, mirá cuánto ya recibió", D) +
    row(card(label("SRTF", D) + p("Primero el que menos tiempo le falta. Óptimo para el tiempo de respuesta promedio.", D, size=28) + p("Necesita la duración: nadie la sabe.", D, size=28, color="warn"), D, pad=36),
        card(label("LAS", D, color="acc") + p("Primero el que <b>menos servicio recibió</b>. Con colas pesadas, el que ya corrió mucho probablemente corra mucho más.", D, size=28), D, accent=True, pad=36),
        card(label("2D-LAS · Tiresias", D, color="acc") + p("El servicio en dos dimensiones: <b>GPUs × tiempo</b>. Un trabajo de 32 GPUs que corrió una hora recibió 32 GPU·horas.", D, size=28), D, accent=True, pad=36), gap=32) +
    p("prioridad = − GPUs × tiempo corriendo · Gu et al., pseudocódigo 1", D, size=32, color="ink", mono=True),
    """Tiempo: 0:48–0:51 (3 min).
Idea clave: LAS aproxima a SRTF sin conocer duraciones porque, con duraciones de cola pesada, haber corrido mucho predice que falta mucho.
Decir: de sistemas operativos: si se sabe cuánto le falta a cada trabajo, atender primero al que menos le falta (SRTF, shortest remaining time first) minimiza el tiempo de respuesta promedio. Pero vimos que nadie sabe cuánto dura su trabajo, y casi un tercio ni termina.
LAS, least attained service: atender primero al que menos servicio recibió hasta ahora. Es un dato que el planificador sí tiene. ¿Por qué funciona? Por la cola pesada: un trabajo que lleva 5 minutos probablemente sea de los muchos cortos; uno que lleva 3 días, de los pocos que duran semanas. Sin saber nada, LAS deja que los cortos terminen rápido y manda a los largos para atrás. Es el clásico foreground-background de los sistemas de tiempo compartido.
La vuelta de Tiresias: el servicio tiene dos dimensiones. Un trabajo de 32 GPUs que corrió una hora le costó al clúster lo mismo que uno de 1 GPU que corrió 32 horas. La prioridad es menos GPUs por tiempo corriendo: el que menos GPU·tiempo recibió va primero.
Pasar a la siguiente: cómo se implementa sin desalojar todo el tiempo.""",
    d.footer("Gu et al., NSDI 2019, §3.2.2"), gap=40, section="Sin conocer las duraciones: 2D-LAS, preempción y Tiresias.")

# 16 ─ colas
def lane(name, sub, items, t=D, accent=False):
    chips_ = "".join(chip(x, t, color=c) for x, c in items)
    return (f'<div style="display:flex;align-items:center;gap:32px;padding:24px 32px;border-radius:14px;border:{"2px solid " + t["acc"] if accent else "1px solid " + t["line"]};background:{t["card"]}">'
            f'<div style="width:420px;display:flex;flex-direction:column;gap:6px">{h3(name, t, size=32)}{p(sub, t, size=24)}</div>'
            f'<div style="display:flex;gap:12px">{chips_}</div></div>')
d.slide("colas", D, head("Tiresias · §3.2.3", "Dos colas alcanzan", D) +
    col(lane("Cola 1 · alta prioridad", "recibieron menos que el umbral", [("1 GPU · 5 min", "blue"), ("8 GPUs · 2 min", "acc"), ("32 GPUs · 1 min", "warn")], accent=True),
        f'<div style="display:flex;align-items:center;gap:16px;padding-left:32px">{p("↓", D, size=40, color="muted")}{p("baja al pasar el umbral de GPU·tiempo: 3.200 GPU·s en su testbed", D, size=28)}</div>',
        lane("Cola 2 · baja prioridad", "dentro de cada cola, FIFO por hora de arranque", [("1 GPU · 3 h", "muted"), ("16 GPUs · 1 h", "muted")]), gap=16) +
    p("Con 3 o 4 colas, el JCT mejora sólo un 1%. Contra el hambre: si un trabajo esperó PROMOTEKNOB veces lo que corrió, vuelve a la cola 1.", D, size=28),
    """Tiempo: 0:51–0:54 (3 min).
Idea clave: LAS puro desalojaría todo el tiempo; Tiresias lo discretiza en pocas colas, y con dos ya alcanza.
Decir: LAS al pie de la letra cambia de prioridad continuamente: cada vez que un trabajo recibe un poco de servicio, otro lo pasa, y habría que desalojar a cada rato. Con GPUs, desalojar es caro (próxima slide). Tiresias discretiza: K colas con umbrales de GPU·tiempo; un trabajo empieza en la cola 1 y baja cuando su GPU·tiempo pasa el umbral. Entre colas, prioridad estricta y preempción; dentro de cada cola, FIFO.
El detalle importante: FIFO por la hora en que arrancó por primera vez, no por la hora de llegada. Si fuera por llegada, un trabajo que llegó antes pero todavía no pudo arrancar desalojaría a uno que ya está corriendo, sin ganar nada.
¿Cuántas colas? El paper dice que con K = 2 se anda casi igual que con más: con 3 o 4 colas el JCT promedio mejora sólo un 1%. En su testbed usan 2 colas con un umbral de 3.200 GPU·segundos.
El hambre: los que bajan a la cola 2 pueden no volver a correr nunca si siguen llegando trabajos nuevos. Tiresias tiene un parámetro, PROMOTEKNOB: si un trabajo lleva esperando PROMOTEKNOB veces lo que corrió, vuelve a la cola 1. Es el aging de los planificadores de procesos. En el testbed lo apagaron.
Pasar a la siguiente: cuánto cuesta desalojar.""",
    d.footer("Gu et al., NSDI 2019, §3.2.3, §5.2 y §5.4.2"), gap=32)

# 17 ─ preempción
d.slide("preempcion", D, head("Desalojar cuesta un checkpoint · Clases 16 y 17", "Sacarle las GPUs a un trabajo es guardarlo y volver a cargarlo", D) +
    row(card(label("Tiresias · testbed", D) + p("62 s", D, size=88, color="warn", mono=True, weight=600, lh=1) + p("por preempción: 13.724 s en 221 preempciones, guardando el checkpoint", D, size=24), D, pad=36),
        card(label("Gandiva · suspender entre pasos", D) + p("&lt; 100 ms", D, size=88, color="acc", mono=True, weight=600, lh=1) + p("para clasificación de imágenes; hasta 1 s para traducción", D, size=24), D, accent=True, pad=36),
        card(label("Por qué tanta diferencia", D) + p("Entre pasos, la memoria de la GPU baja a casi nada: Gandiva midió 77× entre el máximo y el mínimo. Suspender ahí es copiar poco.", D, size=28), D, pad=36), gap=32),
    """Tiempo: 0:54–0:57 (3 min).
Idea clave: la preempción no es gratis: es un checkpoint y una restauración, como en las Clases 16 y 17, y su costo decide cuán seguido conviene desalojar.
Decir: para desalojar un trabajo de entrenamiento hay que guardar su estado (pesos, estado del optimizador) y después volver a cargarlo, en las mismas GPUs u otras. Es el checkpoint de las clases 16 y 17, con otra razón: allá era por fallas, acá por prioridad.
Tiresias mide el costo en su testbed: 221 preempciones costaron 13.724 segundos en total, unos 62 segundos cada una. Por eso no se puede desalojar a cada rato, y por eso dos colas.
Gandiva (OSDI 2018, también de Microsoft) lo hace mucho más barato aprovechando la estructura del entrenamiento: la memoria de la GPU sube y baja en cada mini-batch, con una diferencia de hasta 77 veces entre el máximo y el mínimo, porque las activaciones se liberan al terminar el paso. Si se suspende justo al final de un paso, hay muy poco para copiar: menos de 100 ms para clasificación de imágenes, hasta un segundo para modelos de traducción. Con un turno de un minuto, eso es un 2% o menos.
Conectar: el checkpoint asincrónico de la Clase 17 es la misma idea: elegir el momento barato.
Pasar a la siguiente: los resultados de Tiresias.""",
    d.footer("Gu et al., §5.2.4 · Xiao et al., OSDI 2018, §3.3 y §4.1"), gap=40)

# 18 ─ resultados tiresias (claro)
d.slide("tiresias", L, col(eyebrow("Resultados · Tiresias, NSDI 2019", L), h2("Sin saber duraciones, como si las supiera", L), gap=16) +
    row(big("5,5×", "mejor JCT promedio que YARN-CS, en 60 GPUs P100", L), big("27×", "mejor JCT mediano", L, color="blue"),
        big("8.000 → 1.000 s", "espera promedio en la cola", L, color="warn", size=72), gap=72) +
    p("En simulación, con la traza de Philly: 2,4× contra YARN-CS, 1,5× contra best-effort y 2× contra Gandiva, y el mismo JCT promedio que SRTF, que conoce las duraciones.", L, size=28),
    """Tiempo: 0:57–1:00 (3 min).
Idea clave: con dos colas de 2D-LAS y consolidando sólo a los sensibles, Tiresias logra lo que lograría SRTF sabiendo las duraciones.
Decir: el testbed: 60 GPUs P100 en 15 máquinas de 4. Contra YARN-CS, un planificador de capacidad de YARN como el que usaba Microsoft (FIFO con gang), el JCT promedio mejora 5,5 veces y el mediano 27 veces: la mayoría de los trabajos, que son cortos, deja de esperar detrás de los largos. La espera promedio en la cola baja de más de 8.000 segundos a unos 1.000; la mitad de los trabajos espera 13 segundos o menos. El makespan, el tiempo hasta que termina el último, también baja, 1,21 veces.
En simulación con la traza de Philly, la tabla 5: 2,4 veces mejor que YARN-CS, 1,5 contra best-effort (FIFO dejando pasar a los que entran, nuestro backfill) y 2 contra Gandiva. Y el JCT promedio queda igual al de SRTF, que conoce las duraciones.
Remarcar: la ganancia grande es contra FIFO estricto; contra backfill es más chica. Lo vamos a ver en la demo.
Pasar a la siguiente: la demo.""",
    d.footer("Gu et al., NSDI 2019, §5.2 y tabla 5", L), gap=64)

# 19 ─ demo duraciones (claro)
d.slide("demoduraciones", L, head("Demo · sin conocer las duraciones", "2D-LAS queda a un 1% de SRTF, sin saber nada", L) +
    row(col(table(["Planificador", "JCT prom.", "Espera 1", "Espera >8", "Preempc."], [
        ["FIFO", "10,03 h", "6,4 h", "6,6 h", "0"],
        ["FIFO + backfill", "3,82 h", "14 s", "3,2 h", "0"],
        ["2D-LAS, umbral 3.200 GPU·s", "4,12 h", "29 min", "24 s", "2.642"],
        ["2D-LAS, umbral 300.000 GPU·s", "3,70 h", "2 s", "23 min", "569"],
        ["SRTF (conoce las duraciones)", "3,67 h", "0 s", "47 min", "1.386"],
    ], L, widths=[40, 16, 14, 15, 15], right=(1, 2, 3, 4), size=26, bold_rows=(3,)), gap=12, flex=1),
        col(label("Costo de cada preempción", L), table(["", "JCT"], [["gratis", "3,70 h"], ["1 min", "3,70 h"], ["10 min", "3,76 h"], ["30 min", "3,91 h"]], L, widths=[50, 50], right=(1,), size=26), gap=12, width=360), gap=48, align="start"),
    """Tiempo: 1:00–1:04 (4 min, con demo).
Idea clave: con un umbral adecuado a la carga, 2D-LAS le gana al backfill y casi empata a SRTF; con un umbral malo, desaloja de más y pierde.
Demo: go run ./cmd/sim duraciones, y en el simulador, cambiar entre FIFO + backfill, 2D-LAS y SRTF y apretar Comparar.
Leer la tabla, la misma carga. FIFO: 10 horas de JCT promedio. Backfill: 3,82. 2D-LAS con el umbral de Tiresias, 3.200 GPU·segundos: 4,12, peor que backfill. ¿Por qué? Porque en esta carga los trabajos duran horas: con 3.200 GPU·s un trabajo de 32 GPUs baja de cola a los 100 segundos y uno de 1 GPU a los 53 minutos; casi todo está en la cola 2 y se desaloja 2.642 veces. Con un umbral de 300.000 GPU·s (unas 80 GPU·horas): 3,70 horas, a un 1% de SRTF (3,67), con 569 preempciones. El umbral se elige mirando la carga.
Mirar la columna de los grandes: con backfill esperan 3,2 horas; con 2D-LAS, 23 minutos. Un trabajo nuevo de 32 GPUs tiene prioridad alta mientras recibió poco, así que no lo pasan los chicos. LAS también arregla el hambre del backfill.
A la derecha, el costo de cada preempción con el umbral bueno: con 30 minutos por desalojo, el JCT sube a 3,91. Si guardar un checkpoint cuesta mucho, conviene desalojar menos.
Lo honesto: la diferencia contra backfill es chica (3%); la grande es contra FIFO estricto. En la simulación del paper, igual: 2,4 veces contra YARN-CS, que tiene bloqueo de cabeza de cola, y sólo 1,5 contra best-effort, que ya hace backfill.
Pasar a la siguiente: compartir.""",
    d.footer("go run ./cmd/sim duraciones · " + SIM, L, plain=True), gap=36)

# 20 ─ gandiva
d.slide("gandiva", D, head("Gandiva · OSDI 2018", "Repartir las GPUs en el tiempo, no sólo en el espacio", D) +
    row(card(label("Time-slicing", D, color="acc") + p("Varios trabajos se turnan las mismas GPUs, de a un minuto, suspendiendo entre pasos. Todos avanzan, y los nuevos dan su primera señal enseguida.", D, size=28), D, accent=True, pad=36),
        card(label("Empaquetar y migrar", D) + p("Dos trabajos chicos en una GPU si no se molestan; mover un trabajo a otras GPUs para juntar uno grande.", D, size=28), D, pad=36), gap=32) +
    row(big("50,1% → 62,8%", "uso promedio de un clúster de 180 GPUs: 26% más", D, size=72), big("10×", "configuraciones exploradas en una búsqueda de hiperparámetros", D, size=72, color="blue"), gap=96),
    """Tiempo: 1:04–1:07 (3 min).
Idea clave: si suspender es barato, las GPUs se pueden repartir en el tiempo como la CPU, y eso sirve sobre todo para los trabajos exploratorios que hay que ver temprano.
Decir: Gandiva parte de que el entrenamiento es predecible: cada mini-batch se parece al anterior, en tiempo y en memoria. Con eso hace tres cosas. Time-slicing: si llegan más trabajos que GPUs, se turnan, de a un minuto, suspendiendo en el borde de un paso (la slide anterior: menos de un segundo). Así un trabajo nuevo da su primera señal (¿baja la pérdida?) en minutos, en lugar de esperar horas en la cola: recordar que casi un tercio de los trabajos se mata o falla. Empaquetar: dos trabajos chicos en la misma GPU, midiendo si se molestan y deshaciéndolo si sí. Y migrar: mover un trabajo a otras GPUs, en segundos, para desfragmentar o para mejorar la localidad.
Resultados: en un clúster de 180 GPUs, el uso promedio pasa de 50,1% a 62,8%, un 26% más. En búsqueda de hiperparámetros, explora casi 10 veces más configuraciones en el mismo tiempo, porque prueba muchas un rato y deja seguir a las prometedoras.
Diferencia con Tiresias: Gandiva busca equidad y aprovechamiento; Tiresias, JCT promedio. En la simulación de Tiresias, Gandiva queda 2 veces peor en JCT.
Pasar a la siguiente: compartir una GPU.""",
    d.footer("Xiao et al., OSDI 2018, §3–§6"), gap=40, section="Compartir: el tiempo, una GPU, el clúster entre equipos, y hoy.")

# 21 ─ mlaas (claro)
d.slide("mlaas", L, col(eyebrow("MLaaS in the Wild · Alibaba PAI · NSDI 2022", L), h2("La mayoría de los trabajos no necesita una GPU entera", L), gap=16) +
    row(big("0,042", "GPUs: lo que usa la instancia mediana", L), big("0,5", "GPUs: lo que pide la mediana", L, color="blue"),
        big("50%", "de las GPUs alcanzan si se comparten; hasta 73% de ahorro en la hora pico", L, color="warn"), gap=72) +
    p("6.742 GPUs, julio y agosto de 2020, 1,2 millones de tareas. Y al revés: reservar las máquinas buenas para los trabajos grandes bajó su espera un 68%.", L, size=28),
    """Tiempo: 1:07–1:10 (3 min).
Idea clave: en un clúster de inferencia y entrenamiento mezclados, la mayoría de las instancias usa una fracción mínima de una GPU; compartirlas ahorra la mitad del clúster.
Decir: Alibaba publicó dos meses de su plataforma de machine learning, PAI: 6.742 GPUs en 1.295 máquinas de 2 GPUs y 519 de 8, más de 1,2 millones de tareas y 7,5 millones de instancias de más de 1.300 usuarios. Mucho más heterogéneo que Philly: entrenamiento, inferencia, notebooks.
La instancia mediana usa 0,042 GPUs, medido en multiprocesadores de la GPU, y la mediana pide 0,5. Asignar GPUs enteras sería tirar casi todo. Con GPU sharing (varias instancias en una GPU, con su límite de memoria), en promedio alcanza con el 50% de las GPUs, y en la hora pico, cerca de las 10 de la mañana, se ahorra hasta el 73%. La interferencia es rara: de las GPUs muy usadas, sólo un 4,5% tiene varias instancias.
La otra mitad del paper: los trabajos grandes con requisitos exigentes (las GPUs nuevas, muchas juntas) se mezclaban con los chicos. Reservar las máquinas buenas para ellos y empaquetar a los chicos en las otras bajó la espera de los grandes un 68% y la de todos un 45%. Es la misma idea que consolidar si importa.
Hoy esto existe en el hardware: las GPUs de NVIDIA se pueden partir en instancias aisladas (MIG) o compartir por tiempo.
Pasar a la siguiente: compartir entre equipos.""",
    d.footer("Weng et al., NSDI 2022, §1, §3 y §5", L), gap=64)

# 22 ─ cuotas
d.slide("cuotas", D, head("Entre equipos", "Cada equipo tiene su cuota, y lo que no usa se presta", D) +
    row(card(label("Philly · clústeres virtuales", D) + p("Una cuota por equipo. Lo que un equipo no usa se presta; si el dueño lo pide, se recupera con preempción, sólo con el clúster por encima del 90%.", D, size=28), D, pad=36),
        card(label("Kueue · Kubernetes", D, color="acc") + p("Una cola por equipo con su cuota, y grupos de colas que se prestan entre sí. <b>waitForPodsReady</b>: si no arrancan todos los pods a tiempo, el trabajo se desaloja y vuelve a la cola.", D, size=28), D, accent=True, pad=36),
        card(label("Volcano · Kubernetes", D) + p("Un scheduler con gang scheduling: el grupo de pods se ubica entero o no se ubica.", D, size=28), D, pad=36), gap=32) +
    p("La espera por cuota es la otra columna de la tabla 2 de Philly: para los trabajos de 2 a 4 GPUs, el 40,6% de las esperas.", D, size=28),
    """Tiempo: 1:10–1:14 (4 min).
Idea clave: la equidad entre equipos se resuelve con cuotas que se prestan cuando sobran y se recuperan con preempción cuando el dueño las necesita.
Decir: Philly tiene 14 clústeres virtuales, uno por equipo, cada uno con su cuota. Si un equipo no usa la suya, el planificador se la presta a otro que tiene trabajos esperando, para no dejar GPUs ociosas. Cuando el dueño la necesita, se recupera desalojando, pero sólo si el clúster está por encima del 90% de uso. La espera por cuota (fair-share) es la otra columna de la tabla 2: para los trabajos chicos, de 2 a 4 GPUs, el 40,6% de las esperas.
Hoy, en Kubernetes: Kueue (un proyecto de la comunidad de Kubernetes) hace admisión por cuotas: cada equipo tiene una ClusterQueue con su cuota, y las colas de un mismo cohort se prestan lo que no usan. Y resuelve el todo o nada con waitForPodsReady: espera a que todos los pods del trabajo estén corriendo, y si no lo logran en un tiempo configurado, desaloja el trabajo y lo vuelve a encolar. La propia documentación lo llama una implementación simple de todo o nada: no evita retener y esperar, pero lo corta con un timeout, como en la detección de deadlocks. Volcano es un scheduler alternativo con gang scheduling propiamente dicho.
Conectar con la materia: es exactamente el problema de asignar recursos en un sistema multiusuario, con préstamo y preempción.
Pasar a la siguiente: cómo se ve a escala hoy.""",
    d.footer("Jeon et al., §2.3 · Kueue, waitForPodsReady · Volcano"), gap=40)

# 23 ─ hoy
d.slide("hoy", D, head("Hoy · a escala de Meta", "Planificar entre regiones, y entrenar y servir en la misma flota", D) +
    row(col(label("MAST · OSDI 2024", D), big("2,63 → 0,98", "demanda sobre oferta de GPUs en la región más cargada, para los trabajos de alta prioridad", D, size=72),
            p("Decenas de regiones, millones de máquinas, decenas de miles de GPUs. Antes, cada usuario elegía la región a mano; ahora la elige un planificador global.", D, size=28), gap=24, flex=1),
        card(label("Una flota para todo", D, color="warn") + p("El serving tiene picos de día y valles de noche. DeepSeek reduce los nodos de inferencia de noche y usa esas GPUs para investigación y entrenamiento.", D, size=28) +
             p("La próxima clase: entrenar con lo que se sirve.", D, size=28, color="ink"), D, width=640, pad=40), gap=64, align="center"),
    """Tiempo: 1:14–1:17 (3 min).
Idea clave: a escala de hiperescalador, el problema sube un nivel: no sólo qué GPUs, sino en qué región; y entrenar y servir comparten la misma flota.
Decir: MAST es el planificador global de entrenamiento de Meta (OSDI 2024). Su flota: decenas de regiones, millones de máquinas y decenas de miles de GPUs. El problema que encontraron: cada usuario elegía a mano en qué región guardar sus datos y correr su entrenamiento (tienen que estar juntos), y esas decisiones aisladas dejaban algunas regiones sobrecargadas mientras otras sobraban. Con MAST, el usuario manda el trabajo al planificador global, que decide dónde van los datos y el trabajo, y dentro de la región otro planificador elige las máquinas. Sus tres principios: desacoplar en el tiempo, desacoplar el alcance y búsqueda exhaustiva. El número: en la región más cargada, la demanda de GPUs para trabajos de alta prioridad era 2,63 veces la oferta; con MAST, 0,98.
Y la otra tendencia: entrenar y servir en la misma flota. El serving sigue a los usuarios: picos de día, valles de noche. Las GPUs de serving que sobran de noche se pueden usar para entrenar, si el planificador puede devolverlas a tiempo, con preempción y checkpoints. DeepSeek lo cuenta en su resumen del sistema de inferencia, que vimos en la Clase 19: reduce los nodos de inferencia de noche y usa esas GPUs para investigación y entrenamiento.
Pasar a la siguiente: qué llevarse.""",
    d.footer("Choudhury et al., OSDI 2024 · DeepSeek, Open Infra Index, 2025"), gap=40)

# 24 ─ takeaways
d.slide("takeaways", D, head("Para llevarse", "Un planificador de GPUs decide tres cosas sin saber cuánto dura nada", D) +
    row(card(label("Todo o nada", D, color="acc") + p("Gang scheduling para no trabarse; backfill para no bloquear la cola; cuidado con el hambre de los grandes.", D, size=28), D),
        card(label("Dónde", D, color="acc") + p("Localidad contra espera, y la fragmentación en el medio. Consolidar a los que les importa.", D, size=28), D),
        card(label("En qué orden", D, color="acc") + p("Sin duraciones, el que menos GPU·tiempo recibió primero. Desalojar cuesta un checkpoint.", D, size=28), D), gap=32),
    """Tiempo: 1:17–1:20 (3 min).
Idea clave: los tres problemas de hoy son clásicos (deadlock, fragmentación, SRTF), con dos vueltas: los trabajos piden muchas GPUs a la vez y dónde corren cambia cuánto tardan.
Decir: tres decisiones. Todo o nada: un trabajo distribuido se admite entero o no se admite, porque si no, retener y esperar lleva al deadlock; y con gang aparece el bloqueo de cabeza de cola, que backfill arregla a costa del hambre de los grandes. Dónde: repartido en más servidores, un trabajo corre peor (la tabla 5 de Philly), pero exigir que esté junto lo hace esperar, casi siempre por fragmentación; la salida de Tiresias es consolidar sólo a los sensibles. En qué orden: sin conocer las duraciones, 2D-LAS (el que menos GPU·tiempo recibió primero) se acerca a SRTF, con dos colas para no desalojar todo el tiempo, porque cada desalojo cuesta un checkpoint.
Y por encima: cuotas entre equipos con préstamo, GPUs compartidas entre trabajos chicos, y a escala, planificar entre regiones.
Conectar con las clases 16 y 17: allá el trabajo tenía todas las GPUs; acá vimos por qué conseguirlas no es trivial.
Pasar a la siguiente: el repositorio.""",
    d.footer())

# 25 ─ demo repo
cmds = [("go run ./cmd/sim carga", "la forma de la carga"), ("go run ./cmd/sim traba", "dos trabajos sin gang, trabados"),
        ("go run ./cmd/sim gang", "sin gang, FIFO y backfill"), ("go run ./cmd/sim localidad", "consolidar, repartir o relajar"),
        ("go run ./cmd/sim duraciones", "FIFO, 2D-LAS y SRTF"), ("go test ./...", "los tests de lo que dijimos")]
lines = "".join(f'<div style="display:flex;gap:48px">{p(c, D, size=30, color="ink", mono=True, width=700)}{p(w, D, size=30, color="muted")}</div>' for c, w in cmds)
d.slide("demo", D, head("El repositorio", "planificar-gpus: un clúster de 256 GPUs en tiempo virtual", D) +
    f'<div style="display:flex;flex-direction:column;gap:16px;background:#0A0C0F;border:1px solid {D["line"]};border-radius:16px;padding:40px 48px">{lines}</div>' +
    p("Paquetes: cluster (topología, ubicación, fragmentación), trace (la carga) y sched (el simulador y los planificadores). Y el simulador Planificador de GPUs.", D, size=28),
    """Tiempo: 1:20–1:24 (4 min).
Idea clave: un simulador de eventos discretos chico alcanza para ver todos los efectos de la clase, y los tests fijan las afirmaciones.
Decir: el repositorio planificar-gpus, en Go sin dependencias. El paquete cluster tiene la topología (4 racks de 8 servidores de 8 GPUs), la búsqueda de lugar con best-fit, consolidada o donde haya lugar, y la velocidad de un trabajo según dónde quedó. El paquete trace arma la carga. El paquete sched es el simulador: procesa llegadas, finales y cambios de cola en tiempo virtual, y tiene todos los planificadores de hoy. Una corrida de 2.000 trabajos tarda milisegundos.
Los tests verifican lo que dijimos, con tres cargas distintas para no depender de una semilla con suerte: sin gang dos trabajos se traban; backfill elimina el bloqueo de cabeza de cola y hace esperar a los grandes; consolidar le gana a repartir; consolidar siempre produce espera por fragmentación; 2D-LAS le gana a FIFO y a backfill y queda a menos de 5% de SRTF; un umbral bajo desaloja de más; y una preempción cara empeora el JCT.
El simulador Planificador de GPUs es el mismo modelo en el navegador: se ve la grilla llenarse, la cola, y se puede cambiar de política en vivo.
Ejercicios del README: EASY backfilling, el PROMOTEKNOB de Tiresias, cuotas por equipo, fallar temprano, migrar para desfragmentar y compartir una GPU.
Pasar a la siguiente: preguntas.""",
    d.footer("go test ./... · " + SIM))

# 26 ─ preguntas
qs = ["¿Qué condición de Coffman rompe el gang scheduling? ¿Y el timeout de Kueue?",
      "¿Por qué con backfill codicioso un trabajo grande puede no arrancar nunca?",
      "En Philly, ¿por qué esperan sobre todo los trabajos grandes, si hay GPUs libres?",
      "¿Por qué LAS se parece a SRTF cuando las duraciones tienen cola pesada?",
      "¿Qué pasa con 2D-LAS si el umbral entre colas es muy bajo? ¿Y si desalojar cuesta mucho?"]
d.slide("preguntas", D, head("Para repasar", "Preguntas y lecturas", D) +
    row(f'<ol style="flex:1.4;font-size:28px;line-height:1.4;color:{D["ink"]}">' + "".join(f"<li>{q}</li>" for q in qs) + "</ol>",
        card(h3("Lecturas", D) + p("<b>Jeon et al.</b>, Philly, USENIX ATC 2019.", D, size=24, color="ink") +
             p("Gu et al., Tiresias, NSDI 2019. Xiao et al., Gandiva, OSDI 2018.", D, size=24) +
             p("Weng et al., MLaaS in the Wild, NSDI 2022. Choudhury et al., MAST, OSDI 2024.", D, size=24), D, flex=1, pad=40, gap=20), gap=64),
    """Tiempo: 1:24–1:30 (6 min).
Idea clave: cerrar con las preguntas de repaso. Las respuestas cortas, para tener a mano:
1. Retener y esperar: el trabajo pide todas las GPUs juntas o ninguna. Kueue no la rompe: deja que el trabajo retenga, y si no completa a tiempo lo desaloja (rompe "sin desalojo", como la recuperación de un deadlock).
2. Porque cada vez que se libera lugar lo toma un trabajo chico que entra; el grande necesita muchas GPUs libres a la vez y nunca las junta. Se arregla reservando (EASY, que necesita duraciones) o con prioridades como LAS.
3. Por fragmentación: piden localidad (un servidor o un rack) y las GPUs libres están desparramadas en servidores a medio usar. Para los de más de 8 GPUs, el 97,9% de las esperas.
4. Porque con cola pesada, haber corrido mucho predice que falta mucho: atender primero al que menos recibió es atender primero, en promedio, al que menos le falta.
5. Con un umbral bajo, casi todo baja a la segunda cola enseguida, que es FIFO, y se desaloja mucho más (en la demo, 2.642 contra 569 preempciones). Si desalojar cuesta mucho, cada preempción le suma ese tiempo al trabajo y el JCT sube.
Lecturas: Philly es el paper de la clase. Tiresias y Gandiva son los dos planificadores; MLaaS in the Wild y MAST, cómo se ve después. Los apuntes tienen más preguntas.""",
    d.footer())

d.write_index("Clase 21 — Repartir un clúster de GPUs")
print(len(d.order), "slides")
