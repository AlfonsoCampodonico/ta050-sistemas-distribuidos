import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidekit import *
from cover import cover

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "clase-22")
d = Deck(ROOT, 22, "Entrenar con lo que se sirve")
D, L = DARK, LIGHT

# 1 ─ portada
d.raw("cover", cover(22, "Entrenar con lo que se sirve",
    "Post-entrenamiento con RL: cada paso genera como un servidor de inferencia y aprende como un entrenamiento, y las dos cargas comparten el clúster",
    "Paper: Sheng et al., <i>HybridFlow: A Flexible and Efficient RLHF Framework</i>, EuroSys 2025 (verl)",
    """Tiempo: 0:00–0:02 (2 min).
Idea clave: el post-entrenamiento con RL es el lugar donde se juntan las dos mitades de la serie: en cada paso hay que servir el modelo para generar respuestas y entrenarlo con ellas.
Decir: las clases 16 y 17 fueron sobre entrenar: cómo se parte un modelo entre miles de GPUs y qué pasa cuando algo falla. Las 18 y 19 fueron sobre servir: el KV cache, el batching continuo, el router. Las 20 y 21 fueron sobre compartir GPUs entre modelos y entre trabajos. Hoy cerramos con el caso en el que todo eso aparece junto: el entrenamiento con refuerzo que convirtió a los modelos de lenguaje en modelos de razonamiento.
El paper de hoy es HybridFlow, de ByteDance y la Universidad de Hong Kong, EuroSys 2025. Su implementación abierta es verl, un framework para hacer RL sobre modelos de lenguaje.
Pasar a la siguiente: dónde estamos en la serie."""), section="Por qué RL: el bucle, GRPO, y los modelos que intervienen.")

# 2 ─ mapa
d.slide("mapa", D, head("Dónde estamos", "Las dos mitades de la serie, en un mismo bucle", D) +
    table(["", "Qué se hace", "El estado que importa"], [
        ["Clases 16 y 17", "entrenar: partir el modelo en dimensiones", "pesos, gradientes, estado del optimizador"],
        ["Clases 18 y 19", "servir: lotes continuos, router, prefill y decode", "el KV cache"],
        ["Clases 20 y 21", "compartir GPUs entre modelos y entre trabajos", "qué ocupa cada GPU"],
        ["Hoy", "generar, puntuar y actualizar, una y otra vez", "los pesos, en dos layouts, y su versión"],
    ], D, widths=[20, 44, 36], size=30, hi_rows=(3,)),
    """Tiempo: 0:02–0:04 (2 min).
Idea clave: el bucle de RL tiene las dos cargas de la serie, y la novedad es que los mismos pesos tienen que estar en las dos a la vez, o ir y venir entre ellas.
Decir: recorrer las filas. En entrenamiento, el estado eran los pesos, los gradientes y el optimizador, partidos en dimensiones. En serving, el KV cache. En las clases 20 y 21, qué ocupa cada GPU y quién la reparte. Hoy el estado que importa son los pesos: están en el layout de entrenamiento y hacen falta en el de generación, y cada vez que cambian hay que llevarlos de un lado al otro. Y aparece algo nuevo: la versión. Cada respuesta se generó con una versión de los pesos, y no siempre es la última.
Pasar a la siguiente: qué es el bucle.""",
    d.footer())

# 3 ─ bucle
def stage(title, sub, clase, color, t=D):
    return (f'<div style="flex:1;display:flex;flex-direction:column;gap:12px;padding:32px;border-radius:16px;background:{t["card"]};border:2px solid {t[color]}">'
            f'{label(clase, t, color=color)}{h3(title, t, size=40)}{p(sub, t, size=26)}</div>')
def arrow(t=D, w=64):
    return f'<x-shape kind="arrow-right" style="width:{w}px;height:32px;background:{t["muted"]};align-self:center"></x-shape>'
d.slide("bucle", D, head("El bucle", "Generar, puntuar, actualizar", D) +
    row(stage("Generar", "El modelo actual contesta G veces cada problema. Es decode: una carga de serving.", "Clase 18", "acc"), arrow(),
        stage("Puntuar", "Un verificador: ¿la respuesta es la correcta? ¿pasa los tests?", "recompensa", "warn"), arrow(),
        stage("Actualizar", "Un paso de gradiente con esas respuestas. Una carga de entrenamiento.", "Clase 16", "blue"), gap=20) +
    p("Y vuelta a generar, con los pesos nuevos. Cada vuelta es un paso de entrenamiento; un post-entrenamiento son miles.", D, size=30),
    """Tiempo: 0:04–0:07 (3 min).
Idea clave: un paso de RL tiene una carga de serving (generar) y una de entrenamiento (actualizar), y la segunda necesita lo que produjo la primera.
Decir: en el pre-entrenamiento de la Clase 17 los datos estaban fijos: texto de internet. En el post-entrenamiento con RL los datos los genera el propio modelo. Se toma un lote de problemas (de matemática, de programación), el modelo contesta cada uno varias veces, un verificador dice cuáles respuestas están bien, y se actualizan los pesos para que las buenas sean más probables. Y se repite con el modelo nuevo.
Generar es exactamente lo de la Clase 18: decode, limitado por la memoria, con KV cache y batching continuo. Actualizar es lo de la Clase 16: forward, backward, all-reduce de gradientes.
Preguntar a la clase: ¿qué tiene de raro, desde sistemas, que los datos de entrenamiento los genere el modelo que se está entrenando? (Que hay una dependencia: no se puede generar el lote siguiente con los pesos nuevos hasta que no se terminó de entrenar con el anterior.)
Pasar a la siguiente: por qué esto importa tanto desde 2025.""",
    d.footer(), gap=48)

# 4 ─ R1 (claro)
d.slide("r1", L, col(eyebrow("DeepSeek-R1-Zero · enero de 2025", L), h2("RL con recompensas que se verifican, sin ejemplos de razonamiento", L), gap=16) +
    row(big("15,6% → 71,0%", "AIME 2024, pass@1, antes y después del RL", L, size=88), big("86,7%", "con voto de mayoría entre varias respuestas", L, color="blue", size=88),
        big("79,8%", "DeepSeek-R1, el modelo final", L, color="warn", size=88), gap=80) +
    p("La recompensa son dos reglas: la respuesta es correcta (o el código pasa los tests) y el razonamiento está entre &lt;think&gt; y &lt;/think&gt;. Un modelo de recompensa neuronal, dicen, «puede sufrir reward hacking».", L, size=28),
    """Tiempo: 0:07–0:11 (4 min).
Idea clave: R1-Zero mostró que con RL y recompensas que se pueden verificar con reglas, un modelo aprende a razonar sin que nadie le muestre cómo.
Decir: DeepSeek tomó su modelo base, V3, y lo entrenó sólo con RL. La recompensa es por reglas: para matemática, si la respuesta final coincide con la correcta; para programación, si el código compila y pasa los tests; y un premio por formato, que el razonamiento esté entre las etiquetas think. No usan un modelo de recompensa neuronal porque, dice el paper, puede sufrir reward hacking en un RL a gran escala: el modelo aprende a engañar al juez.
Los números son de la versión de enero de 2025 del paper: en AIME 2024 pasa de 15,6% a 71,0% de acierto con una sola respuesta, y a 86,7% votando entre varias. El modelo final, DeepSeek-R1, llega a 79,8%. (La versión revisada del paper reporta 77,9% para R1-Zero.)
Kimi k1.5 de Moonshot, publicado el mismo mes, reporta 77,5 en AIME. Desde entonces, todos los laboratorios hacen esto.
Para la materia: si la recompensa es un programa (comparar, correr tests), puntuar es barato y paralelizable; lo caro es generar y entrenar.
Pasar a la siguiente: el algoritmo.""",
    d.footer("DeepSeek-AI, arXiv 2501.12948 v1, §1 y §2.2 · Kimi Team, arXiv 2501.12599", L), gap=56)

# 5 ─ GRPO
d.slide("grpo", D, head("GRPO · DeepSeek", "La ventaja se mide contra el resto del grupo, sin crítico", D) +
    row(col(card(p("A<sub>i</sub> = (r<sub>i</sub> − media(r<sub>1</sub>…r<sub>G</sub>)) / desvío(r<sub>1</sub>…r<sub>G</sub>)", D, size=40, color="acc", mono=True, weight=600), D, pad=36),
            p("Para cada problema se generan G respuestas. Las que sacaron más que el promedio del grupo se refuerzan; las que sacaron menos, se castigan. Un término de KL las mantiene cerca de la política de referencia.", D, size=28), gap=24, flex=1.3),
        card(label("Lo que se ahorra", D, color="warn") + p("PPO usa un crítico que estima el valor de cada estado, «típicamente del mismo tamaño que la política». GRPO lo saca: un modelo menos para entrenar, guardar y mover.", D, size=28), D, flex=1, pad=36), gap=48, align="center"),
    """Tiempo: 0:11–0:15 (4 min).
Idea clave: GRPO reemplaza el crítico de PPO por una comparación dentro del grupo, y eso le saca al sistema un modelo entero.
Decir: el gradiente de política necesita saber si una respuesta fue mejor o peor de lo esperado: eso es la ventaja. PPO la estima con un crítico, otra red que predice cuánta recompensa espera, y que hay que entrenar en paralelo. El paper de R1 dice que ese crítico es típicamente del mismo tamaño que la política: para un modelo de cientos de miles de millones de parámetros, es otro modelo de ese tamaño.
GRPO, Group Relative Policy Optimization, hace otra cosa: genera G respuestas al mismo problema y usa el grupo como referencia. La ventaja de cada una es su recompensa menos la media del grupo, dividido por el desvío. Si todas acertaron o todas fallaron, la ventaja es cero: ese problema no enseña nada en ese paso.
El resto es como PPO: la razón entre la política nueva y la que generó, recortada, y un término de KL contra la política de referencia. La versión revisada del paper detalla la configuración: 16 respuestas por problema, con un tope de 32.768 tokens que después sube a 65.536.
Para sistemas: las G respuestas de un problema son un grupo; el trainer necesita el grupo completo para calcular la ventaja de cualquiera de ellas.
Pasar a la siguiente: cuántos modelos hay en juego.""",
    d.footer("DeepSeek-AI, arXiv 2501.12948, §2.2.1 (v1) y §2.1 (v2)"), gap=48)

# 6 ─ modelos
d.slide("modelos", D, head("HybridFlow §2.1", "Hasta cuatro modelos, tres etapas, layouts distintos", D) +
    table(["Modelo", "Qué hace", "Generación", "Preparación", "Entrenamiento"], [
        ["Actor", "la política: genera y aprende", "✓", "", "✓"],
        ["Crítico", "estima el valor (PPO; GRPO no lo usa)", "", "✓", "✓"],
        ["Referencia", "la política original, para el KL", "", "✓", ""],
        ["Recompensa", "puntúa; o un verificador por reglas", "", "✓", ""],
    ], D, widths=[18, 40, 14, 14, 14], size=30) +
    p("El actor está en dos etapas con cargas opuestas: decode limitado por la memoria y entrenamiento limitado por el cómputo. Cada etapa quiere otro paralelismo.", D, size=28),
    """Tiempo: 0:15–0:18 (3 min).
Idea clave: un paso de RLHF clásico (PPO) mueve cuatro modelos por tres etapas, y el actor aparece en dos etapas que quieren paralelismos distintos.
Decir: HybridFlow describe el flujo de datos de RLHF así. Etapa 1, generación: el actor produce las respuestas. Etapa 2, preparación: el crítico calcula valores, la referencia calcula log-probabilidades para el KL, y el modelo de recompensa puntúa. Etapa 3, entrenamiento: se actualizan el actor y el crítico con Adam.
Con GRPO y recompensas por reglas, se van el crítico y el modelo de recompensa: queda el actor, la referencia (que es un forward) y un verificador que es un programa. Pero el problema de sistemas central no cambia: el actor genera y entrena.
Y esas dos cargas son opuestas. Generar es decode: en la Clase 18 vimos que está limitado por leer los pesos, y conviene poco paralelismo de tensores y muchas réplicas. Entrenar está limitado por el cómputo y por la memoria del optimizador, y conviene partir el modelo (Clase 16). El mismo modelo, dos layouts.
Pasar a la siguiente: qué pasa cuando los modelos aprenden a razonar.""",
    d.footer("Sheng et al., EuroSys 2025, §2.1 y figura 1"), gap=40)

# 7 ─ alargan
d.slide("alargan", D, head("Al aprender, las respuestas se alargan", "El rollout se vuelve más caro a medida que el modelo mejora", D) +
    row(card(label("DeepSeek-R1-Zero", D, color="acc") + p("El largo medio de las respuestas crece durante todo el RL (figura 3): el modelo aprende solo a pensar más.", D, size=28) +
             p("Tope por respuesta: 32.768 tokens, y 65.536 desde el paso 8.200.", D, size=28, color="ink"), D, pad=36),
        card(label("Kimi k1.5", D, color="blue") + p("El RL usa una ventana de contexto de 128k tokens.", D, size=28), D, pad=36),
        card(label("La demo", D, color="warn") + p("El largo medio pasa de 353 a 3.822 tokens; la respuesta más larga de un lote, de 1.388 al tope de 8.192.", D, size=28), D, pad=36), gap=32),
    """Tiempo: 0:18–0:21 (3 min).
Idea clave: en los modelos de razonamiento, aprender quiere decir pensar más, así que el costo de generar crece durante el entrenamiento.
Decir: la figura 3 de R1 muestra el largo medio de las respuestas durante el RL: sube todo el tiempo. Nadie le dijo al modelo que pensara más; lo aprendió porque pensar más acierta más. La versión revisada del paper dice que el tope de generación era 32.768 tokens y que lo subieron a 65.536 a partir del paso 8.200. Kimi k1.5 hace el RL con ventana de 128k.
La demo de hoy está armada para eso: el "modelo" decide cuánto pensar; pensar más acierta más, pero si pasa el tope la respuesta se corta y vale cero. Al empezar, la respuesta media tiene 353 tokens; después de entrenar, 3.822, y la más larga del lote llega al tope.
Para sistemas: la carga de serving del bucle no es fija. Lo que dimensionaste al principio no alcanza al final, y la forma de la distribución cambia: más larga y con más cola.
Pasar a la siguiente: qué hace esa cola con un paso sincrónico.""",
    d.footer("DeepSeek-AI, 2501.12948 (v1, figura 3; v2, §2.1) · Kimi Team, 2501.12599 · go run ./cmd/sim cola"), gap=48)

# 8 ─ cola (claro)
d.slide("cola", L, head("Demo · la cola larga", "Un paso sincrónico dura lo que la respuesta más larga", L) +
    table(["Política", "Media", "p50", "p99", "Máx", "1.ª GPU termina", "Última", "GPU ociosa"], [
        ["Al empezar", "353", "305", "1.165", "1.388", "16 s", "30 s", "86%"],
        ["Ya entrenada", "3.822", "3.478", "8.192", "8.192", "178 s", "192 s", "75%"],
    ], L, widths=[19, 10, 9, 10, 10, 16, 11, 15], right=(1, 2, 3, 4, 5, 6, 7), size=28) +
    p("256 respuestas en 8 GPUs, hasta 64 por GPU. Largos en tokens. Ociosa: GPU·tiempo con lugares vacíos en el lote, hasta que termina la última.", L, size=24, color="muted") +
    p("Una GPU con una sola secuencia paga casi lo mismo por iteración que una con 64: el decode está limitado por leer los pesos.", L, size=28),
    """Tiempo: 0:21–0:25 (4 min, con demo).
Idea clave: con largos de cola larga, un paso sincrónico está casi todo el tiempo esperando a unas pocas respuestas.
Demo: go run ./cmd/sim cola, y en el simulador Bucle de RL, modo Colocado: mirar cómo se vacían los lotes de las cuatro GPUs y una sola barra sigue creciendo.
Leer la tabla. 32 problemas por 8 respuestas, repartidas en 8 GPUs. Al empezar, la respuesta mediana tiene 305 tokens pero la más larga 1.388: la primera GPU termina a los 16 segundos y la última a los 30. Con la política ya entrenada, la mediana es 3.478 y la cola llega al tope: 178 contra 192 segundos. En los dos casos, entre 75% y 86% del GPU·tiempo de generación es lugar vacío en el lote.
¿Por qué lugar vacío y no sólo GPU parada? Por la Clase 18: una iteración de decode cuesta casi lo mismo con una secuencia que con 64, porque lo caro es leer los pesos. Una GPU con una respuesta larga y 63 lugares vacíos está al 100% de uso y produce el 2% de lo que podría.
AReaL lo dice así: la generación tiene que esperar a que termine la salida más larga del lote antes de actualizar el modelo, y eso subutiliza las GPUs.
Pasar a la siguiente: este problema ya lo conocemos.""",
    d.footer("go run ./cmd/sim cola · simulador Bucle de RL · Fu et al., AReaL, abstract", L, plain=True), gap=36, section="El rollout domina: la cola larga y la barrera.")

# 9 ─ barrera
def lanes(t=D):
    lens = [0.32, 0.41, 0.28, 0.36, 0.95, 0.30, 0.44, 0.38]
    out = []
    for i, f in enumerate(lens):
        out.append(f'<div style="display:flex;align-items:center;gap:16px"><p style="font-family:{MONO};font-size:22px;color:{t["muted"]};width:96px">GPU {i}</p>'
                   f'<div style="flex:1;height:30px;border-radius:6px;background:{t["fill"]};position:relative;overflow:hidden">'
                   f'<div style="position:absolute;left:0;top:0;bottom:0;width:{f*100:.0f}%;background:{t["acc"]};border-radius:6px"></div></div></div>')
    out.append(f'<div style="position:absolute;right:0;top:-12px;bottom:-12px;width:3px;background:{t["warn"]}"></div>')
    return f'<div style="position:relative;display:flex;flex-direction:column;gap:10px;padding-right:12px">{"".join(out)}</div>'
d.slide("barrera", D, head("Ya lo vimos en la Clase 16", "Un paso sincrónico es una barrera, y la barrera espera al más lento", D) +
    row(col(lanes(), p("generando · la línea naranja es la barrera: nadie entrena hasta que llega la última", D, size=22, color="muted"), gap=16, flex=1.4),
        col(card(p("Paralelismo de datos: un straggler frena el all-reduce de todos.", D, size=28), D, pad=32),
            card(p("RL sincrónico: una respuesta larga frena el paso de todos.", D, size=28, color="ink"), D, accent=True, pad=32), gap=24, flex=1), gap=48, align="center"),
    """Tiempo: 0:25–0:28 (3 min).
Idea clave: es el problema de los stragglers de la Clase 16 con otro disfraz: en un modelo sincrónico, el paso dura lo que el más lento.
Decir: en la Clase 16 vimos paralelismo de datos sincrónico: cada réplica calcula su gradiente, y el all-reduce espera a todas. Si una GPU es más lenta, todas la esperan. Acá pasa lo mismo, pero el straggler no es una GPU mala: es una respuesta larga. Y no es la excepción: con una distribución de cola larga, en un lote de cientos de respuestas siempre hay alguna larga.
Las soluciones de siempre para stragglers sirven de inspiración: no esperar al más lento (asincronía), acotar cuánto se puede adelantar el rápido (staleness acotada), o partir el trabajo del lento en pedazos (rollouts parciales). Son las tres cosas que vamos a ver.
Preguntar a la clase: en un sistema de parameter server con workers asincrónicos, ¿qué se pierde? (Los workers calculan gradientes con pesos viejos.) Acá va a pasar exactamente eso.
Pasar a la siguiente: cuánto pesa la generación en un paso.""",
    d.footer("Clase 16 · Fu et al., AReaL, 2025"), gap=48)

# 10 ─ cuánto pesa
d.slide("pesa", D, head("Cuánto pesa cada parte", "Generar y entrenar el actor se llevan casi todo", D) +
    row(big("58,9%", "del tiempo de RLHF con HybridFlow: entrenar y generar el actor", D, size=96),
        big("81,2%", "hasta ese porcentaje del paso es generación en NeMo-Aligner", D, color="warn", size=96),
        big("36,4%", "de un paso, mover los 140 GB de pesos de un 70B entre GPUs distintas", D, color="blue", size=96), gap=72) +
    p("Los tres son de HybridFlow (§2.3 y §8.2); el último cita a OpenRLHF. El cuello de botella depende de dónde esté cada trabajo.", D, size=28),
    """Tiempo: 0:28–0:31 (3 min).
Idea clave: el tiempo de un paso se reparte entre generar, entrenar y mover pesos, y cuál domina depende de cómo esté armado el sistema.
Decir: tres números del paper de hoy. Con HybridFlow, entrenar y generar el actor son el 58,9% del tiempo de RLHF: el resto son los otros modelos (crítico, referencia, recompensa). En NeMo-Aligner, uno de los sistemas contra los que comparan, la generación llega al 81,2% del paso, porque genera con el mismo motor de entrenamiento y no con un motor de inferencia como vLLM.
Y el tercero: si la generación y el entrenamiento están en GPUs distintas, hay que mandar los pesos después de cada paso. Para un 70B son 140 GB, y según OpenRLHF, citado por HybridFlow, eso puede llevarse hasta el 36,4% de una iteración.
Lección de sistemas: generar en un motor de inferencia de verdad (lo de la Clase 18) es casi obligatorio; y una vez que se hace eso, aparece el costo de mover los pesos entre el mundo del entrenamiento y el de la inferencia. Las dos partes que siguen.
Pasar a la siguiente: dónde poner cada trabajo.""",
    d.footer("Sheng et al., EuroSys 2025, §2.3 y §8.2"), gap=56)

# 11 ─ ubicaciones
d.slide("ubicaciones", D, head("Dónde va cada trabajo · HybridFlow §8.3", "Colocado, separado o partido", D) +
    row(card(label("Colocado", D, color="acc") + p("Todos los modelos en las mismas GPUs, por turnos.", D, size=28, color="ink") + p("DeepSpeed-Chat; así trabaja el 3D-HybridEngine.", D, size=24), D, accent=True, pad=32),
        card(label("Separado", D, color="blue") + p("Cada modelo en su grupo de GPUs.", D, size=28, color="ink") + p("OpenRLHF: Ray, vLLM para generar, DeepSpeed para entrenar.", D, size=24), D, pad=32),
        card(label("Partido", D, color="warn") + p("Actor y referencia en un grupo; crítico y recompensa en otro.", D, size=28, color="ink") + p("NeMo-Aligner.", D, size=24), D, pad=32), gap=28) +
    p("Con 16 a 64 GPUs gana colocar todo junto. Con 96 a 128 GPUs y modelos de 13B o 34B, gana partir o separar.", D, size=30, color="ink"),
    """Tiempo: 0:31–0:35 (4 min).
Idea clave: no hay una ubicación que gane siempre; HybridFlow la elige según el tamaño del clúster y de los modelos.
Decir: las tres formas que compara el paper. Colocado: todos los modelos comparten todas las GPUs y se turnan; es lo que hace DeepSpeed-Chat, y es el caso para el que HybridFlow diseñó su 3D-HybridEngine. Separado: cada modelo tiene su grupo de GPUs; es OpenRLHF, que está armado sobre Ray, con vLLM para generar y DeepSpeed para entrenar. Partido: actor y referencia en un grupo, crítico y recompensa en otro; es NeMo-Aligner. Para PPO hay 15 ubicaciones posibles, y HybridFlow tiene un algoritmo que las recorre y simula cada una.
El resultado (§8.3): de 16 a 64 GPUs, colocar todo en las mismas GPUs da el mejor rendimiento. Con 96 a 128 GPUs y modelos de 34B, o 96 GPUs con 13B, la partida es la óptima; con 13B en 128 GPUs gana la separada.
¿Por qué? Colocado no deja ninguna GPU esperando a otra, pero todos los modelos tienen que entrar en la memoria de cada GPU y las etapas no se superponen. Separado deja que cada modelo use el paralelismo que le conviene y que las etapas corran a la vez, pero cada grupo espera al otro.
Pasar a la siguiente: el costo escondido de colocar.""",
    d.footer("Sheng et al., EuroSys 2025, §6 y §8.3 · Hu et al., OpenRLHF"), gap=40, section="Dónde va cada trabajo, y cuánto cuesta mover los pesos.")

# 12 ─ reshard
def grid(lbl, groups, t=D, color="acc"):
    cells = "".join(f'<div style="width:120px;height:72px;border-radius:10px;background:{t[c]};display:flex;align-items:center;justify-content:center">'
                    f'<p style="font-family:{MONO};font-size:22px;font-weight:600;color:{t["onfill"]}">{txt}</p></div>' for txt, c in groups)
    return col(label(lbl, t), f'<div style="display:flex;gap:12px">{cells}</div>', gap=12)
d.slide("reshard", D, head("Colocado · HybridFlow §5", "El mismo actor, dos layouts", D) +
    row(grid("Entrenar: t = 2, d = 2", [("w₀ · d0", "blue"), ("w₁ · d0", "blue"), ("w₀ · d1", "acc"), ("w₁ · d1", "acc")]),
        f'<x-shape kind="arrow-right" style="width:96px;height:40px;background:{D["muted"]};align-self:flex-end;margin-bottom:16px"></x-shape>',
        grid("Generar: tg = 1, cuatro réplicas", [("w", "warn"), ("w", "warn"), ("w", "warn"), ("w", "warn")]), gap=40, align="center") +
    p("Entrenar quiere partir el modelo para que entren el optimizador y las activaciones. Generar quiere poco paralelismo de tensores y muchas réplicas: el decode está limitado por leer los pesos.", D, size=28) +
    p("Con generación en tg = 2 para un 7B y tg = 4 para un 13B, la latencia de generar baja 60,3% y 36,4% (16 GPUs, entrenamiento con t = 8).", D, size=28, color="ink"),
    """Tiempo: 0:35–0:39 (4 min).
Idea clave: si el actor genera y entrena en las mismas GPUs, entre una etapa y la otra hay que cambiar cómo está partido el modelo, y eso es comunicación.
Decir: el ejemplo de la figura 7 del paper, simplificado. Para entrenar, el actor está partido con paralelismo de tensores 2 y de datos 2: cada GPU tiene la mitad de los pesos. Para generar conviene otra cosa. En la Clase 18 vimos que el decode está limitado por leer los pesos de memoria; partir en más GPUs reduce eso pero agrega dos all-reduce por capa por cada token, y para generar conviene menos paralelismo de tensores y más réplicas independientes. En el dibujo, cada GPU pasa a tener el modelo entero y genera sola.
HybridFlow lo mide: con 16 GPUs y el entrenamiento en t = 8, generar con tg = 2 para un 7B baja la latencia de generación 60,3%, y con tg = 4 para un 13B, 36,4%.
Así que en cada paso: entrenar en un layout, juntar los pedazos que faltan para el otro layout, generar, y volver. HybridFlow llama a eso la transición, y a su componente que la hace, 3D-HybridEngine. En su notación, el layout de entrenamiento es p-t-d y el de generación pg-tg-dg-d: los grupos dg son micro grupos de datos que sólo existen durante la generación.
Pasar a la siguiente: cuánto cuesta esa transición.""",
    d.footer("Sheng et al., EuroSys 2025, §5.1, figura 7 y §8.4"), gap=40)

# 13 ─ hybridengine (claro)
d.slide("hybridengine", L, head("3D-HybridEngine · tabla 2", "Juntar sólo lo que falta, sin copias de más", L) +
    row(col(table(["Transición", "Volumen", "Repetida", "Ejemplo"], [
        ["DS-Chat", "(tpd − 1)/(tpd) · M", "M/(tpd)", "0,94 M"],
        ["HybridFlow-V", "(tp − 1)/(tp) · M", "M/(tp)", "0,88 M"],
        ["HybridFlow", "(tp − tg·pg)/(tg·pg·tp) · M", "0", "0,38 M"],
    ], L, widths=[22, 44, 17, 17], size=28, bold_rows=(2,), right=(3,)),
        p("M: tamaño del modelo; repetida: memoria que queda duplicada. Ejemplo: entrenamiento con t = 8, p = 1, d = 2 y generación con tg = 2 (go run ./cmd/sim ubicacion).", L, size=22, color="muted"), gap=16, flex=1.8),
        col(big("−55,2%", "tiempo de transición, en promedio (11,7 s)", L, size=80), big("−89,1%", "con modelos de 70B (78,2 s menos)", L, color="blue", size=80), gap=40, flex=1), gap=56, align="center"),
    """Tiempo: 0:39–0:42 (3 min).
Idea clave: la transición entre layouts es un all-gather, y elegir bien qué junta cada GPU baja mucho el volumen y elimina la memoria repetida.
Decir: la tabla 2 del paper compara tres formas de hacer la transición. DeepSpeed-Chat junta el modelo entero en todas las GPUs: un all-gather sobre las t·p·d GPUs, y además cada GPU se queda con una copia de su pedazo de entrenamiento que ya no usa. HybridFlow-V, una versión intermedia, junta dentro de cada grupo de t·p GPUs. HybridFlow elige los grupos de generación para que cada GPU ya tenga parte de lo que necesita, junta sólo lo que le falta, y no repite nada en memoria.
La última columna es la cuenta con un caso concreto: entrenamiento con t = 8 y d = 2, generación con tg = 2. Pasa de mover casi todo el modelo a mover un 38%.
Medido: el tiempo de transición baja 55,2% en promedio, 11,7 segundos, y hasta 89,1%, 78,2 segundos menos, con modelos de 70B. En el total, HybridFlow reporta entre 1,53 y 20,57 veces más throughput que DeepSpeed-Chat, OpenRLHF y NeMo-Aligner, en 128 A100.
Conectar con la Clase 16: es un all-gather, el mismo colectivo de ZeRO. Lo que cambia es que hay que diseñar los grupos.
Pasar a la siguiente: la otra opción, GPUs separadas.""",
    d.footer("Sheng et al., EuroSys 2025, tabla 2, §8.2 y §8.4", L), gap=40)

# 14 ─ broadcast
d.slide("broadcast", D, head("Separado: mandar los pesos después de cada paso", "Es el broadcast de la Clase 16", D) +
    row(col(table(["GPUs de rollout", "Ingenuo", "En anillo"], [["6", "34 s", "5,9 s"], ["48", "269 s", "8,2 s"], ["384", "2.150 s", "27 s"]],
                  D, widths=[40, 30, 30], right=(1, 2), size=30, bold_rows=()),
            p("140 GB (un 70B en bf16) a 25 GB/s (200 Gb/s, la red del testbed de HybridFlow). Ingenuo: el trainer manda una copia a cada una. En anillo: cada GPU reenvía a la siguiente mientras recibe.", D, size=22, color="muted"), gap=16, flex=1.2),
        card(label("Kimi k1.5", D, color="acc") + p("Megatron y vLLM en el mismo pod de Kubernetes. Los pesos pasan de uno al otro por <b>Mooncake</b>, por RDMA: menos de un minuto de entrenar a generar, unos diez segundos al revés.", D, size=28), D, accent=True, flex=1, pad=36), gap=48, align="center"),
    """Tiempo: 0:42–0:46 (4 min).
Idea clave: con pools separados, cada paso termina con un broadcast de los pesos, y cómo se haga decide si escala.
Decir: la tabla es la cuenta de la demo, go run ./cmd/sim ubicacion. 140 GB, el ejemplo de HybridFlow, por enlaces de 200 Gb/s, unos 25 GB/s. Si el trainer manda una copia a cada GPU de rollout, el tiempo crece con la cantidad: con 384 GPUs, más de media hora por paso. Si se arma un anillo o un pipeline, partiendo los pesos en pedazos y reenviando mientras se recibe, el tiempo casi no depende de cuántas sean. Es el mismo argumento del ring all-reduce de la Clase 16.
El recuadro es Kimi k1.5, que hace algo intermedio: entrenamiento con Megatron e inferencia con vLLM en contenedores separados del mismo pod de Kubernetes, compartiendo las GPUs. Un proceso que llaman checkpoint engine pasa los pesos de Megatron a vLLM, y entre nodos usa Mooncake, el sistema de la Clase 19, por RDMA. Reportan menos de un minuto para pasar de entrenar a generar y unos diez segundos al revés. vLLM arranca con pesos de mentira y los reemplaza.
Conectar con la Clase 19: el mismo sistema que movía KV cache entre prefill y decode mueve pesos entre trainer y rollout.
Pasar a la siguiente: la demo.""",
    d.footer("Kimi Team, 2501.12599, §2.6.3 · Sheng et al., §2.3 y §8.1 · go run ./cmd/sim ubicacion"), gap=40)

# 15 ─ demo ubicación (claro)
d.slide("demoubic", L, head("Demo · ubicación", "Con 8 GPUs y todo sincrónico, colocado gana", L) +
    table(["Ubicación", "Segundos por paso", "GPU ociosa", "En qué se va"], [
        ["Colocado (verl)", "217", "69%", "generar 89%, entrenar 10%, transición 1%"],
        ["Separado, 6 rollout + 2 trainer", "279", "76%", "rollout ociosas 77%, trainer ociosas 70%"],
        ["Separado, 4 rollout + 4 trainer", "237", "71%", "rollout ociosas 60%, trainer ociosas 82%"],
    ], L, widths=[30, 18, 14, 38], right=(1, 2), size=28, bold_rows=(0,)) +
    p("Separado y sincrónico, cada pool espera al otro: mientras unas generan, las otras miran.", L, size=30, color="ink"),
    """Tiempo: 0:46–0:49 (3 min, con demo).
Idea clave: si todo es sincrónico, separar sólo agrega espera: cada pool está ocioso mientras el otro trabaja.
Demo: go run ./cmd/sim ubicacion. En el simulador, alternar Colocado y Separado con η = 0 y mirar la línea de tiempo: en separado, las GPUs de rollout y la del trainer se turnan.
Leer la tabla. Colocado: las 8 GPUs generan y después las 8 entrenan; 217 segundos por paso, casi todo generando, y la transición es un 1%. Igual 69% del GPU·tiempo está ocioso: es la cola larga de antes. Separado con 6 y 2: el paso es más largo, porque generan menos GPUs y entrenan sólo 2, y cada lado está ocioso casi tres cuartos del tiempo. Con 4 y 4, mejor, pero no le gana al colocado.
Esto es lo que encontró HybridFlow para clústeres chicos. La pregunta es qué pasa si dejamos que cada pool trabaje sin esperar al otro.
Pasar a la siguiente: la parte central de la clase, la asincronía.""",
    d.footer("go run ./cmd/sim ubicacion · simulador Bucle de RL", L, plain=True), gap=40)

# 16 ─ asincrónico
def timeline(blocks, t=D):
    out = []
    for w, c, txt in blocks:
        out.append(f'<div style="width:{w}px;height:56px;border-radius:8px;background:{t[c] if c in t else c};display:flex;align-items:center;justify-content:center">'
                   f'<p style="font-family:{MONO};font-size:22px;font-weight:600;color:{t["onfill"]}">{txt}</p></div>')
    return f'<div style="display:flex;gap:6px">{"".join(out)}</div>'
FILL = D["fill"]
d.slide("asinc", D, head("Que los rollouts no esperen", "Generar el lote siguiente mientras se entrena con este", D) +
    col(label("Sincrónico", D),
        row(p("rollout", D, size=22, mono=True, width=130), timeline([(300, "acc", "gen v0"), (150, FILL, ""), (300, "acc", "gen v1"), (150, FILL, ""), (300, "acc", "gen v2")]), gap=16, align="center"),
        row(p("trainer", D, size=22, mono=True, width=130), timeline([(300, FILL, ""), (150, "blue", "v0→1"), (300, FILL, ""), (150, "blue", "v1→2"), (300, FILL, "")]), gap=16, align="center"), gap=10) +
    col(label("Asincrónico, staleness 1", D),
        row(p("rollout", D, size=22, mono=True, width=130), timeline([(300, "acc", "gen v0"), (300, "acc", "gen v0"), (300, "acc", "gen v1"), (300, "acc", "gen v2")]), gap=16, align="center"),
        row(p("trainer", D, size=22, mono=True, width=130), timeline([(300, FILL, ""), (150, "blue", "v0→1"), (150, FILL, ""), (150, "blue", "v1→2"), (150, FILL, ""), (150, "blue", "v2→3")]), gap=16, align="center"), gap=10) +
    p("El precio: el lote que entrena v1 → v2 lo generó v0. Los datos son de una versión vieja de la política.", D, size=28, color="ink"),
    """Tiempo: 0:49–0:53 (4 min).
Idea clave: si se deja que la generación del lote siguiente empiece antes de que termine el entrenamiento, ningún pool espera, pero lo que se entrena lo generaron pesos viejos.
Decir: recorrer el dibujo. Arriba, sincrónico: el rollout genera con v0, para; el trainer entrena y produce v1; el rollout genera con v1. Cada pool trabaja la mitad del tiempo.
Abajo, asincrónico: el rollout no para. Mientras el trainer entrena con el primer lote, el rollout ya genera el segundo, todavía con v0, porque v1 no existe. Cuando el trainer termina, ese lote está listo, y entrena con él para producir v2. Pero ese lote lo generó v0: los datos tienen una staleness de 1 versión.
Eso es aprendizaje off-policy. El gradiente de política supone que las muestras salen de la política que se está actualizando; si salen de una vieja, el gradiente está sesgado. Con poca staleness casi no se nota; con mucha, el modelo aprende de algo que ya no es.
Conectar con la Clase 16: es exactamente lo que pasaba con un parameter server asincrónico, donde un worker calculaba su gradiente con pesos que ya habían cambiado.
Pasar a la siguiente: cómo se acotaba eso.""",
    d.footer("Fu et al., AReaL, §3 y §5"), gap=28, section="Asincrónico: staleness acotada, rollouts interrumpibles y parciales.")

# 17 ─ SSP
d.slide("ssp", D, head("Stale Synchronous Parallel · NIPS 2013", "Asincrónico, pero con un límite de cuánto adelantarse", D) +
    table(["", "Parameter server con SSP", "RL asincrónico (AReaL)"], [
        ["Los rápidos", "workers que calculan gradientes", "GPUs de rollout que generan"],
        ["El estado compartido", "los parámetros en el servidor", "los pesos del trainer"],
        ["El límite", "el más rápido, a lo sumo s relojes del más lento", "los rollouts, a lo sumo η versiones adelante"],
        ["Con límite 0", "s = 0: Bulk Synchronous Parallel", "η = 0: RL sincrónico"],
    ], D, widths=[22, 39, 39], size=28) +
    card(p("AReaL no deja empezar una respuesta si ⌊(N<sub>r</sub> − 1)/B⌋ &gt; i + η: N<sub>r</sub> respuestas pedidas, B por lote, i la versión actual.", D, size=28, color="ink"), D, pad=28),
    """Tiempo: 0:53–0:57 (4 min).
Idea clave: el RL asincrónico con staleness acotada es el modelo Stale Synchronous Parallel de los parameter servers, con otros nombres.
Decir: Ho et al. propusieron en NIPS 2013 un punto medio entre sincrónico y asincrónico para parameter servers. Cada worker tiene un reloj, que avanza una vez por iteración. La regla: el worker más rápido y el más lento no pueden estar a más de s relojes de distancia; si el rápido se adelanta, espera. Y un worker en el reloj c ve seguro todas las actualizaciones hasta c − s − 1. Con s = 0 es Bulk Synchronous Parallel.
AReaL hace lo mismo para RL. Tiene un hiperparámetro η, la staleness máxima. El controlador de rollouts lleva la cuenta de cuántas respuestas se pidieron, N_r; con B por lote, la respuesta número N_r va a caer en el lote ⌊(N_r − 1)/B⌋. Si eso está más de η versiones adelante de la versión i del trainer, no la deja empezar. Con η = 0, dice el paper, el sistema degenera en RL sincrónico.
Lo honesto: el límite es sobre cuándo empieza una respuesta. Una respuesta larga que empezó temprano termina tarde, y se entrena con más staleness que η. En la demo se ve: con η = 4 el promedio es 3,6 pero el máximo llega a 10. La cola larga aparece otra vez.
Pasar a la siguiente: AReaL.""",
    d.footer("Ho et al., NIPS 2013, §2 · Fu et al., AReaL, §5.1, ecuación 3"), gap=36)

# 18 ─ AReaL
d.slide("areal", D, head("AReaL · 2025", "Rollouts que no esperan y un objetivo que tolera lo viejo", D) +
    row(card(label("Interrumpibles", D, color="acc") + p("Cuando llegan pesos nuevos, se interrumpe la generación, se descarta el KV cache, se recalcula con los pesos nuevos y se sigue. Una respuesta puede tener pedazos de varias versiones.", D, size=26), D, pad=32),
        card(label("PPO desacoplado", D, color="blue") + p("Separa la política que generó (behavior) de la que sirve de ancla para el recorte (proximal). Así lo viejo no tira hacia atrás.", D, size=26), D, pad=32),
        card(label("Sin barrera", D, color="warn") + p("Tres cuartos de las GPUs generan, un cuarto entrena. η = 8 para matemática, η = 4 para código.", D, size=26), D, pad=32), gap=28) +
    row(big("2,77×", "menos tiempo que su versión sincrónica, mismas GPUs (1,5B: 41,0 h → 14,8 h)", D, size=64),
        big("+12% / +17%", "throughput por interrumpir, con 1,5B y 7B", D, color="blue", size=64), big("512", "H800, modelos de 1,5B a 32B", D, color="warn", size=64), gap=56),
    """Tiempo: 0:57–1:01 (4 min).
Idea clave: AReaL saca la barrera del todo: los rollouts generan sin parar, el trainer entrena cuando hay un lote, y dos mecanismos hacen que eso funcione.
Decir: tres piezas. Uno, rollouts interrumpibles: cuando el trainer publica pesos nuevos, cada worker interrumpe lo que está generando, descarta el KV cache calculado con los pesos viejos, lo recalcula con los nuevos y sigue generando. Sin esto, un worker con una respuesta larga tendría que terminarla antes de cargar los pesos; con esto, una respuesta puede quedar hecha de pedazos de varias versiones.
Dos, PPO desacoplado: en PPO la política que generó los datos es también el ancla del recorte. AReaL las separa: la behavior es la que generó, y la proximal es una política reciente que sirve de ancla. Si el ancla fuera la vieja, dice el paper, tiraría la política nueva hacia versiones viejas y de menor calidad.
Tres, sin barrera: en sus experimentos, tres cuartos de las GPUs generan y un cuarto entrena, con η = 8 para matemática y 4 para código.
Resultados: hasta 2,77 veces menos tiempo de entrenamiento que la versión sincrónica de su propio sistema con las mismas GPUs, con igual o mejor resultado final; para 1,5B, 41 horas contra 14,8. Las interrupciones solas suben el throughput 12% y 17% con 1,5B y 7B. En un clúster de 512 H800, con modelos de 1,5B a 32B.
Pasar a la siguiente: qué le hace la staleness al aprendizaje.""",
    d.footer("Fu et al., arXiv 2505.24298, §4.1, §5, §7.1, §7.2, §7.5 y tabla 1"), gap=32)

# 19 ─ AReaL tabla 2 (claro)
d.slide("arealtabla", L, head("AReaL · tabla 2", "Con PPO común, la staleness cuesta; con el desacoplado, casi no", L) +
    row(col(table(["η", "PPO común", "Desacoplado"], [
        ["0 (sincrónico)", "42,0", "42,0"], ["1", "41,8", "42,1"], ["2", "40,0", "41,8"], ["4", "23,3", "42,2"],
        ["8", "35,7", "41,0"], ["16", "35,8", "38,7"], ["sin límite", "34,0", "36,9"],
    ], L, widths=[36, 32, 32], right=(1, 2), size=28, hi_rows=(3,)), gap=12, width=900),
        col(p("AIME 2024 con un modelo de 1,5B, según la staleness máxima permitida.", L, size=28),
            p("«Una staleness moderada (η ≤ 8) tiene un impacto mínimo en el rendimiento final», con el objetivo desacoplado.", L, size=28, color="ink"),
            p("Con el sincrónico como referencia, la ganancia en tiempo es más del doble.", L, size=28), gap=24, flex=1), gap=64, align="center"),
    """Tiempo: 1:01–1:04 (3 min).
Idea clave: sin corrección, cuanta más staleness peor aprende el modelo; con el objetivo desacoplado, hasta η = 8 casi no se nota.
Decir: la ablación de AReaL, con un modelo de 1,5B en matemática; el número es el acierto en AIME 2024. La columna del medio es PPO común con datos viejos: con η = 1 casi igual, con η = 2 baja a 40, con η = 4 se cae a 23,3, y con más staleness queda en 34 a 36. La columna de la derecha es con el objetivo desacoplado: hasta η = 8 se queda entre 41 y 42; recién con 16 o sin límite empieza a bajar.
La frase del paper: una staleness moderada, η ≤ 8, tiene un impacto mínimo en el rendimiento final. Y en la figura 5, el entrenamiento avanza más de dos veces más rápido que el sincrónico.
Notar que la caída no es monótona en la columna del medio (η = 4 es peor que 8): son corridas únicas, y un entrenamiento de RL que se desestabiliza no siempre se recupera igual. Lo robusto es la diferencia entre columnas.
Pasar a la siguiente: lo mismo en la demo, donde se ve por qué pasa.""",
    d.footer("Fu et al., arXiv 2505.24298, §7.4 y tabla 2", L), gap=40)

# 20 ─ demo asinc (claro)
d.slide("demoasinc", L, head("Demo · asincrónico", "Con las mismas 8 GPUs, tres veces más rápido", L) +
    table(["", "Tiempo (40 pasos)", "Resp./s", "GPU ociosa", "Staleness media", "Máx", "vs verl"], [
        ["Colocado, sincrónico", "8.692 s", "1,18", "69%", "0", "0", "1,00×"],
        ["Separado, sincrónico (η = 0)", "11.141 s", "0,92", "76%", "0", "0", "0,78×"],
        ["Separado, η = 1", "5.785 s", "1,77", "53%", "0,97", "2", "1,50×"],
        ["Separado, η = 2", "3.867 s", "2,65", "33%", "1,92", "5", "2,25×"],
        ["Separado, η = 4", "2.901 s", "3,53", "12%", "3,60", "10", "3,00×"],
        ["Separado, η = 8", "2.874 s", "3,56", "8%", "3,95", "10", "3,02×"],
    ], L, widths=[30, 16, 9, 12, 14, 7, 12], right=(1, 2, 3, 4, 5, 6), size=26, bold_rows=(4,)) +
    p("6 GPUs generan y 2 entrenan; interrumpibles y con corrección. Promedio de 6 corridas.", L, size=24, color="muted"),
    """Tiempo: 1:04–1:08 (4 min, con demo).
Idea clave: dejar que los rollouts se adelanten saca casi toda la espera; la GPU ociosa baja de 69% a 12% y el mismo entrenamiento termina en un tercio del tiempo.
Demo: go run ./cmd/sim asinc. En el simulador, modo Separado, mover el slider de η de 0 a 8 y mirar la línea de tiempo: las GPUs de rollout dejan de tener huecos.
Leer la tabla. Los dos sincrónicos, de antes. Con η = 1 ya se gana 1,5 veces: el rollout genera el lote siguiente mientras se entrena. Con η = 2, 2,25. Con η = 4, tres veces más rápido que el colocado, con 12% de GPU ociosa. Más de 4 casi no suma: con esta proporción de GPUs el trainer ya da abasto.
Mirar las dos columnas de staleness: el promedio sigue a η, pero el máximo lo pasa (10 con η = 4). Son las respuestas largas, que empezaron con pesos viejos y terminaron tarde.
El 3× es de nuestro modelo de juguete; AReaL reporta 2,77× en tiempo de entrenamiento contra su versión sincrónica. La forma es la misma.
Pasar a la siguiente: qué le pasa al aprendizaje.""",
    d.footer("go run ./cmd/sim asinc · simulador Bucle de RL", L, plain=True), gap=32)

# 21 ─ demo aprender
d.slide("demoaprender", D, head("Demo · aprender con lo viejo", "Sin corrección, el gradiente viejo empuja de más", D) +
    row(col(table(["η", "Staleness", "Sin corrección", "Con corrección"], [
        ["0", "0", "0,795", "0,795"], ["1", "0,97", "0,791", "0,796"], ["2", "1,92", "0,695", "0,794"],
        ["4", "3,30", "0,663", "0,794"], ["8", "4,41", "0,205", "0,792"], ["sin límite", "7,00", "0,000", "0,792"],
    ], D, widths=[22, 22, 28, 28], right=(1, 2, 3), size=28), p("Recompensa esperada después de 40 pasos; la mejor posible es 0,799.", D, size=22, color="muted"), gap=12, flex=1.2),
        col(card(p("El óptimo es pensar bastante sin pasar el tope. Un gradiente calculado con la política de hace 4 pasos dice «más largo» cuando ya se pasó: las respuestas se cortan y valen 0.", D, size=26), D, pad=32),
            card(p("Con corrección, cada muestra pesa π<sub>actual</sub>/π<sub>vieja</sub> (truncado): lo que la política actual ya no generaría, pesa menos.", D, size=26, color="ink"), D, accent=True, pad=32), gap=24, flex=1), gap=48, align="center"),
    """Tiempo: 1:08–1:12 (4 min, con demo).
Idea clave: la staleness no es sólo un número de sistemas: con un óptimo en el medio, el gradiente viejo se pasa de largo, y la corrección por importancia lo arregla.
Demo: go run ./cmd/sim aprender. En el simulador, modo Separado con η = 8, primero sin "Corregir lo viejo" (mirar cómo la mediana de tokens se pasa de 8.192 y la recompensa se cae) y después con.
Decir: la tarea del repositorio: el modelo decide cuánto pensar; pensar más acierta más, pero pasar los 8.192 tokens vale cero. El óptimo está en el medio: mediana de unos 3.300 tokens y 6,5% de respuestas cortadas.
Leer la tabla. Con η = 1, nada. Con η = 2 y 4 se nota, y con 8 o sin límite el entrenamiento colapsa. ¿Por qué? Un gradiente calculado con muestras de una política vieja es, en promedio, el gradiente de la política vieja. Cuando la vieja pensaba poco, ese gradiente dice "pensá más", aunque la actual ya esté en el óptimo. Se pasa del tope, las respuestas se cortan, valen cero; y si todas valen cero, GRPO no tiene señal y no se recupera. Es la inestabilidad clásica de los gradientes con demora.
Con corrección por importancia, cada muestra pesa la probabilidad que le da la política actual sobre la que le daba la que la generó, truncado en 2. No es exactamente el objetivo de AReaL, pero es la misma idea: con eso, todos los η llegan a la misma recompensa, a cambio de algunos pasos más.
Pasar a la siguiente: la otra forma de no esperar a la cola.""",
    d.footer("go run ./cmd/sim aprender · simulador Bucle de RL"), gap=40)

# 22 ─ parciales
d.slide("parciales", D, head("Rollouts parciales · Kimi k1.5", "Cortar la generación cuando hay un lote, y seguir después", D) +
    row(col(p("Un presupuesto fijo de tokens por vuelta. Si una respuesta no termina, «la parte sin terminar se guarda en el replay buffer y sigue en la próxima iteración».", D, size=28),
            p("Sólo el pedazo nuevo se calcula con la política actual; los anteriores se reusan. Así «ninguna trayectoria larga monopoliza los recursos del sistema».", D, size=28), gap=24, flex=1.2),
        col(table(["Colocado", "Tiempo", "Ociosa", "Staleness"], [["Sincrónico", "8.692 s", "69%", "0"], ["Parciales", "5.121 s", "47%", "0,97"]],
                  D, widths=[34, 24, 20, 22], right=(1, 2, 3), size=28, bold_rows=(1,)),
            p("La demo, 40 pasos en 8 GPUs. El paper de Kimi no da un número para esto.", D, size=22, color="muted"), gap=16, flex=1), gap=56, align="center"),
    """Tiempo: 1:12–1:15 (3 min).
Idea clave: otra forma de sacarse de encima la cola larga, sin pools separados: cortar la generación cuando ya hay suficiente y dejar lo largo para la vuelta siguiente.
Decir: Kimi k1.5 es colocado: entrenamiento e inferencia comparten las GPUs. Para que una respuesta larga no frene todo, fijan un presupuesto de tokens por vuelta. Si una respuesta pasa ese presupuesto, lo que falta se guarda en el replay buffer y se sigue generando en la próxima iteración. Sólo el pedazo de la iteración actual se calcula con la política actual; los anteriores se reusan del buffer. La frase del paper: así ninguna trayectoria larga monopoliza los recursos.
Es asincronía de otra forma: una respuesta larga queda hecha de pedazos generados por varias versiones, como en las interrumpibles de AReaL. Y aparece la staleness otra vez, concentrada justo en las respuestas largas.
En la demo, el colocado con rollouts parciales: la generación termina cuando hay 32 grupos completos, y lo que está a medias sigue después de entrenar, recalculando su KV cache. 41% menos tiempo, a cambio de staleness de casi 1.
Lo honesto: el paper de Kimi dice que esto reduce mucho el costo, pero no publica una ablación con números.
Pasar a la siguiente: qué pasa cuando algo falla.""",
    d.footer("Kimi Team, arXiv 2501.12599, §2.6.2 · go run ./cmd/sim parciales"), gap=40)

# 23 ─ fallas
d.slide("fallas", D, head("Cuando algo falla", "Cada parte del bucle falla distinto", D) +
    table(["Pieza", "Si se cae", "Qué se hace"], [
        ["Una GPU de rollout", "se pierden respuestas a medias", "se vuelven a generar: no tienen efectos, como un map de MapReduce"],
        ["El verificador", "un sandbox que cuelga o tarda", "timeout y reintento; es un servicio aparte"],
        ["El trainer", "se pierde el estado del optimizador", "checkpoints, como en la Clase 17"],
        ["Los pesos en viaje", "un rollout queda con una versión vieja", "la versión viaja con cada respuesta: se detecta y se acota"],
    ], D, widths=[22, 34, 44], size=28) +
    p("Sincrónico, cualquier falla frena el paso. Asincrónico, una GPU de rollout menos es menos throughput, no una barrera rota.", D, size=28, color="ink"),
    """Tiempo: 1:15–1:18 (3 min).
Idea clave: la parte de generación es fácil de hacer tolerante a fallas porque sus tareas no tienen efectos; la difícil sigue siendo el trainer.
Decir: recorrer la tabla. Una respuesta es una función pura de los pesos, el prompt y la semilla: si una GPU de rollout se cae, se vuelve a pedir, igual que una tarea map que se re-ejecuta en otro nodo. El verificador corre código de los modelos: Kimi describe un servicio de ejecución de código en sandbox como parte del modelo de recompensa; para eso, timeouts y reintentos, y aislarlo del resto. El trainer es un entrenamiento distribuido como el de la Clase 17, con su estado de optimizador: necesita checkpoints. Y los pesos en viaje: cada respuesta lleva la versión con que se generó, y eso permite tanto acotar la staleness como detectar a un worker que se quedó atrás.
Lo que cambia la asincronía: en un sistema sincrónico, una GPU de rollout que se cae en el medio del paso frena a todas, como un straggler infinito. En uno asincrónico, el controlador simplemente tiene un worker menos.
Ejercicio 4 del repositorio.
Pasar a la siguiente: cerrar la serie.""",
    d.footer("Kimi Team, 2501.12599, §2.6 · Clase 17 · ejercicio 4 de post-entrenamiento"), gap=40, section="Cerrar la serie.")

# 24 ─ serie
d.slide("serie", D, head("Las mismas tres preguntas", "Qué se parte, qué se comunica, qué pasa si falla", D) +
    table(["", "Entrenar (16–17)", "Servir (18–19)", "El bucle de RL (22)"], [
        ["Qué se parte", "el modelo, en 4 dimensiones", "el KV cache, en bloques y en niveles", "el clúster, entre generar y entrenar"],
        ["Qué se comunica", "gradientes y activaciones", "KV cache entre prefill y decode", "pesos, después de cada paso"],
        ["A quién se espera", "al straggler, en cada all-reduce", "al prefill largo, en cada iteración", "a la respuesta más larga, en cada paso"],
        ["Si algo falla", "volver al checkpoint", "recalcular el pedido", "regenerar; checkpoint del trainer"],
    ], D, widths=[19, 27, 27, 27], size=26),
    """Tiempo: 1:18–1:22 (4 min).
Idea clave: el bucle de RL no trae problemas nuevos: trae los de entrenar y los de servir al mismo tiempo, y uno nuevo que es la versión de los datos.
Decir: recorrer por filas. Qué se parte: en entrenamiento, el modelo en datos, tensores, pipeline y contexto; en serving, el KV cache en bloques y en niveles; en RL, además, el clúster entre los dos trabajos, y cada uno con su propio layout del mismo modelo.
Qué se comunica: gradientes y activaciones; el KV cache entre prefill y decode o desde el pool; y en RL los pesos, que viajan de un layout al otro o de un pool al otro después de cada paso.
A quién se espera: siempre hay una barrera y un más lento. En entrenamiento, la GPU lenta en el all-reduce. En serving, el prefill largo que frena a los decodes. En RL, la respuesta más larga del lote. Y las soluciones se parecen: no esperar (asincronía), acotar cuánto se adelanta el rápido (SSP), partir lo largo (prefill en pedazos, rollouts parciales).
Si algo falla: checkpoint para el estado que no se puede recalcular; recálculo para lo que sí.
Pasar a la siguiente: qué llevarse.""",
    d.footer())

# 25 ─ takeaways
d.slide("takeaways", D, head("Para llevarse", "Entrenar con lo que se sirve", D) +
    row(card(label("No esperar a la cola", D, color="acc") + p("La respuesta más larga decide el paso sincrónico. Rollouts parciales o asincrónicos sacan esa espera.", D, size=28), D),
        card(label("Mover los pesos bien", D, color="acc") + p("Colocado: cambiar de layout juntando sólo lo que falta. Separado: broadcast en anillo, no uno por uno.", D, size=28), D),
        card(label("Acotar lo viejo", D, color="acc") + p("Staleness acotada, como SSP, y una corrección para que los datos viejos no empujen de más.", D, size=28), D), gap=32) +
    p("Dieciséis a veintidós: el mismo modelo, partido para entrenar, servido para generar, y llevado de un lado al otro miles de veces.", D, size=30, color="ink"),
    """Tiempo: 1:22–1:24 (2 min).
Idea clave: tres ideas para el bucle de RL, y una frase para toda la serie.
Decir: no esperar a la cola: con respuestas de largo de cola larga, un paso sincrónico espera casi todo el tiempo; partir las largas (Kimi) o no esperar (AReaL) recupera la mayor parte. Mover los pesos bien: si se comparte GPUs, la transición entre layouts es un all-gather y se puede diseñar para mover poco (HybridFlow); si se separan, es un broadcast y tiene que ser en anillo o en pipeline. Acotar lo viejo: la asincronía hace que se entrene con datos de pesos viejos; se acota cuánto, como en SSP, y se corrige el objetivo para que lo viejo no tire de más.
Cerrar la serie: en la 16 partimos un modelo para entrenarlo. En la 18 lo servimos. En la 22 vimos que, para los modelos de razonamiento de hoy, entrenar es servir miles de veces: el mismo modelo, en dos layouts, yendo y viniendo. Todo lo de las seis clases anteriores aparece en ese bucle.
Pasar a la siguiente: el repositorio.""",
    d.footer(), gap=48)

# 26 ─ demo
cmds = [("go run ./cmd/sim cola", "un paso espera a la más larga"), ("go run ./cmd/sim ubicacion", "colocado vs separado, y mover los pesos"),
        ("go run ./cmd/sim asinc", "staleness acotada: throughput y espera"), ("go run ./cmd/sim parciales", "rollouts parciales e interrumpibles"),
        ("go run ./cmd/sim aprender", "qué le hace lo viejo al aprendizaje"), ("go test ./...", "los tests de lo que dijimos")]
lines = "".join(f'<div style="display:flex;gap:48px">{p(c, D, size=30, color="ink", mono=True, width=720)}{p(w, D, size=30, color="muted")}</div>' for c, w in cmds)
d.slide("demo", D, head("El repositorio", "post-entrenamiento", D) +
    f'<div style="display:flex;flex-direction:column;gap:18px;background:#0A0C0F;border:1px solid {D["line"]};border-radius:16px;padding:44px">{lines}</div>' +
    p("Paquetes: tarea (cuánto pensar, GRPO), pesos (broadcast y transición de layout) y bucle (el sistema en tiempo virtual). Y el simulador Bucle de RL.", D, size=28),
    """Tiempo: 1:24–1:27 (3 min).
Idea clave: el repositorio es el bucle entero en tiempo virtual: GPUs de rollout con batching continuo, un trainer, los pesos que viajan, y un problema de juguete que aprende de verdad.
Decir: el paquete tarea es el problema: una política que elige cuánto pensar, una recompensa por verificador y GRPO. El paquete pesos tiene el costo de mandar los pesos (ingenuo y en anillo) y las fórmulas de la tabla 2 de HybridFlow. El paquete bucle es la simulación: colocado o separado, sincrónico o asincrónico con η, rollouts parciales e interrumpibles, con o sin corrección.
Los tests verifican lo de la clase: que un paso sincrónico espera a la respuesta más larga; que con η = 0 no hay staleness; que más η da más throughput y más staleness; que el asincrónico termina en menos de la mitad del tiempo con la misma recompensa; que sin corrección η = 8 arruina el aprendizaje y con corrección no; que los parciales y las interrupciones aceleran; y que las respuestas se alargan al aprender.
El simulador Bucle de RL muestra todo en vivo, con un slider de η y un botón Comparar.
Pasar a la siguiente: preguntas.""",
    d.footer("simulador Bucle de RL · go test ./..."))

# 27 ─ preguntas
qs = ["¿Por qué un paso de RL tiene una carga de serving y una de entrenamiento? ¿Qué depende de qué?",
      "¿Qué le saca GRPO a PPO, y por qué eso importa para el sistema?",
      "¿Por qué un paso sincrónico tiene la GPU ociosa la mayor parte del tiempo aunque todas «estén trabajando»?",
      "Colocado o separado: ¿qué se paga en cada caso para llevar los pesos a donde se genera?",
      "¿Qué es η en AReaL y qué es s en SSP? ¿Por qué el máximo medido puede pasar a η?"]
d.slide("preguntas", D, head("Para repasar", "Preguntas y lecturas", D) +
    row(f'<ol style="flex:1.4;font-size:28px;line-height:1.4;color:{D["ink"]}">' + "".join(f"<li>{q}</li>" for q in qs) + "</ol>",
        card(h3("Lecturas", D) + p("<b>Sheng et al.</b>, HybridFlow, EuroSys 2025.", D, size=24, color="ink") +
             p("DeepSeek-AI, DeepSeek-R1, 2025. Kimi Team, Kimi k1.5, 2025.", D, size=24) +
             p("Fu et al., AReaL, 2025. Ho et al., SSP, NIPS 2013.", D, size=24), D, flex=1, pad=40, gap=20), gap=64),
    """Tiempo: 1:27–1:30 (3 min).
Idea clave: cerrar con las preguntas de repaso. Las respuestas cortas, para tener a mano:
1. Generar las respuestas es decode con el modelo actual, como un servidor de inferencia; actualizar es un paso de entrenamiento. El entrenamiento necesita las respuestas, y la generación siguiente necesita los pesos nuevos.
2. El crítico, un modelo típicamente del tamaño de la política: la ventaja se calcula contra el promedio del grupo. Es un modelo menos para entrenar, guardar en memoria y mover; a cambio, el trainer necesita el grupo completo.
3. Porque el paso espera a la respuesta más larga y el decode cuesta casi lo mismo con una secuencia que con 64: las GPUs que quedan con pocas respuestas están ocupadas pero producen poco, y las que terminaron esperan.
4. Colocado: la transición del layout de entrenamiento al de generación, un all-gather que se puede diseñar para mover poco (3D-HybridEngine). Separado: un broadcast de todos los pesos a las GPUs de rollout después de cada paso, que tiene que ser en anillo o en pipeline para escalar.
5. η es cuántas versiones puede adelantarse la generación respecto del trainer; s, cuántos relojes puede adelantarse el worker más rápido respecto del más lento. Con 0, los dos son sincrónicos. El límite se aplica cuando una respuesta empieza; una respuesta larga que empezó temprano termina tarde y se entrena con más staleness.
Lecturas: HybridFlow es el paper de la clase. Complementarias: R1 (GRPO y por qué RL), Kimi k1.5 (rollouts parciales y despliegue híbrido), AReaL (asincronía con staleness acotada) y SSP. Los apuntes tienen más preguntas.""",
    d.footer())

d.write_index("Clase 22 — Entrenar con lo que se sirve")
print(len(d.order), "slides")
