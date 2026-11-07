"""Pasa un deck al estilo de la materia, en oscuro.

Fondo oscuro sobrio, títulos en Chakra Petch, texto en Rubik y mono en IBM Plex Mono (las fuentes de
fiubata050.github.io), un turquesa apagado como único acento, separadores por sección y número de slide.
Se aplica sobre la carpeta de un deck (deck.json + slides/*.html) y se puede correr más de una vez.
"""
import html, json, os, re

BG, INSET, CARD, LINE = "#0F1416", "#0B0F11", "#161D20", "#263034"
INK, BODY, MUTED = "#E8ECEE", "#B4BEC3", "#7F8B91"
ACCENT, ACCENT_FILL = "#3AA6B9", "#1F6E7C"
SANS = "Rubik, Arial, sans-serif"
MONO = "'IBM Plex Mono', monospace"
HEAD = "'Chakra Petch', Arial, sans-serif"

# Paleta vieja (slides oscuras y claras) -> paleta nueva. FILLS sólo para fondos: un color de acento
# usado como relleno va más oscuro, para que el texto encima se lea.
COLORS = {
    "#F2F3EF": INK, "#A8B3BC": BODY, "#7D8892": MUTED, "#161B20": CARD, "#2A3138": LINE,
    "#34D399": ACCENT, "#F5A524": "#E0A050", "#232A31": "#222B2F", "#7AA2FF": "#8FAEE8", "#F5B544": "#E8B86A",
    "#3A434C": "#3A464C", "#4A5560": "#4E5A60", "#3D2C10": "#2A2114", "#173B2F": "#12292D", "#132A22": "#12292D",
    "#0A0C0F": INSET, "#F472B6": "#D88AAE", "#1A2027": "#1C2428", "#5A6570": "#5E6A70", "#1F7A5C": ACCENT_FILL,
    "#F3F2EE": BG, "#48525C": BODY, "#66707A": MUTED, "#FBFBF8": CARD, "#D6D7D2": LINE,
    "#08724F": ACCENT, "#C2410C": "#E0A050", "#2C56C9": "#8FAEE8", "#DDF3EA": "#12292D", "#E6E6E0": "#222B2F",
}
FILLS = {"#34D399": ACCENT_FILL, "#08724F": ACCENT_FILL, "#1F7A5C": ACCENT_FILL, "#F5A524": "#8A5A1E", "#C2410C": "#8A5A1E",
         "#7AA2FF": "#3E5C99", "#2C56C9": "#3E5C99", "#F5B544": "#9C7A3C", "#F472B6": "#8E4A6B"}
OLD_BG = "#0E1114"  # fondo de las slides oscuras, texto de las claras y texto sobre rellenos de color

# Títulos de los separadores: deck -> {slide donde empieza la sección: título}.
DIVIDERS = {
    16: {"datos": "Paralelismo de datos", "memoria": "Dónde se va la memoria", "tensores": "Partir el modelo",
         "ptdp": "PTD-P: las tres dimensiones", "fallas": "Fallas y checkpoints"},
    17: {"cuatrod": "Paralelismo 4D en Llama 3", "moe": "DeepSeek-V3: expertos", "fallas": "Las fallas son el estado normal",
         "multidc": "Más allá de un datacenter"},
    18: {"kvcache": "El KV cache", "estatico": "Batching por iteración", "desperdicio": "PagedAttention",
         "cow": "Compartir y desalojar", "sse": "El stream de vuelta"},
    19: {"hash": "El caché de prefijos", "router": "El router", "interferencia": "Prefill y decode",
         "mooncake": "Mooncake", "moeserve": "La frontera"},
    20: {"tamanos": "Anatomía de un arranque en frío", "estimar": "Dónde arrancar", "lora": "Muchos fine-tunes",
         "lente": "La lente de la materia"},
    21: {"gang": "Todo o nada", "localidad": "Localidad y fragmentación", "las": "Sin conocer las duraciones",
         "gandiva": "Compartir el clúster"},
    22: {"cola": "El rollout domina", "ubicaciones": "Dónde va cada trabajo", "asinc": "Asincrónico",
         "fallas": "Cerrar la serie"},
}


def _lum(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _fill(old):
    old = old.upper()
    return FILLS.get(old) or COLORS.get(old) or (INSET if old == OLD_BG else old)


def _recolor(s):
    def text_color(m):
        bgs = re.findall(r"background:(#[0-9A-Fa-f]{6})", s[:m.start()])[1:]  # el primero es el de la sección
        return m.group(1) + (BG if bgs and _lum(_fill(bgs[-1])) > 0.5 else INK)

    s = re.sub(r"((?<![-\w])color:)#0E1114", text_color, s)
    s = s.replace(f"solid {OLD_BG}", f"solid {BG}")
    head, rest = s.split(">", 1)
    head = re.sub(r"background:#[0-9A-Fa-f]{6}", f"background:{BG}", head)
    rest = re.sub(r"(background:|fill=\")(#[0-9A-Fa-f]{6})", lambda m: m.group(1) + _fill(m.group(2)), rest)
    s = head + ">" + rest
    return re.sub(r"#[0-9A-Fa-f]{6}\b", lambda m: COLORS.get(m.group(0).upper(), m.group(0)), s)


def _fonts(s):
    s = s.replace("font-family:'Geist Mono', 'Courier New', monospace", f"font-family:{MONO}")
    s = s.replace("font-family:Geist, Arial, sans-serif", f"font-family:{SANS}")

    def heading(m):
        tag, style = m.group(1), m.group(2)
        style = re.sub(r"letter-spacing:-?[\d.]+(px|em);?", "", style)
        style = re.sub(r"font-family:[^;]+;?", "", style)
        weight = "font-weight:600" if tag == "h3" else "font-weight:700"
        return f'<{tag} style="font-family:{HEAD};{style.replace("font-weight:600", weight)};letter-spacing:-0.5px"'

    return re.sub(r'<(h1|h2|h3) style="([^"]*)"', heading, s)


def _strip(s):
    return re.sub(r"<[^>]+>", "", s).strip()


SERIES = [(16, "Entrenamiento distribuido"), (17, "Escala de frontera"), (18, "Servir un modelo"), (19, "Del motor al clúster"),
          (20, "Muchos modelos"), (21, "Repartir GPUs"), (22, "Entrenar con lo que se sirve")]


def cover(n, title, subtitle, paper, notes):
    series = "".join(
        f'<div style="flex:1;padding-top:16px;border-top:{"4px solid " + ACCENT if k == n else "1px solid " + LINE}">'
        f'<p style="font-family:{MONO};font-size:22px;color:{ACCENT if k == n else MUTED}">{k}</p>'
        f'<p style="font-size:20px;line-height:1.3;color:{INK if k == n else MUTED}">{name}</p></div>' for k, name in SERIES)
    return f'''<section id="cover" data-estilo="curso" style="background:{BG};color:{INK};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:space-between">
<div style="display:flex;justify-content:space-between;align-items:baseline">
<p style="font-family:{MONO};font-size:24px;letter-spacing:2px;text-transform:uppercase;color:{BODY}">TA050 · Sistemas Distribuidos I · FIUBA</p>
<p style="font-family:{MONO};font-size:24px;letter-spacing:2px;text-transform:uppercase;color:{ACCENT}">Clase {n}</p>
</div>
<div style="display:flex;flex-direction:column;gap:36px">
<div style="width:120px;height:6px;background:{ACCENT}"></div>
<h1 data-title style="font-family:{HEAD};font-size:116px;font-weight:700;line-height:1.02;letter-spacing:-1px;color:{INK};max-width:1600px">{title}</h1>
<p data-sub style="font-size:40px;line-height:1.35;color:{BODY};max-width:1400px">{subtitle}</p>
<p data-paper style="font-size:24px;line-height:1.4;color:{MUTED};max-width:1500px">{paper}</p>
</div>
<div style="display:flex;gap:24px">{series}</div>
<aside>{notes}</aside>
</section>
'''


def divider(sid, part, title, desc):
    return f'''<section id="{sid}" data-estilo="curso" style="background:{INSET};color:{INK};font-family:{SANS};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:36px">
<p style="font-family:{MONO};font-size:26px;letter-spacing:3px;text-transform:uppercase;color:{ACCENT}">Parte {part}</p>
<h2 style="font-family:{HEAD};font-size:104px;font-weight:700;line-height:1.05;letter-spacing:-1px;color:{INK};max-width:1500px">{title}</h2>
<div style="width:120px;height:6px;background:{ACCENT}"></div>
{f'<p style="font-size:36px;line-height:1.4;color:{BODY};max-width:1300px">{desc}</p>' if desc else ""}
<aside>Separador de sección. {html.escape(title)}: {html.escape(desc)}</aside>
</section>
'''


def _number(s, i):
    s = re.sub(r'\n?<p data-pnum[^>]*>\d+</p>', "", s)
    pnum = f'<p data-pnum style="position:absolute;right:64px;bottom:44px;font-family:{MONO};font-size:24px;color:{MUTED}">{i}</p>'
    return s.replace("\n<aside>", f"\n{pnum}\n<aside>", 1) if "\n<aside>" in s else s.replace("<aside>", pnum + "<aside>", 1)


def restyle(deck_dir, num):
    path = f"{deck_dir}/deck.json"
    deck = json.load(open(path))
    order, sections = deck["order"], deck["sections"]
    read = lambda sid: open(f"{deck_dir}/slides/{sid}.html").read()
    write = lambda sid, s: open(f"{deck_dir}/slides/{sid}.html", "w").write(s)

    c = read("cover")
    ps = re.findall(r"<p style=\"([^\"]*)\">(.*?)</p>", c, re.S)
    if "data-estilo" in c:  # ya convertida: se rearma con sus propios campos
        title = re.search(r"<h1 data-title[^>]*>(.*?)</h1>", c, re.S).group(1)
        subtitle = re.search(r"<p data-sub[^>]*>(.*?)</p>", c, re.S).group(1)
        paper = re.search(r"<p data-paper[^>]*>(.*?)</p>", c, re.S).group(1)
    else:
        title = _strip(re.search(r"<h1[^>]*>(.*?)</h1>", c, re.S).group(1))
        subtitle = next(t for st, t in ps if "font-size:44px" in st)
        paper = next(t for st, t in ps if t.startswith("Paper"))
    notes = re.search(r"<aside>(.*?)</aside>", c, re.S).group(1)
    write("cover", cover(num, title, subtitle, paper, notes))

    for sid in order:
        if sid == "cover" or sid.startswith("div-"):
            continue
        s = read(sid)
        if "data-estilo" in s:
            continue
        s = _fonts(_recolor(s))
        s = s.replace("<section ", '<section data-estilo="curso" ', 1)
        write(sid, s)

    for part, (key, sec) in enumerate(sections.items(), 1):
        start = sec["start"].removeprefix("div-")
        title = DIVIDERS.get(num, {}).get(start)
        if not title:
            continue
        sid = f"div-{start}"
        if sid not in order:
            order.insert(order.index(start), sid)
        desc = sec["description"].split(": ", 1)[-1] if ": " in sec["description"] else sec["description"]
        if desc.lower().startswith(title.lower()):
            desc = ""
        write(sid, divider(sid, part, title, desc[:1].upper() + desc[1:]))
        sec["start"] = sid

    for i, sid in enumerate(order, 1):
        s = read(sid)
        if sid != "cover":
            write(sid, _number(s, i))
    json.dump(deck, open(path, "w"), ensure_ascii=False, indent=2)
