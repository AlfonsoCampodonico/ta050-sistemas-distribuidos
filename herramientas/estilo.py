"""Pasa un deck al estilo de las slides de la materia (fiubata050.github.io).

Slides claras (#F3F4F5), títulos en Chakra Petch con sombra turquesa, texto en Rubik,
mono en IBM Plex Mono, separadores oscuros por sección y número de slide abajo a la derecha.
Se aplica sobre la carpeta de un deck (deck.json + slides/*.html) y se puede correr más de una vez.
"""
import html, json, os, re

TEAL, GLOW, BG, INK = "#046B7D", "#25F6FE", "#F3F4F5", "#0D0D0D"
SANS = "Rubik, Arial, sans-serif"
MONO = "'IBM Plex Mono', monospace"
HEAD = "'Chakra Petch', Arial, sans-serif"
DARK_BG = "linear-gradient(180deg,#010101 0%,#03181A 100%)"
GLOW_SHADOW = "0 0 0.6em rgba(37,246,254,.5),0 0 0.22em rgba(37,246,254,.7),0.04em 0.04em 0 rgba(0,0,0,.7)"

# Paleta vieja (oscura y clara) -> paleta de la materia.
COLORS = {
    "#F2F3EF": INK, "#A8B3BC": "#33393E", "#7D8892": "#6C757D", "#161B20": "#FFFFFF", "#2A3138": "#DEE2E6",
    "#34D399": TEAL, "#F5A524": "#B45309", "#232A31": "#E9ECEF", "#7AA2FF": "#2C56C9", "#F5B544": "#E9A23B",
    "#3A434C": "#C3CAD0", "#4A5560": "#ADB5BD", "#3D2C10": "#FDF3E0", "#173B2F": "#E3EFF1", "#132A22": "#E3EFF1",
    "#0A0C0F": "#E9ECEF", "#F472B6": "#C0307F", "#1A2027": "#EEF0F2", "#5A6570": "#868E96", "#1F7A5C": "#035A69",
    "#F3F2EE": BG, "#48525C": "#33393E", "#66707A": "#6C757D", "#FBFBF8": "#FFFFFF", "#D6D7D2": "#DEE2E6",
    "#08724F": TEAL, "#C2410C": "#B45309", "#2C56C9": "#2C56C9", "#DDF3EA": "#E3EFF1", "#E6E6E0": "#E9ECEF",
}
OLD_BG = "#0E1114"  # fondo de las slides oscuras y, a la vez, texto sobre rellenos de color

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


def _on(bg_old):
    """Color de texto para lo que antes iba en #0E1114 sobre un relleno."""
    new = COLORS.get(bg_old.upper(), BG if bg_old.upper() == OLD_BG else bg_old)
    return "#FFFFFF" if _lum(new) < 0.45 else INK


def _recolor(s):
    def text_color(m):
        before = s[:m.start()]
        bgs = re.findall(r"background:(#[0-9A-Fa-f]{6})", before)
        inner = [b for b in bgs[1:]]  # el primero es el de la sección
        return m.group(1) + (_on(inner[-1]) if inner else INK)

    s = re.sub(r"((?<![-\w])color:)#0E1114", text_color, s)
    s = s.replace(f"solid {OLD_BG}", f"solid {BG}")
    head, rest = s.split(">", 1)
    head = head.replace(f"background:{OLD_BG}", f"background:{BG}")
    rest = rest.replace(f"background:{OLD_BG}", "background:#FFFFFF")
    s = head + ">" + rest
    return re.sub(r"#[0-9A-Fa-f]{6}\b", lambda m: COLORS.get(m.group(0).upper(), m.group(0)), s)


def _fonts(s):
    s = s.replace("font-family:'Geist Mono', 'Courier New', monospace", f"font-family:{MONO}")
    s = s.replace("font-family:Geist, Arial, sans-serif", f"font-family:{SANS}")

    def heading(m):
        tag, style = m.group(1), m.group(2)
        style = re.sub(r"letter-spacing:-?[\d.]+(px|em);?", "", style)
        style = re.sub(r"font-family:[^;]+;?", "", style)
        if tag == "h3":
            return f'<h3 style="font-family:{HEAD};{style.replace("font-weight:600", "font-weight:700")}"'
        style = style.replace("font-weight:600", "font-weight:700")
        return f'<{tag} style="font-family:{HEAD};{style};letter-spacing:-0.5px;text-shadow:3px 3px 0 {TEAL}"'

    return re.sub(r'<(h1|h2|h3) style="([^"]*)"', heading, s)


def _strip(s):
    return re.sub(r"<[^>]+>", "", s).strip()


def cover(n, title, subtitle, paper, notes):
    nodes = [(110, 300), (280, 190), (230, 470), (430, 360), (180, 700), (560, 230), (520, 600), (360, 880), (620, 820), (90, 960)]
    links = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (3, 6), (4, 6), (1, 5), (6, 8), (4, 7), (7, 8), (4, 9), (7, 9)]
    svg = "".join(f'<line x1="{nodes[a][0]}" y1="{nodes[a][1]}" x2="{nodes[b][0]}" y2="{nodes[b][1]}" stroke="{GLOW}" stroke-opacity=".35" stroke-width="2"/>' for a, b in links)
    svg += "".join(f'<circle cx="{x}" cy="{y}" r="{10 if i % 3 else 16}" fill="#03181A" stroke="{GLOW}" stroke-width="3"/>' for i, (x, y) in enumerate(nodes))
    return f'''<section id="cover" data-estilo="curso" style="background:{DARK_BG};color:#FFFFFF;font-family:{SANS};font-weight:300;padding:128px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden">
<div style="position:absolute;inset:0;background:radial-gradient(900px 600px at 30% 70%,rgba(37,246,254,.16),transparent 70%)"></div>
<svg style="position:absolute;left:0;top:0" width="1100" height="1080" viewBox="0 0 1100 1080">{svg}</svg>
<div style="position:relative;display:flex;flex-direction:column;align-items:flex-end;gap:28px;text-align:right">
<h1 style="font-family:{HEAD};font-size:128px;font-weight:700;line-height:.95;text-transform:uppercase;color:#FFFFFF;text-shadow:{GLOW_SHADOW}">Sistemas<br>distribuidos</h1>
<p style="font-family:{HEAD};font-size:44px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:{GLOW};text-shadow:0 0 .6em rgba(37,246,254,.5),.03em .03em 0 rgba(0,0,0,.85)">Clase {n} — {title}</p>
</div>
<div style="position:relative;display:flex;flex-direction:column;align-items:flex-end;gap:24px;text-align:right">
<p style="width:1100px;font-size:36px;line-height:1.35;color:#DEE2E6">{subtitle}</p>
<p style="width:1100px;font-size:24px;line-height:1.4;color:#ADB5BD">{paper}</p>
</div>
<aside>{notes}</aside>
</section>
'''


def divider(sid, n, title, desc):
    return f'''<section id="{sid}" data-estilo="curso" style="background:{DARK_BG};color:#FFFFFF;font-family:{SANS};font-weight:300;padding:128px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:40px;text-align:center">
<h2 style="font-family:{HEAD};font-size:120px;font-weight:700;line-height:1.1;text-transform:uppercase;color:#FFFFFF;text-shadow:3px 3px 0 #000;max-width:1600px">{title}</h2>
{f'<p style="font-size:36px;line-height:1.4;color:#ADB5BD;max-width:1300px">{desc}</p>' if desc else ""}
<aside>Separador de sección. {html.escape(title)}: {html.escape(desc)}</aside>
</section>
'''


def _number(s, i, dark):
    s = re.sub(r'\n?<p data-pnum[^>]*>\d+</p>', "", s)
    color = "#6C757D" if dark else TEAL
    pnum = f'<p data-pnum style="position:absolute;right:64px;bottom:44px;font-family:{MONO};font-size:24px;color:{color}">{i}</p>'
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
        title = next(t for st, t in ps if "letter-spacing:.08em" in st).split(" — ", 1)[1]
        subtitle = next(t for st, t in ps if "font-size:36px" in st)
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
        s = re.sub(r'(<section [^>]*style=")', r"\1font-weight:300;", s, count=1)
        write(sid, s)

    for key, sec in sections.items():
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
        write(sid, divider(sid, num, title, desc[:1].upper() + desc[1:]))
        sec["start"] = sid

    for i, sid in enumerate(order, 1):
        s = read(sid)
        if sid != "cover":
            write(sid, _number(s, i, sid.startswith("div-")))
    json.dump(deck, open(path, "w"), ensure_ascii=False, indent=2)
