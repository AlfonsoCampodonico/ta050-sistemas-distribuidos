"""Restyle TA050 decks to the new look (Geist + Geist Mono, dark-first, mint/amber).

Colors are mapped by role: backgrounds and borders through per-theme tables,
text by its role (strong/body/muted/accent/warn) against the *target*
background it sits on, so contrast is chosen, not inherited by accident.
Usage: python3 restyle.py <deckdir>
"""
import re, sys, glob
from bs4 import BeautifulSoup

DARK_BG, LIGHT_BG = "#0E1114", "#F3F2EE"

BG = {
    "L2D": {"#F6F4EF": DARK_BG, "#FCFBF8": "#161B20", "#E4DFD3": "#232A31", "#E7E2D8": "#232A31",
            "#9AA5AF": "#4A5560", "#2558B8": "#34D399", "#9DB5E8": "#1F7A5C", "#DCE5F5": "#173B2F",
            "#F8E3D0": "#3D2C10", "#F2B27A": "#F5B544", "#E08A3C": "#F5A524", "#13212C": "#1A2027",
            "#0C161E": "#1A2027"},
    "D2L": {"#13212C": LIGHT_BG, "#0C161E": "#0E1114", "#2558B8": "#34D399", "#E08A3C": "#F5A524",
            "#FCFBF8": "#FBFBF8", "#9AA5AF": "#9AA3AB", "#F6F4EF": LIGHT_BG},
}
BORDER = {
    "L2D": {"#DDD8CC": "#2A3138", "#CFC9BC": "#2A3138", "#B9B3A6": "#3A434C", "#9AA5AF": "#4A5560",
            "#2558B8": "#34D399", "#C76A1E": "#F5A524", "#F6F4EF": DARK_BG, "#2A3B48": "#2A3138",
            "#4A5568": "#5A6570"},
    "D2L": {"#2A3B48": "#2A3138", "#2558B8": "#08724F", "#F6F4EF": LIGHT_BG, "#C76A1E": "#9A5B00",
            "#9AA5AF": "#9AA3AB", "#DDD8CC": "#D6D7D2"},
}
ROLE = {"#1B2A36": "strong", "#13212C": "strong", "#F6F4EF": "strong", "#3A2410": "strong",
        "#4A5568": "body", "#BFD0DA": "body", "#6B7480": "muted", "#8A9BA8": "muted", "#9AA5AF": "muted",
        "#2558B8": "accent", "#E08A3C": "warn", "#A3530F": "warn", "#C76A1E": "warn"}
ON_DARK = {"strong": "#F2F3EF", "body": "#A8B3BC", "muted": "#7D8892", "accent": "#34D399", "warn": "#F5A524"}
ON_LIGHT = {"strong": "#0E1114", "body": "#48525C", "muted": "#66707A", "accent": "#08724F", "warn": "#C2410C"}
FILLS = {"#34D399", "#F5A524", "#F5B544"}  # saturated fills: text on them is always strong

SANS = "Geist, Arial, sans-serif"
MONO = "'Geist Mono', 'Courier New', monospace"


def lum(h):
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def parse(style):
    out = []
    for part in style.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out.append([k.strip(), v.strip()])
    return out


def dump(props):
    return ";".join(f"{k}:{v}" for k, v in props)


def get(props, key):
    for k, v in props:
        if k == key:
            return v
    return None


def setp(props, key, val):
    for p in props:
        if p[0] == key:
            p[1] = val
            return
    props.append([key, val])


def hexes(v):
    return re.findall(r"#[0-9A-Fa-f]{6}", v)


def restyle(path):
    soup = BeautifulSoup(open(path).read(), "html.parser")
    sec = soup.find("section")
    sp = parse(sec.get("style", ""))
    src_bg = hexes(get(sp, "background") or "#F6F4EF")[0].upper()
    theme = "D2L" if src_bg == "#13212C" else "L2D"
    target_bg = {}  # element id -> target background

    def tbg(el):
        while el is not None and el.name:
            if id(el) in target_bg:
                return target_bg[id(el)]
            el = el.parent
        return DARK_BG if theme == "L2D" else LIGHT_BG

    # pass 1: backgrounds and borders (parents before children: document order)
    for el in [sec] + sec.find_all(True):
        st = el.get("style")
        if not st:
            continue
        props = parse(st)
        for p in props:
            k, v = p
            if k == "background" or k == "background-color":
                for h in hexes(v):
                    new = BG[theme].get(h.upper())
                    if new is None:
                        print(f"  ? {path}: background {h}", file=sys.stderr)
                        continue
                    v = v.replace(h, new)
                p[1] = v
                hs = hexes(v)
                if hs:
                    target_bg[id(el)] = hs[0]
            elif k.startswith("border") and "radius" not in k:
                for h in hexes(v):
                    new = BORDER[theme].get(h.upper())
                    if new is None:
                        print(f"  ? {path}: border {h}", file=sys.stderr)
                        continue
                    v = v.replace(h, new)
                p[1] = v
        el["style"] = dump(props)

    # pass 2: text color by role against the target background
    for el in [sec] + sec.find_all(True):
        st = el.get("style")
        if not st:
            continue
        props = parse(st)
        c = get(props, "color")
        if c and hexes(c):
            h = hexes(c)[0].upper()
            role = ROLE.get(h)
            if role is None:
                print(f"  ? {path}: color {h}", file=sys.stderr)
            else:
                bg = tbg(el)
                if theme == "D2L" and role == "warn":
                    role = "accent"  # orange was the only accent on the old dark slides
                if bg.upper() in FILLS and role in ("accent", "warn", "muted", "body"):
                    role = "strong" if role in ("accent", "warn") else role
                pal = ON_DARK if lum(bg) < 0.25 else ON_LIGHT
                setp(props, "color", pal[role])
        # typefaces
        ff = get(props, "font-family")
        if ff:
            if "Mono" in ff:
                setp(props, "font-family", MONO)
            else:
                setp(props, "font-family", SANS)
        # eyebrows: mono, lighter, tighter tracking
        if get(props, "text-transform") == "uppercase":
            setp(props, "font-family", MONO)
            setp(props, "font-weight", "500")
            setp(props, "letter-spacing", "2px")
        # headings: tighter
        if el.name in ("h1", "h2"):
            fs = get(props, "font-size")
            size = int(fs.rstrip("px")) if fs else 72
            setp(props, "letter-spacing", "-4px" if size >= 100 else "-2px")
            setp(props, "line-height", "1.05")
        if el.name == "h3":
            setp(props, "letter-spacing", "-0.5px")
        # footer band: mono
        if get(props, "position") == "absolute" and get(props, "bottom") == "64px":
            setp(props, "font-family", MONO)
        el["style"] = dump(props)
    open(path, "w").write(str(soup))
    return theme


if __name__ == "__main__":
    for f in sorted(glob.glob(sys.argv[1] + "/project/slides/*.html")):
        print(f.split("/")[-1], restyle(f))
