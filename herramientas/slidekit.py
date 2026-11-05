"""Plantillas de slides en el estilo nuevo de TA050 (Geist, oscuro primero)."""
import json, os

MONO = "'Geist Mono', 'Courier New', monospace"
SANS = "Geist, Arial, sans-serif"
DARK = dict(bg="#0E1114", ink="#F2F3EF", body="#A8B3BC", muted="#7D8892", card="#161B20", line="#2A3138",
            acc="#34D399", warn="#F5A524", blue="#7AA2FF", accbg="#132A22", fill="#232A31", onfill="#0E1114")
LIGHT = dict(bg="#F3F2EE", ink="#0E1114", body="#48525C", muted="#66707A", card="#FBFBF8", line="#D6D7D2",
             acc="#08724F", warn="#C2410C", blue="#2C56C9", accbg="#DDF3EA", fill="#E6E6E0", onfill="#0E1114")


class Deck:
    def __init__(self, root, num, short):
        self.root, self.num, self.short = root, num, short
        self.order, self.sections = [], {}
        os.makedirs(f"{root}/project/slides", exist_ok=True)

    def footer(self, src=None, t=DARK, plain=False):
        text = src if plain else f"TA050 · Clase {self.num} · {self.short}" + (f" · {src}" if src else "")
        return (f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-family:{MONO};'
                f'font-size:24px;color:{t["muted"]}">{text}</p>')

    def slide(self, sid, t, body, notes, footer="", gap=48, justify=None, transition=None, section=None):
        assert len(notes) <= 4000, (sid, len(notes))
        just = f";justify-content:{justify}" if justify else ""
        tr = f' data-transition="{transition}"' if transition else ""
        html = (f'<section id="{sid}"{tr} style="background:{t["bg"]};color:{t["ink"]};font-family:{SANS};'
                f'padding:128px 128px 160px;display:flex;flex-direction:column;gap:{gap}px{just}">\n'
                f'{body}\n{footer}\n<aside>{notes.strip()}</aside>\n</section>\n')
        open(f"{self.root}/project/slides/{sid}.html", "w").write(html)
        self.order.append(sid)
        if section:
            self.sections[f"s{len(self.sections) + 1}"] = {"description": section, "start": sid}

    def raw(self, sid, html, section=None):
        open(f"{self.root}/project/slides/{sid}.html", "w").write(html)
        self.order.append(sid)
        if section:
            self.sections[f"s{len(self.sections) + 1}"] = {"description": section, "start": sid}

    def write_index(self, title):
        idx = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-11-04T13:00:00Z"}, "lists": "css", "title": title,
               "order": self.order, "sections": self.sections, "cover": self.order[0],
               "faces": {"geist": {"family": "Geist", "href": "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap"},
                         "geist-mono": {"family": "Geist Mono", "href": "https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&display=swap"}},
               "designSystems": []}
        json.dump(idx, open(f"{self.root}/project/deck.json", "w"), ensure_ascii=False, indent=2)


def eyebrow(text, t=DARK, color=None):
    return (f'<p style="font-family:{MONO};font-size:24px;font-weight:500;letter-spacing:2px;'
            f'text-transform:uppercase;color:{color or t["acc"]}">{text}</p>')


def h2(text, t=DARK, size=72, width=None):
    w = f";width:{width}px" if width else ""
    return (f'<h2 style="font-size:{size}px;font-weight:600;line-height:1.05;letter-spacing:-2px;'
            f'color:{t["ink"]}{w}">{text}</h2>')


def head(eb, title, t=DARK):
    return f'<div style="display:flex;flex-direction:column;gap:16px">{eyebrow(eb, t)}{h2(title, t)}</div>'


def p(text, t=DARK, size=32, color="body", width=None, mono=False, weight=None, lh=1.4):
    w = f";width:{width}px" if width else ""
    f = f";font-family:{MONO}" if mono else ""
    wt = f";font-weight:{weight}" if weight else ""
    return f'<p style="font-size:{size}px;line-height:{lh};color:{t[color] if color in t else color}{w}{f}{wt}">{text}</p>'


def h3(text, t=DARK, size=32, color="ink"):
    return f'<h3 style="font-size:{size}px;font-weight:600;letter-spacing:-0.5px;line-height:1.2;color:{t[color] if color in t else color}">{text}</h3>'


def label(text, t=DARK, color="muted"):
    return (f'<p style="font-family:{MONO};font-size:24px;font-weight:500;letter-spacing:2px;'
            f'text-transform:uppercase;color:{t[color] if color in t else color}">{text}</p>')


def card(inner, t=DARK, accent=False, flex=1, pad=40, gap=16, width=None, bg=None, border=None):
    b = f"2px solid {t['acc']}" if accent else f"1px solid {border or t['line']}"
    size = f"width:{width}px" if width else f"flex:{flex}"
    return (f'<div style="{size};display:flex;flex-direction:column;gap:{gap}px;background:{bg or t["card"]};'
            f'border:{b};border-radius:16px;padding:{pad}px">{inner}</div>')


def row(*items, gap=40, align=None):
    a = f";align-items:{align}" if align else ""
    return f'<div style="display:flex;gap:{gap}px{a}">{"".join(items)}</div>'


def col(*items, gap=16, flex=None, width=None):
    s = f"flex:{flex};" if flex else ""
    s += f"width:{width}px;" if width else ""
    return f'<div style="{s}display:flex;flex-direction:column;gap:{gap}px">{"".join(items)}</div>'


def big(num, lab, t=DARK, color="acc", size=96, labsize=24, mono=True):
    f = f"font-family:{MONO};" if mono else ""
    return (f'<div style="display:flex;flex-direction:column;gap:8px">'
            f'<p style="{f}font-size:{size}px;font-weight:600;line-height:1;letter-spacing:-2px;color:{t[color] if color in t else color}">{num}</p>'
            f'<p style="font-size:{labsize}px;line-height:1.35;color:{t["body"]}">{lab}</p></div>')


def table(header, rows, t=DARK, widths=None, right=(), size=32, bold_rows=(), hi_rows=(), mono_cols=()):
    out = [f'<table style="font-size:{size}px;color:{t["ink"]};font-family:{SANS}">']
    cells = []
    for i, h in enumerate(header):
        st = []
        if widths:
            st.append(f"width:{widths[i]}%")
        if i in right:
            st.append("text-align:right")
        cells.append(f'<th style="{";".join(st)}">{h}</th>' if st else f"<th>{h}</th>")
    out.append("<tr>" + "".join(cells) + "</tr>")
    for r, rw in enumerate(rows):
        tr_style = f' style="background:{t["fill"]}"' if r in hi_rows else ""
        cs = []
        for i, c in enumerate(rw):
            st = []
            if i in right:
                st.append("text-align:right")
            if i in mono_cols:
                st.append(f"font-family:{MONO}")
            txt = f"<b>{c}</b>" if r in bold_rows else c
            cs.append(f'<td style="{";".join(st)}">{txt}</td>' if st else f"<td>{txt}</td>")
        out.append(f"<tr{tr_style}>" + "".join(cs) + "</tr>")
    out.append("</table>")
    return "".join(out)


def chip(text, t=DARK, color="acc", fg=None, size=24, pad="10px 16px", mono=True):
    c = t[color] if color in t else color
    f = f"font-family:{MONO};" if mono else ""
    return (f'<div style="background:{c};border-radius:10px;padding:{pad};display:flex;align-items:center;justify-content:center">'
            f'<p style="{f}font-size:{size}px;font-weight:600;color:{fg or t["onfill"]}">{text}</p></div>')
