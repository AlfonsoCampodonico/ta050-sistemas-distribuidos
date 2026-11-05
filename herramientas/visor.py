"""Arma slides/clase-NN/ (deck.json, slides/*.html e index.html) a partir de un proyecto de Claude Slides.

Uso: python3 herramientas/visor.py herramientas/out/clase-20/project slides/clase-20
"""
import json, os, shutil, sys

FONTS = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in [
    {"href": "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap"},
    {"href": "https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&display=swap"}])
STYLE = """<style>
body{margin:0;background:#0A0C0F;color:#F2F3EF;font-family:Geist,Arial,sans-serif}
header{padding:24px 32px;font-family:'Geist Mono',monospace;color:#7D8892}
.frame{width:min(1280px,96vw);aspect-ratio:16/9;margin:0 auto 24px;position:relative;overflow:hidden;border:1px solid #2A3138;border-radius:8px}
.stage{width:1920px;height:1080px;transform-origin:0 0;position:absolute}
.stage section{width:1920px;height:1080px;box-sizing:border-box;position:relative}
.stage *{margin:0;box-sizing:border-box}.stage aside{display:none}
.stage x-shape{display:block;clip-path:polygon(0 30%,70% 30%,70% 0,100% 50%,70% 100%,70% 70%,0 70%)}
table{border-collapse:collapse;width:100%} th,td{padding:12px 16px;border-bottom:1px solid #2A3138;text-align:left}
body.notes .frame::after{content:none}
.note{width:min(1280px,96vw);margin:-12px auto 32px;white-space:pre-wrap;color:#A8B3BC;font-size:15px;line-height:1.5;display:none}
body.notes .note{display:block}
</style>"""
TAIL = """<script>
function fit(){document.querySelectorAll('.frame').forEach(f=>{f.firstElementChild.style.transform='scale('+f.clientWidth/1920+')'})}
document.querySelectorAll('.frame').forEach(f=>{const a=f.querySelector('aside');if(a){const n=document.createElement('div');n.className='note';n.textContent=a.textContent;f.after(n)}});
addEventListener('resize',fit);fit();
</script>
<script src="../present.js"></script></body></html>
"""


def build(src, dst):
    deck = json.load(open(f"{src}/deck.json"))
    title, order = deck["title"], deck["order"]
    os.makedirs(f"{dst}/slides", exist_ok=True)
    shutil.copy(f"{src}/deck.json", f"{dst}/deck.json")
    frames = []
    for sid in order:
        shutil.copy(f"{src}/slides/{sid}.html", f"{dst}/slides/{sid}.html")
        frames.append('<div class="frame"><div class="stage">' + open(f"{src}/slides/{sid}.html").read() + "</div></div>")
    header = (f'<header>{title} · {len(order)} slides · <label><input type="checkbox" '
              "onchange=\"document.body.classList.toggle('notes',this.checked)\"> notas</label></header>\n")
    html = ('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>{title}</title>{FONTS}\n{STYLE}</head><body>\n{header}" + "".join(frames) + "\n" + TAIL)
    open(f"{dst}/index.html", "w").write(html)
    print(f"{dst}: {len(order)} slides")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
