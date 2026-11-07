"""Arma slides/clase-NN/ (deck.json, slides/*.html e index.html) a partir de un proyecto de Claude Slides.

Uso: python3 herramientas/visor.py herramientas/out/clase-20/project slides/clase-20
     python3 herramientas/visor.py slides/clase-16 slides/clase-16   (rearma un deck sin generador)
Aplica el estilo de la materia (estilo.py) antes de armar el visor.
"""
import json, os, re, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estilo import restyle

FACES = {"chakra-petch": {"family": "Chakra Petch", "href": "https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&display=swap"},
         "rubik": {"family": "Rubik", "href": "https://fonts.googleapis.com/css2?family=Rubik:ital,wght@0,300..900;1,300..900&display=swap"},
         "ibm-plex-mono": {"family": "IBM Plex Mono", "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap"}}
FONTS = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in FACES.values())
STYLE = """<style>
body{margin:0;background:linear-gradient(180deg,#010101,#03181A) fixed;color:#F3F4F5;font-family:Rubik,Arial,sans-serif;font-weight:300}
header{padding:24px 32px;font-family:'IBM Plex Mono',monospace;color:#ADB5BD}
header b{font-family:'Chakra Petch',sans-serif;font-weight:700;color:#FFFFFF;text-transform:uppercase;letter-spacing:.02em;text-shadow:0 0 .6em rgba(37,246,254,.5)}
header a{color:#25F6FE}
.frame{width:min(1280px,96vw);aspect-ratio:16/9;margin:0 auto 24px;position:relative;overflow:hidden;border:1px solid #0B2E33;border-radius:8px;box-shadow:0 0 24px rgba(37,246,254,.08)}
.stage{width:1920px;height:1080px;transform-origin:0 0;position:absolute}
.stage section{width:1920px;height:1080px;box-sizing:border-box;position:relative}
.stage *{margin:0;box-sizing:border-box}.stage aside{display:none}
.stage x-shape{display:block;clip-path:polygon(0 30%,70% 30%,70% 0,100% 50%,70% 100%,70% 70%,0 70%)}
.stage b,.stage strong{font-weight:500}
.stage ol li::marker{color:#046B7D;font-family:'IBM Plex Mono',monospace;font-weight:500}
.stage ul{list-style:none;padding-left:0}
.stage ul li{position:relative;padding-left:1.1em}
.stage ul li::before{content:"";position:absolute;left:0;top:.55em;width:.32em;height:.32em;background:#046B7D}
table{border-collapse:collapse;width:100%} th,td{padding:12px 16px;border-bottom:1px solid #DEE2E6;text-align:left}
th{font-weight:500;color:#046B7D;font-family:'IBM Plex Mono',monospace;font-size:.8em;text-transform:uppercase;letter-spacing:.04em}
body.notes .frame::after{content:none}
.note{width:min(1280px,96vw);margin:-12px auto 32px;white-space:pre-wrap;color:#ADB5BD;font-size:15px;line-height:1.5;display:none}
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
    num = int(re.search(r"(\d+)$", os.path.normpath(dst)).group(1))
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.isdir(f"{dst}/slides"):
            shutil.rmtree(f"{dst}/slides")
        shutil.copytree(f"{src}/slides", f"{dst}/slides")
        shutil.copy(f"{src}/deck.json", f"{dst}/deck.json")
    restyle(dst, num)
    deck = json.load(open(f"{dst}/deck.json"))
    deck["faces"] = FACES
    json.dump(deck, open(f"{dst}/deck.json", "w"), ensure_ascii=False, indent=2)
    src = dst
    title, order = deck["title"], deck["order"]
    frames = []
    for sid in order:
        frames.append('<div class="frame"><div class="stage">' + open(f"{src}/slides/{sid}.html").read() + "</div></div>")
    header = (f'<header><b>{title}</b> · {len(order)} slides · <label><input type="checkbox" '
              "onchange=\"document.body.classList.toggle('notes',this.checked)\"> notas</label></header>\n")
    html = ('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f"<title>{title}</title>{FONTS}\n{STYLE}</head><body>\n{header}" + "".join(frames) + "\n" + TAIL)
    open(f"{dst}/index.html", "w").write(html)
    print(f"{dst}: {len(order)} slides")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
