"""Render each slide of the deck at 1920x1080 and report layout problems.

Approximates the Slides runtime: the <section> is the 1920x1080 canvas
(border-box, so its padding is the margins); everything else is plain CSS.
x-shape arrows are drawn as blocks; x-connector lines are not drawn.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

deck = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
order = json.loads((deck / "project/deck.json").read_text())["order"]

HEAD = """<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;padding:0;background:#888}
section{box-sizing:border-box;width:1920px;height:1080px;position:relative;overflow:hidden}
h1,h2,h3,p,ul,ol,table{margin:0}
ul,ol{padding-left:1.2em}
x-shape{display:block;flex:none}
x-connector{display:none}
aside{display:none}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid rgba(128,128,128,.35);padding:0.35em 0.6em;text-align:left}
</style></head><body>"""

CHECK = """() => {
  const sec = document.querySelector('section');
  const problems = [];
  const lim = {l: 128, r: 1792, t: 128, b: 952};
  for (const el of sec.querySelectorAll('*')) {
    if (el.tagName === 'ASIDE' || el.closest('aside')) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const pinned = getComputedStyle(el).position === 'absolute';
    if (r.right > 1920.5 || r.bottom > 1080.5 || r.left < -0.5)
      problems.push(`OFF-CANVAS <${el.tagName.toLowerCase()}> ${Math.round(r.left)},${Math.round(r.top)} → ${Math.round(r.right)},${Math.round(r.bottom)}`);
    else if (!pinned && (r.right > lim.r + 1 || r.bottom > lim.b + 1))
      problems.push(`PAST MARGIN <${el.tagName.toLowerCase()}> right=${Math.round(r.right)} bottom=${Math.round(r.bottom)} "${(el.textContent||'').trim().slice(0,40)}"`);
    // text overflowing its own box (a word wider than the box)
    if (['P','H1','H2','H3','LI','TD','TH'].includes(el.tagName) && el.scrollWidth > el.clientWidth + 2)
      problems.push(`TEXT OVERFLOW <${el.tagName.toLowerCase()}> "${el.textContent.trim().slice(0,40)}"`);
  }
  return [...new Set(problems)].slice(0, 12);
}"""

report = {}
with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome")
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    for i, sid in enumerate(order, 1):
        html = (deck / f"project/slides/{sid}.html").read_text()
        page.set_content(HEAD + html + "</body></html>")
        page.wait_for_timeout(150)
        report[f"{i:02d}-{sid}"] = page.evaluate(CHECK)
        page.screenshot(path=str(out / f"{i:02d}-{sid}.png"))
    browser.close()

for k, v in report.items():
    print(k, "OK" if not v else "")
    for p in v:
        print("   ", p)
