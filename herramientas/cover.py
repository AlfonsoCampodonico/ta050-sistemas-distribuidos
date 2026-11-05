"""Cover slide for the TA050 16–19 series in the new style."""
SERIES = [(16, "Fundamentos del entrenamiento"), (17, "Entrenamiento a escala de frontera"),
          (18, "Cómo se sirve un modelo"), (19, "Del motor al clúster")]
SERIES2 = [(20, "Muchos modelos, pocas GPUs"), (21, "Repartir un clúster de GPUs"),
           (22, "Entrenar con lo que se sirve")]
MONO = "'Geist Mono', 'Courier New', monospace"


def cover(n, title, subtitle, paper, notes, series=None):
    chips = []
    for k, name in series or (SERIES if n <= 19 else SERIES2):
        on = k == n
        chips.append(
            f'<div style="flex:1;display:flex;flex-direction:column;gap:6px;padding:20px 24px;border-radius:12px;'
            f'border:1px solid {"#34D399" if on else "#2A3138"};background:{"#132A22" if on else "#0E1114"}">'
            f'<p style="font-family:{MONO};font-size:24px;font-weight:500;color:{"#34D399" if on else "#7D8892"}">{k}</p>'
            f'<p style="font-size:24px;line-height:1.3;color:{"#F2F3EF" if on else "#A8B3BC"}">{name}</p></div>')
    return f'''<section id="cover" style="background:#0E1114;color:#F2F3EF;font-family:Geist, Arial, sans-serif;padding:128px;display:flex;flex-direction:column;justify-content:space-between">
<div style="display:flex;justify-content:space-between;align-items:center">
<p style="font-family:{MONO};font-size:24px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#34D399">TA050 · Sistemas Distribuidos I</p>
<p style="font-family:{MONO};font-size:24px;font-weight:500;letter-spacing:2px;text-transform:uppercase;color:#7D8892">Clase {n}</p>
</div>
<div style="display:flex;flex-direction:column;gap:40px">
<h1 style="font-size:120px;font-weight:600;line-height:1.05;letter-spacing:-4px;color:#F2F3EF">{title}</h1>
<p style="width:1400px;font-size:44px;line-height:1.3;color:#A8B3BC">{subtitle}</p>
</div>
<div style="display:flex;flex-direction:column;gap:24px">
<p style="font-size:24px;line-height:1.4;color:#A8B3BC">{paper}</p>
<div style="display:flex;gap:16px">{"".join(chips)}</div>
</div>
<aside>{notes}</aside>
</section>
'''
