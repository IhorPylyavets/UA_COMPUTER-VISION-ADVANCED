"""panels.py — спільний стиль графіків і програвач кадрів для ноутбуків заняття 10.

Палітра і стиль збігаються з колодою лекції та з panels.py занять 5 і 9
(темний фон, teal / blue / orange). Залежності: matplotlib, Pillow, IPython.
"""
from __future__ import annotations

import base64
import io
import json
import uuid
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

PAL = dict(bg="#181A19", card="#1E2124", line="#3A3D42", txt="#E6E8EB", mu="#9AA3AD",
           te="#4FD1C5", bl="#38BDF8", or_="#F97316", re="#F87171", vi="#A78BFA", ok="#4ADE80")
CYCLE = [PAL["te"], PAL["bl"], PAL["or_"], PAL["vi"], PAL["re"], PAL["mu"], PAL["ok"]]


def set_style(dark: bool = True) -> None:
    """Темний стиль matplotlib у кольорах колоди."""
    if not dark:
        plt.rcdefaults()
        return
    plt.rcParams.update({
        "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["card"], "savefig.facecolor": PAL["bg"],
        "axes.edgecolor": PAL["line"], "axes.labelcolor": PAL["mu"], "text.color": PAL["txt"],
        "xtick.color": PAL["mu"], "ytick.color": PAL["mu"], "grid.color": "#2A2E33",
        "axes.grid": True, "grid.linewidth": 0.6, "axes.prop_cycle": plt.cycler(color=CYCLE),
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "axes.axisbelow": True,
    })


def frames_player(frames: Sequence[np.ndarray], captions: Optional[Sequence[str]] = None, fps: float = 15,
                  width: int = 480, quality: int = 75, chart: Optional[dict] = None):
    """Анімація кадрів у виводі ноутбука: JPEG у base64 і кнопка «пуск / пауза» з повзунком.

    chart — графік праворуч, що «малюється» разом із відео: dict(x=..., y=..., at=..., title=, xlabel=, ylabel=),
    де x, y — уся крива, а at[k] — значення x для k-го кадру анімації (точки з x ≤ at[k] уже видно).
    Без ffmpeg і без ipywidgets: працює в JupyterLab і VS Code, зберігається разом із виводом ноутбука.
    """
    from IPython.display import HTML
    from PIL import Image

    urls = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, np.uint8))
        im = im.resize((width, round(im.height * width / im.width)), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=quality)
        urls.append('data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode())
    caps = list(captions) if captions is not None else [''] * len(urls)
    ch = None
    if chart is not None:
        ch = dict(x=[float(v) for v in chart['x']], y=[float(v) for v in chart['y']], at=[float(v) for v in chart['at']],
                  title=chart.get('title', ''), xlabel=chart.get('xlabel', ''), ylabel=chart.get('ylabel', ''))
    uid = 'fp' + uuid.uuid4().hex[:8]
    btn = (f'font:600 13px monospace;color:{PAL["txt"]};background:{PAL["card"]};border:1px solid {PAL["line"]};'
           'border-radius:6px;padding:3px 10px;cursor:pointer')
    canvas = (f'<canvas class="ch" style="display:block;width:{width}px;max-width:100%;height:{round(width * 9 / 16) + 34}px;'
              f'background:{PAL["card"]};border-radius:4px"></canvas>') if ch else ''
    return HTML(f"""
<div id="{uid}" style="display:inline-block;background:{PAL['bg']};padding:8px;border-radius:8px;max-width:100%">
  <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <img class="im" style="display:block;width:{width}px;max-width:100%;border-radius:4px">
      <div style="display:flex;gap:8px;align-items:center;margin-top:6px">
        <button class="bt" style="{btn}">❚❚</button>
        <input class="sl" type="range" min="0" max="{len(urls) - 1}" value="0" style="flex:1;accent-color:{PAL['te']}">
      </div>
    </div>
    {canvas}
  </div>
  <div class="cp" style="font:12.5px monospace;color:{PAL['mu']};margin-top:4px;min-height:1.3em"></div>
</div>
<script>
(() => {{
  const root = document.getElementById("{uid}"), q = c => root.querySelector("." + c);
  const img = q("im"), btn = q("bt"), bar = q("sl"), cap = q("cp"), cv = q("ch");
  const U = {json.dumps(urls)}, C = {json.dumps(caps, ensure_ascii=False)}, D = {json.dumps(ch, ensure_ascii=False)};
  const P = {json.dumps(PAL)};
  let i = 0, timer = null;
  function draw() {{
    if (!cv || !D) return;
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = r.width, H = r.height, L = 46, R = 10, T = 26, B = 34, pw = W - L - R, ph = H - T - B;
    const x0 = Math.min(...D.x), x1 = Math.max(...D.x), y1 = Math.max(...D.y) * 1.1 || 1;
    const X = v => L + (v - x0) / ((x1 - x0) || 1) * pw, Y = v => T + ph - v / y1 * ph;
    g.clearRect(0, 0, W, H); g.font = "11px monospace"; g.textBaseline = "middle";
    const st = (() => {{ const raw = y1 / 4, m = Math.pow(10, Math.floor(Math.log10(raw)));
      return [1, 2, 2.5, 5, 10].map(k => k * m).find(s => s >= raw); }})();
    for (let v = 0; v <= y1; v += st) {{
      g.strokeStyle = "#2A2E33"; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + pw, Y(v)); g.stroke();
      g.fillStyle = P.mu; g.textAlign = "right"; g.fillText(+v.toFixed(1), L - 6, Y(v)); }}
    g.textAlign = "center"; g.textBaseline = "top";
    for (const v of [x0, (x0 + x1) / 2, x1]) g.fillText(Math.round(v), X(v), T + ph + 5);
    g.fillText(D.xlabel, L + pw / 2, T + ph + 19);
    g.save(); g.translate(11, T + ph / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText(D.ylabel, 0, 0); g.restore();
    g.fillStyle = P.txt; g.font = "600 12px sans-serif"; g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(D.title, L, 12);
    const now = D.at[i]; let last = -1;
    g.strokeStyle = P.te; g.lineWidth = 1.6; g.beginPath();
    for (let k = 0; k < D.x.length && D.x[k] <= now; k++) {{ k ? g.lineTo(X(D.x[k]), Y(D.y[k])) : g.moveTo(X(D.x[k]), Y(D.y[k])); last = k; }}
    g.stroke();
    g.fillStyle = P.te; for (let k = 0; k <= last; k++) {{ g.beginPath(); g.arc(X(D.x[k]), Y(D.y[k]), 1.8, 0, 7); g.fill(); }}
    if (last >= 0) {{ g.fillStyle = P.or_; g.beginPath(); g.arc(X(D.x[last]), Y(D.y[last]), 4.5, 0, 7); g.fill();
      g.strokeStyle = P.or_; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(D.x[last]), T); g.lineTo(X(D.x[last]), T + ph); g.stroke(); g.setLineDash([]); }}
  }}
  const show = k => {{ i = k; img.src = U[i]; bar.value = i; cap.textContent = C[i]; draw(); }};
  const play = () => {{ timer = setInterval(() => show((i + 1) % U.length), {1000 / fps:.1f}); btn.textContent = "❚❚"; }};
  const stop = () => {{ clearInterval(timer); timer = null; btn.textContent = "▶"; }};
  btn.onclick = () => (timer ? stop() : play());
  bar.oninput = () => {{ stop(); show(+bar.value); }};
  show(0); play();
}})();
</script>""")
