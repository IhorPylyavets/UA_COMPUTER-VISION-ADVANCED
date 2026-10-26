"""panels.py — спільні візуалізації для ноутбуків заняття 9.

Палітра і стиль збігаються з колодою лекції та з panels.py заняття 5
(темний фон, teal / blue / orange). Залежності: numpy, matplotlib.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


def bars_with_err(ax, labels: Sequence[str], means: Sequence[float], stds: Sequence[float],
                  colors: Sequence[str], title: str = "", fmt: str = "{:.2f}") -> None:
    """Стовпці із розкидом і підписами значень."""
    ax.bar(list(labels), means, yerr=stds, color=colors, capsize=4)
    ax.set_title(title)
    for i, v in enumerate(means):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom")


def id_timeline(ax, lanes: Dict[int, List[Tuple[int, int]]], title: str = "") -> None:
    """Часова шкала ідентичностей: рядок — GT-об'єкт, колір — ID трекера."""
    cmap = plt.get_cmap("tab20")
    for g, pts in lanes.items():
        for t, tid in pts:
            ax.plot([t, t + 1], [g, g], color=cmap(tid % 20), lw=6, solid_capstyle="butt")
    ax.set_title(title)
    ax.set_xlabel("кадр")
    ax.grid(False)
