"""streamlab.py — спільний код для ноутбуків заняття 10.

Усе працює на одному комп’ютері: відправник (appsrc → x264 → RTP або MPEG-TS/SRT), емулятор лінку
(UDP-relay на Python: ємність, затримка, джиттер, втрати), приймач (udpsrc / srtsrc → декодер → appsink).
Кожен кадр несе штрихкод зі своїм номером, тому вік кадру = час прийому − час відправлення того самого
кадру за одним годинником.

GStreamer запускається в окремому процесі (gstworker.py) — ноутбук сам gi не імпортує. Тому ноутбук може
працювати в ядрі cv-advanced, а gstworker — у Python із GStreamer: STREAMLAB_PYTHON, поточний інтерпретатор,
conda-середовище cv-streaming поруч із поточним або системний python3 — перший, де імпортується gi. Див. README.
Стиль графіків — у panels.py.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional

from streamcore import Scene, SkyScene, Impair, CVKalman, pct, BITS, detect_small, match  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; import numpy, scipy"


def _candidates() -> list:
    out = [os.environ.get('STREAMLAB_PYTHON'), sys.executable]
    prefix = os.path.realpath(sys.prefix)                    # .../anaconda3 або .../anaconda3/envs/<назва>
    root = os.path.dirname(os.path.dirname(prefix)) if os.path.basename(os.path.dirname(prefix)) == 'envs' else prefix
    for exe in ('bin/python', 'python.exe'):
        out.append(os.path.join(root, 'envs', 'cv-streaming', exe))
    out.append('/usr/bin/python3')
    return [p for i, p in enumerate(out) if p and os.path.exists(p) and p not in out[:i]]


_PYTHON = None


def worker_python() -> str:
    """Перший інтерпретатор, у якому імпортуються gi (GStreamer), numpy і scipy."""
    global _PYTHON
    if _PYTHON:
        return _PYTHON
    tried = _candidates()
    for exe in tried:
        if subprocess.run([exe, '-c', _PROBE], capture_output=True, timeout=60).returncode == 0:
            _PYTHON = exe
            return exe
    raise RuntimeError('Не знайдено Python із GStreamer (PyGObject) для gstworker.py. Перевірено:\n  ' + '\n  '.join(tried) +
                       '\nСтворіть середовище cv-streaming за README заняття 10 або вкажіть шлях у змінній STREAMLAB_PYTHON.')


def _call(job: dict, timeout: float = 300) -> dict:
    p = subprocess.run([worker_python(), os.path.join(HERE, 'gstworker.py')], input=pickle.dumps(job),
                       capture_output=True, timeout=timeout, cwd=HERE)
    if p.returncode != 0:
        raise RuntimeError(f'gstworker завершився з кодом {p.returncode}:\n{p.stderr.decode(errors="replace")[-2000:]}')
    return pickle.loads(p.stdout)


_INFO = None


def gst_version() -> str:
    """Версія GStreamer у процесі gstworker."""
    global _INFO
    _INFO = _INFO or _call(dict(cmd='info'))
    return _INFO['version']


def appsink_has_leaky_type() -> bool:
    """leaky-type в appsink з’явився в GStreamer 1.28; до того — max-buffers + drop."""
    gst_version()
    return _INFO['leaky_type']


def has_srt() -> bool:
    """Чи є srtsrc / srtsink (gst-plugins-bad, зібраний з libsrt)."""
    gst_version()
    return _INFO['srt']


def run_stream(scene: Scene, seconds: float, enc: str, transport: str = 'rtp', impair: Optional[Impair] = None,
               reverse: Optional[Impair] = None, jb_ms: int = 20, srt_latency_ms: int = 120, max_buffers: int = 1,
               drop: bool = True, work_s: float = 0.0, with_psnr: bool = False, pli_rtt: Optional[float] = None,
               seed: int = 1, settle_s: float = 0.6, keep_every: int = 0):
    """Повний прогін: відправник → (емулятор) → приймач в окремому процесі.

    pli_rtt — якщо задано, після серії втрат приймач просить IDR (PLI) із цим RTT, с.
    Повертає (records, tx, link): tx.t_send / tx.sizes, link.stats.
    """
    job = dict(scene=scene.params, seconds=seconds, enc=enc, transport=transport, impair=impair, reverse=reverse,
               jb_ms=jb_ms, srt_latency_ms=srt_latency_ms, max_buffers=max_buffers, drop=drop, work_s=work_s,
               with_psnr=with_psnr, pli_rtt=pli_rtt, seed=seed, settle_s=settle_s, keep_every=keep_every)
    out = _call(job, timeout=seconds * 4 + 120)
    tx = SimpleNamespace(t_send=out['t_send'], sizes=out['sizes'], pli_sent=out['pli_sent'])
    link = SimpleNamespace(stats=out['stats']) if out['stats'] else None
    return out['rec'], tx, link
