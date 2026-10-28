"""streamcore.py — частина streamlab без GStreamer і matplotlib: сцена зі штрихкодом, параметри лінку, Калман.

Імпортується і в ноутбуці, і в окремому процесі gstworker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ============================================================== сцена і штрихкод
BITS = 16


class Scene:
    """Детерміновані кадри: текстура з повільним панорамуванням, рухома ціль, штрихкод номера кадру."""

    def __init__(self, w: int = 640, h: int = 360, fps: int = 30, seed: int = 1, pan_px: float = 1.5):
        self.w, self.h, self.fps, self.pan = w, h, fps, pan_px
        self.params = dict(w=w, h=h, fps=fps, seed=seed, pan_px=pan_px)
        r = np.random.default_rng(seed)
        base = r.integers(0, 255, (h // 8 + 1, (w + 512) // 8 + 1, 3), dtype=np.uint8)
        base = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))          # блоки 8×8 — «листя»
        yy = np.linspace(0, 1, base.shape[0])[:, None, None]
        self.tex = (0.55 * base + 70 * yy).clip(0, 255).astype(np.uint8)
        self.code_h = max(16, h // 18)

    def target(self, idx: int) -> np.ndarray:
        """Центр цілі (x, y) у пікселях: проліт з маневром."""
        t = idx / self.fps
        x = 0.5 * self.w + 0.36 * self.w * np.sin(2 * np.pi * t / 7.0)
        y = 0.55 * self.h + 0.18 * self.h * np.sin(2 * np.pi * t / 2.3)
        return np.array([x, y])

    def frame(self, idx: int) -> np.ndarray:
        off = int(idx * self.pan) % 512
        img = self.tex[:self.h, off:off + self.w].copy()
        cx, cy = self.target(idx).astype(int)
        img[max(0, cy - 12):cy + 12, max(0, cx - 20):cx + 20] = (240, 70, 60)
        self._stamp(img, idx)
        return img

    @staticmethod
    def _chk(idx: int) -> int:
        return (idx ^ (idx >> 4) ^ (idx >> 8) ^ (idx >> 12) ^ 0x5) & 0xF

    def _bands(self):
        return (0, self.code_h), (self.h - self.code_h, self.h)

    def _stamp(self, img: np.ndarray, idx: int) -> None:
        word = (idx & 0xFFFF) | (self._chk(idx) << BITS)
        nb = BITS + 4
        bw = self.w // (nb + 2)
        for y0, y1 in self._bands():                    # дві копії: зверху і знизу
            img[y0:y1, :] = 0
            for b in range(nb):
                if (word >> b) & 1:
                    img[y0 + 2:y1 - 2, (b + 1) * bw + 2:(b + 2) * bw - 2] = 255

    def read(self, img: np.ndarray) -> int:
        """Номер кадру або −1, якщо обидві копії штрихкоду зіпсовані (контрольна сума не збігається)."""
        nb = BITS + 4
        bw = self.w // (nb + 2)
        for y0, y1 in self._bands():
            y = (y0 + y1) // 2
            word = 0
            for b in range(nb):
                if img[y, (b + 1) * bw + bw // 2].mean() > 127:
                    word |= 1 << b
            idx = word & 0xFFFF
            if (word >> BITS) == self._chk(idx):
                return idx
        return -1

    def corrupt(self, img: np.ndarray, idx: int, thr: float = 30.0, bs: int = 16) -> float:
        """Частка блоків 16×16 з грубою помилкою (середня |різниця| > thr) — артефакти втрат, а не стиснення."""
        c = self.code_h
        ref = self.frame(idx)[c:self.h - c].astype(np.int16)
        d = np.abs(img[c:self.h - c].astype(np.int16) - ref).mean(axis=2)
        H, W = (d.shape[0] // bs) * bs, (d.shape[1] // bs) * bs
        blk = d[:H, :W].reshape(H // bs, bs, W // bs, bs).mean(axis=(1, 3))
        return float((blk > thr).mean())

    def psnr(self, img: np.ndarray, idx: int) -> float:
        c = self.code_h
        ref = self.frame(idx)[c:self.h - c].astype(np.float32)
        d = img[c:self.h - c].astype(np.float32) - ref
        mse = float((d * d).mean())
        return 99.0 if mse < 1e-6 else 10 * np.log10(255 ** 2 / mse)


@dataclass
class Impair:
    rate_mbps: float = 0.0          # 0 — без обмеження
    delay_ms: float = 0.0
    jitter_ms: float = 0.0          # σ нормального розподілу, затримка ≥ 0
    loss: float = 0.0               # незалежні втрати
    ge: Optional[tuple] = None      # (p_good→bad, p_bad→good, loss_in_bad) — модель Гілберта–Еліотта
    keep_order: bool = False        # True — без перестановок (як радіо), False — як netem з джиттером
    queue_limit_ms: float = 1000.0  # drop-tail, щоб черга не росла нескінченно
    blackout: List[tuple] = field(default_factory=list)   # [(t0, t1)] у секундах від старту — серії втрат


def pct(a, q) -> float:
    a = np.asarray([x for x in a if np.isfinite(x)])
    return float(np.percentile(a, q)) if len(a) else float('nan')


class CVKalman:
    """Константна швидкість у 2D. dt_mode='pts' — крок у секундах з PTS; 'step' — крок = одне оновлення."""

    def __init__(self, q: float = 400.0, r: float = 4.0, dt_mode: str = 'pts'):
        self.q, self.r, self.mode = q, r, dt_mode
        self.x = None

    def _F(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        return F

    def _Q(self, dt):
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        return g @ g.T * self.q

    def predict_at(self, dt):
        return (self._F(dt) @ self.x)[:2]

    def update(self, z, t):
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0]); self.P = np.diag([self.r, self.r, 1e4, 1e4]); self.t = t
            return
        dt = (t - self.t) if self.mode == 'pts' else 1.0
        F, Q = self._F(dt), self._Q(dt)
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        H = np.eye(2, 4); S = H @ self.P @ H.T + np.eye(2) * self.r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x); self.P = (np.eye(4) - K @ H) @ self.P
        self.t = t


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Швидке box-розмиття (2r+1) по обох осях через кумулятивні суми; a — float32 2D або 3D."""
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode='edge'), axis=ax)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[ax]), axis=ax)
        lo = np.take(c, np.arange(0, c.shape[ax] - 2 * r - 1), axis=ax)
        a = (hi - lo) / (2 * r + 1)
    return a


class SkyScene(Scene):
    """Небо з хмарами і дрібними темними цілями (дрони вдалині) — для досліду «компресія проти точності».

    small_targets(idx) дає ground truth: (x, y, розмір) кожної цілі.
    """

    def __init__(self, w: int = 640, h: int = 360, fps: int = 30, seed: int = 1, pan_px: float = 0.6,
                 n_targets: int = 16, contrast: float = 25.0, noise: float = 4.0, sizes=(3, 4, 6, 8)):
        super().__init__(w, h, fps, seed, pan_px)
        self.params = dict(kind='sky', w=w, h=h, fps=fps, seed=seed, pan_px=pan_px, n_targets=n_targets, contrast=contrast,
                           noise=noise, sizes=tuple(sizes))
        self.noise = noise
        r = np.random.default_rng(seed + 100)
        yy = np.linspace(0, 1, h)[:, None, None]
        top, bot = np.array([118, 148, 186], np.float32), np.array([186, 196, 206], np.float32)
        grad = top + (bot - top) * yy
        coarse = r.normal(0, 1, (h // 24 + 2, (w + 512) // 24 + 2)).astype(np.float32)
        clouds = np.kron(coarse, np.ones((24, 24), np.float32))[:h, :w + 512]
        clouds = _box(_box(clouds, 12), 12)
        clouds = (clouds - clouds.min()) / (clouds.max() - clouds.min() + 1e-6)
        sky = grad[:, :1, :].repeat(w + 512, axis=1) + 38 * clouds[..., None]
        self.tex = sky.clip(0, 255).astype(np.uint8)
        c = self.code_h
        self.t_xy0 = np.stack([r.uniform(30, w - 30, n_targets), r.uniform(c + 25, h - c - 25, n_targets)], 1)
        self.t_v = r.uniform(-0.8, 0.8, (n_targets, 2))
        self.t_size = np.array(list(sizes) * (n_targets // len(sizes) + 1))[:n_targets]
        self.contrast = contrast

    def small_targets(self, idx: int) -> np.ndarray:
        c, w, h = self.code_h, self.w, self.h
        xy = self.t_xy0 + self.t_v * idx
        x = 30 + np.mod(xy[:, 0] - 30, w - 60)
        y = c + 25 + np.mod(xy[:, 1] - c - 25, h - 2 * c - 50)
        return np.stack([x, y, self.t_size], 1)

    def frame(self, idx: int) -> np.ndarray:
        off = int(idx * self.pan) % 512
        img = self.tex[:self.h, off:off + self.w].astype(np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        for x, y, s in self.small_targets(idx):
            r = s / 2.0
            x0, x1, y0, y1 = int(x - r - 2), int(x + r + 3), int(y - r - 2), int(y + r + 3)
            sub = (slice(max(0, y0), y1), slice(max(0, x0), x1))
            d = np.hypot((xx[sub] - x) / r, (yy[sub] - y) / (0.6 * r))      # еліпс 1 : 0.6, як силует крила
            alpha = np.clip(1.5 - d, 0, 1)[..., None]
            img[sub] -= self.contrast * alpha
        if self.noise:                                   # шум сенсора: свій для кожного кадру, детермінований за номером
            img += np.random.default_rng(10_000 + idx).normal(0, self.noise, img.shape[:2])[..., None]
        img = img.clip(0, 255).astype(np.uint8)
        self._stamp(img, idx)
        return img


def make_scene(params: dict) -> Scene:
    p = dict(params)
    return SkyScene(**{k: v for k, v in p.items() if k != 'kind'}) if p.get('kind') == 'sky' else Scene(**p)


def detect_small(img: np.ndarray, code_h: int, thr: float = 12.0, r_bg: int = 10):
    """Іграшковий детектор дрібних темних цілей: віднімання розмитого фону, поріг, зв’язні компоненти.

    Повертає масив центрів (x, y). Навмисно простий: суть досліду — як змінюється його якість зі стисненням.
    """
    from scipy import ndimage
    g = img.astype(np.float32).mean(axis=2)
    res = _box(g, r_bg) - g                       # цілі темніші за фон
    res[:code_h + 4] = 0
    res[-code_h - 4:] = 0
    lab, n = ndimage.label(res > thr)
    if n == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(res), lab, idx)
    cy, cx = np.array(ndimage.center_of_mass(res > thr, lab, idx)).T
    keep = (area >= 3) & (area <= 300)
    return np.stack([cx[keep], cy[keep]], 1)


def match(gt: np.ndarray, det: np.ndarray, extra: float = 3.0):
    """Жадібне зіставлення: TP, якщо центр детекції ближче за розмір/2 + extra. Повертає (tp, fp, fn, tp_by_size)."""
    used = np.zeros(len(det), bool)
    tp_sizes = []
    for x, y, s in gt:
        if len(det) == 0:
            continue
        d = np.hypot(det[:, 0] - x, det[:, 1] - y)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= s / 2 + extra:
            used[j] = True
            tp_sizes.append(s)
    tp = len(tp_sizes)
    return tp, int((~used).sum()), len(gt) - tp, tp_sizes
