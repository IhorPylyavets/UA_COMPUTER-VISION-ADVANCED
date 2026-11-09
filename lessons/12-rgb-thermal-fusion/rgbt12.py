"""Реальні RGB + LWIR дані для заняття 12: LLVIP і FLIR-aligned (tools/fetch_rgbt.py), опційно 16-біт TIFF.

  * LLVIP (Jia et al., ICCVW 2021) — 1280×1024, пари вирівняні апаратно, майже все — ніч; спліт авторів за сценами.
  * FLIR-aligned (Zhang et al., ICIP 2020) — 640×512, пари FLIR ADAS v1, вирівняні вручну; день і ніч;
    спліт авторів align_train / align_validation — різні поїздки.

Обидва датасети віддають T уже як 8-біт JPEG після AGC камери — це й є «домен», про який Demo 1.
Розмітка — бокси person у пікселях кадру; день / ніч: LLVIP — за сценою, FLIR — за яскравістю неба (поріг — з гістограми, Demo 3).
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2] / 'artifacts' / 'rgbt'
LLVIP = ROOT / 'llvip' / 'LLVIP'
FLIR = ROOT / 'flir_aligned' / 'align'
RAW16 = ROOT / 'raw16'
FETCH = 'python tools/fetch_rgbt.py'


def available(name='all'):
    ok = dict(llvip=(LLVIP / 'Annotations').is_dir(), flir=(FLIR / 'Annotations').is_dir())
    return all(ok.values()) if name == 'all' else ok[name]


def require(name='all'):
    if not available(name):
        raise FileNotFoundError(f'Немає реальних даних у {ROOT}. Завантажте з кореня репозиторію: {FETCH}')


# =====================================================================
# Списки пар і розмітка
# =====================================================================
def _boxes(xml_path, cls='person'):
    if not xml_path.exists():
        return np.zeros((0, 4), np.float32)
    out = []
    for o in ET.parse(xml_path).getroot().iter():
        if o.tag.lower() == 'object' and (o.findtext('name') or '').strip().lower() == cls:
            b = o.find('bndbox')
            out.append([float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
    return np.array(out, np.float32).reshape(-1, 4)


def llvip_pairs(split):
    """split: 'train' / 'test' → список dict(rgb, t, boxes, scene, src, native)."""
    out = []
    for t in sorted((LLVIP / 'infrared' / split).glob('*.jpg')):
        rgb = LLVIP / 'visible' / split / t.name
        if rgb.exists():
            out.append(dict(rgb=rgb, t=t, boxes=_boxes(LLVIP / 'Annotations' / (t.stem + '.xml')),
                            scene=t.stem[:2], src='llvip', native=(1280, 1024)))
    return out


def flir_pairs(split):
    """split: 'train' / 'validation' (= test)."""
    lst = FLIR / f'align_{split}.txt'
    out = []
    for stem in lst.read_text().split():
        fid = stem.replace('_PreviewData', '')
        t, rgb = FLIR / 'JPEGImages' / f'{fid}_PreviewData.jpeg', FLIR / 'JPEGImages' / f'{fid}_RGB.jpg'
        if t.exists() and rgb.exists():
            out.append(dict(rgb=rgb, t=t, boxes=_boxes(FLIR / 'Annotations' / f'{fid}_PreviewData.xml'),
                            scene=split, src='flir', native=(640, 512)))
    return sorted(out, key=lambda p: p['t'].name)


def read_pair(p, size=None):
    """→ rgb (H, W, 3) uint8 RGB, t (H, W) uint8. size=(W, H) — зменшити (JPEG декодується одразу в меншому масштабі)."""
    flag_c, flag_g = cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE
    if size is not None:
        k = p['native'][0] // size[0]
        red = {2: (cv2.IMREAD_REDUCED_COLOR_2, cv2.IMREAD_REDUCED_GRAYSCALE_2),
               4: (cv2.IMREAD_REDUCED_COLOR_4, cv2.IMREAD_REDUCED_GRAYSCALE_4),
               8: (cv2.IMREAD_REDUCED_COLOR_8, cv2.IMREAD_REDUCED_GRAYSCALE_8)}
        flag_c, flag_g = red.get(k, (flag_c, flag_g))
    rgb = cv2.cvtColor(cv2.imread(str(p['rgb']), flag_c), cv2.COLOR_BGR2RGB)
    t = cv2.imread(str(p['t']), flag_g)
    if size is not None:
        if rgb.shape[1::-1] != tuple(size):
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if t.shape[1::-1] != tuple(size):
            t = cv2.resize(t, size, interpolation=cv2.INTER_AREA)
    return rgb, t


def luminance(rgb):
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


def sky_luma(rgb):
    """Медіана яскравості верхньої третини RGB без чорних полів вирівнювання: небо вдень світле, уночі темне,
    а середнє всього кадру вночі «піднімають» ліхтарі й автоекспозиція."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[: rgb.shape[0] // 3]
    v = g[g > 3]
    return float(np.median(v)) if v.size else 0.0


DAY_SKY = 128.0     # FLIR: поріг день / ніч за яскравістю неба (0…255); провал гістограми, обґрунтування — Demo 3
LLVIP_DAY_SCENES = {'25', '26'}   # LLVIP: 24 нічні сцени і 2 денні (25 — train, 26 — test)


def condition(p, rgb):
    if p['src'] == 'llvip':
        return 'llvip_day' if p['scene'] in LLVIP_DAY_SCENES else 'llvip_night'
    return 'flir_day' if sky_luma(rgb) >= DAY_SKY else 'flir_night'


COND_NAMES = {'llvip_night': 'LLVIP ніч', 'llvip_day': 'LLVIP день', 'flir_day': 'FLIR день', 'flir_night': 'FLIR ніч'}


# =====================================================================
# Масиви для мережі
# =====================================================================
def load_arrays(pairs, size=(160, 128), min_h=6.0):
    """→ X uint8 [N, 4, H, W] (R, G, B, T), список GT [n, 3] (cx, cy, h) і ігнор [m, 3] у px сітки size,
    умова кожного кадру, яскравість неба. Люди нижчі за min_h px на цій сітці — «ігнор» (ні TP, ні FP)."""
    W, H = size
    X = np.zeros((len(pairs), 4, H, W), np.uint8)
    gts, ign, cond, luma = [], [], [], []
    for i, p in enumerate(pairs):
        rgb, t = read_pair(p, size)
        X[i, :3] = rgb.transpose(2, 0, 1); X[i, 3] = t
        s = W / p['native'][0]
        b = p['boxes'] * s
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 3] - b[:, 1]], 1)
        gts.append(c[c[:, 2] >= min_h]); ign.append(c[c[:, 2] < min_h])
        cond.append(condition(p, rgb)); luma.append(sky_luma(rgb))
    return X, gts, ign, np.array(cond), np.array(luma)


# =====================================================================
# Demo 1: AGC-домени і контраст людини на реальних кадрах
# =====================================================================
def person_contrast(pairs, size=(320, 256), ring=0.5, min_h=12):
    """Для кожної людини: медіана T (і сірого RGB) у центральній частині боксу мінус медіана кільця навколо, у рівнях 0…255.
    Реальний аналог crossover: що менший |ΔT|, то гірше людину видно в LWIR."""
    rows = []
    for p in pairs:
        rgb, t = read_pair(p, size)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = size[0] / p['native'][0]
        for x0, y0, x1, y1 in p['boxes'] * s:
            if y1 - y0 < min_h:
                continue
            w, h = x1 - x0, y1 - y0
            cx0, cx1 = int(x0 + 0.3 * w), int(np.ceil(x1 - 0.3 * w))
            cy0, cy1 = int(y0 + 0.15 * h), int(np.ceil(y1 - 0.15 * h))
            X0, X1 = int(max(0, x0 - ring * w)), int(min(size[0], x1 + ring * w))
            Y0, Y1 = int(max(0, y0 - ring * h * 0.5)), int(min(size[1], y1 + ring * h * 0.5))
            m = np.zeros(t.shape, bool); m[Y0:Y1, X0:X1] = True
            m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = False
            if m.sum() < 20 or cx1 <= cx0 or cy1 <= cy0:
                continue
            rows.append(dict(src=p['src'], cond=condition(p, rgb), dT=float(np.median(t[cy0:cy1, cx0:cx1]) - np.median(t[m])),
                             dRGB=float(np.median(g[cy0:cy1, cx0:cx1]) - np.median(g[m])),
                             h=float(h)))
    return rows


def raw16_frames():
    return sorted(RAW16.glob('*.tif*')) if RAW16.is_dir() else []


# =====================================================================
# Demo 2: залишкова розбіжність «вирівняних» пар за mutual information
# =====================================================================
def _mi(a, b, bins=32):
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 256], [0, 256]])
    p = h / h.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def grad_mag(img, sigma=1.5):
    """Модуль градієнта після згладжування: не залежить від полярності (людина світла в T і темна в RGB — однаково)."""
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def best_shift(gray, t, r=6, margin=16, method='grad', bins=32):
    """Зсув (dx, dy) T відносно RGB перебором у ±r px. method: 'mi' — MI інтенсивностей, 'ncc' / 'ncc_min' — максимум /
    мінімум NCC інтенсивностей, 'grad' — NCC модулів градієнта, 'grad_mi' — MI модулів градієнта.
    Повертає dx, dy, «різкість» піку (max − медіана) і чи впав пік на межу перебору (тоді він ненадійний)."""
    if method in ('grad', 'grad_mi'):
        gray, t = grad_mag(gray), grad_mag(t)
    if method == 'grad_mi':
        gray, t = (np.clip(x / (np.percentile(x, 99.5) + 1e-6) * 255, 0, 255) for x in (gray, t))
    f = {'mi': lambda a, b: _mi(a, b, bins), 'grad_mi': lambda a, b: _mi(a, b, bins), 'grad': _ncc,
         'ncc': _ncc, 'ncc_min': lambda a, b: -_ncc(a, b)}[method]
    H, W = t.shape
    core_g = gray[margin:H - margin, margin:W - margin]
    grid = np.zeros((2 * r + 1, 2 * r + 1))
    for i, dy in enumerate(range(-r, r + 1)):
        for j, dx in enumerate(range(-r, r + 1)):
            grid[i, j] = f(core_g, t[margin + dy:H - margin + dy, margin + dx:W - margin + dx])
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    return j - r, i - r, float(grid.max() - np.median(grid)), bool(abs(j - r) == r or abs(i - r) == r)


def misalignment(pairs, size, r=6, n=120, seed=0, method='grad'):
    """Розподіл оптимального зсуву T для n випадкових пар (на сітці size). Масив [n, 4]: dx, dy, різкість, пік на межі."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    out = []
    for k in idx:
        rgb, t = read_pair(pairs[k], size)
        out.append(best_shift(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), t, r=r, method=method))
    return np.array(out, float)


def shift_recovery(pairs, true=(3, -2), size=(320, 256), r=6, n=30, seed=0,
                   methods=('mi', 'ncc', 'ncc_min', 'grad', 'grad_mi')):
    """Частка реальних пар, на яких метод знаходить відомий штучний зсув T (з точністю 1 px сітки size)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), min(n, len(pairs)), replace=False)
    M = np.float32([[1, 0, true[0]], [0, 1, true[1]]])
    hits = {m: 0 for m in methods}
    for k in idx:
        rgb, t = read_pair(pairs[k], size)
        moved = cv2.warpAffine(t, M, size, borderMode=cv2.BORDER_REFLECT)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for m in methods:
            dx, dy, _, _ = best_shift(g, moved, r=r, method=m)
            hits[m] += max(abs(dx - true[0]), abs(dy - true[1])) <= 1
    return {m: h / len(idx) for m, h in hits.items()}
