"""Fusion у мережі для заняття 12: RGB-only, T-only, early 4-ch, early + modality dropout, early + аугментація зсувом T,
mid two-stream, late (WBF точок). Детектор — CenterNet-подібна heatmap-мережа: ціль — центри людей.

Основні дані — реальні (rgbt12.py, tools/fetch_rgbt.py): LLVIP і FLIR-aligned, умови — ніч / день кожного датасету; кадри зменшено
до 160×128, щоб навчання вкладалося в хвилини на CPU. Розмітка — бокси person; детекція — центр у радіусі
max(2 px, 0.2·h) від центру боксу; люди нижчі за 6 px на сітці 160×128 — зона ігнору (ні TP, ні FP).

Запасний варіант без даних — процедурна синтетика 64×64 (make_dataset): умови day / night / crossover / smoke.
"""
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S = 64                      # розмір кадру синтетики
STRIDE = 2                  # heatmap = кадр / 2
CONDS = ('day', 'night', 'crossover', 'smoke')       # умови синтетики
REAL_SIZE = (160, 128)      # (W, H) реальних кадрів для мережі
COND_LABEL = {'llvip_night': 'LLVIP ніч', 'llvip_day': 'LLVIP день', 'flir_day': 'FLIR день', 'flir_night': 'FLIR ніч'}
NATIVE_PX = {'llvip': 8, 'flir': 4}   # скільки px нативного кадру в одному px сітки 160×128


def label(c):
    return COND_LABEL.get(c, c)
CACHE = Path(__file__).resolve().parent / '_cache12'
torch.set_num_threads(max(1, torch.get_num_threads()))

# =====================================================================
# Синтетика (запасний варіант): умови з відомою відповіддю
# =====================================================================
COND = {
    'day':       dict(rgb_gain=1.00, rgb_noise=0.03, haze=0.00, dT=0.30),
    'night':     dict(rgb_gain=0.07, rgb_noise=0.05, haze=0.00, dT=0.38),
    'crossover': dict(rgb_gain=0.55, rgb_noise=0.04, haze=0.00, dT=0.015),
    'smoke':     dict(rgb_gain=0.90, rgb_noise=0.04, haze=0.80, dT=0.32),
}


def _smooth_noise(rng, n=S, scale=8):
    g = rng.normal(0, 1, (n // scale + 2, n // scale + 2))
    from scipy.ndimage import zoom
    z = zoom(g, scale, order=3)[:n, :n]
    return (z - z.mean()) / (z.std() + 1e-6)


def _ellipse(cx, cy, w, h):
    yy, xx = np.mgrid[0:S, 0:S]
    return (((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2) <= 1.0


def _rect(cx, cy, w, h):
    yy, xx = np.mgrid[0:S, 0:S]
    return (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)


def make_sample(rng, cond):
    p = COND[cond]
    bg_col = rng.choice([[0.22, 0.34, 0.16], [0.40, 0.38, 0.34], [0.30, 0.28, 0.22]])
    tex = _smooth_noise(rng)
    rgb = np.clip(np.array(bg_col)[:, None, None] * (1 + 0.25 * tex[None]), 0, 1)
    T = 0.40 + 0.06 * _smooth_noise(rng, scale=16)
    centers = []
    for _ in range(rng.integers(0, 4)):                              # люди
        h = rng.uniform(9, 18); w = h * rng.uniform(0.35, 0.5)
        cx, cy = rng.uniform(5, S - 5), rng.uniform(h / 2 + 1, S - h / 2 - 1)
        m = _ellipse(cx, cy, w, h)
        col = np.array(bg_col) * rng.uniform(0.85, 1.15, 3) if rng.random() < 0.3 else rng.uniform(0.05, 0.95, 3)
        rgb[:, m] = col[:, None] * (1 + 0.1 * tex[m][None])
        T[m] = T[m] + p['dT'] * rng.uniform(0.8, 1.2)
        centers.append((cx, cy))
    for _ in range(rng.integers(0, 3)):                              # RGB-«псевдо-люди» без тепла
        h = rng.uniform(9, 18); w = h * rng.uniform(0.35, 0.5)
        cx, cy = rng.uniform(5, S - 5), rng.uniform(h / 2 + 1, S - h / 2 - 1)
        rgb[:, _ellipse(cx, cy, w, h)] = rng.uniform(0.05, 0.95, 3)[:, None]
    for _ in range(rng.integers(0, 3)):                              # T-«псевдо-люди»: гарячі труби, двигуни
        h = rng.uniform(9, 18); w = h * rng.uniform(0.3, 0.55)
        cx, cy = rng.uniform(5, S - 5), rng.uniform(h / 2 + 1, S - h / 2 - 1)
        m = _rect(cx, cy, w, h) if rng.random() < 0.5 else _ellipse(cx, cy, w, h)
        g = rng.uniform(0.45, 0.65)
        rgb[:, m] = np.array([g, g, g * 1.02])[:, None]
        T[m] = T[m] + rng.uniform(0.25, 0.45)
    rgb = rgb * p['rgb_gain']
    if p['haze'] > 0:
        a = p['haze'] * rng.uniform(0.85, 1.0)
        rgb = (1 - a) * rgb + a * 0.6
    rgb = rgb + rng.normal(0, p['rgb_noise'], rgb.shape)
    T = T + rng.normal(0, 0.02, T.shape) + rng.normal(0, 0.01, (1, S))
    x = np.concatenate([np.clip(rgb, 0, 1), np.clip(T, 0, 1)[None]], 0).astype(np.float32)
    return x, np.array(centers, np.float32).reshape(-1, 2)


def make_dataset(n, seed, conds=CONDS):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4, S, S), np.float32); C = []; cond = []
    for i in range(n):
        c = conds[i % len(conds)]
        X[i], ctr = make_sample(rng, c)
        C.append(ctr); cond.append(c)
    return X, C, np.array(cond)


def heatmaps(C, shape=(S, S), sigma=1.2):
    """C: список масивів [n, 2] (cx, cy) або [n, 3] (cx, cy, h) у px кадру. Для h — sigma росте з розміром людини."""
    n, h, w = len(C), shape[0] // STRIDE, shape[1] // STRIDE
    Hm = np.zeros((n, 1, h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for i, ctr in enumerate(C):
        for row in ctr:
            cx, cy = row[0] / STRIDE, row[1] / STRIDE
            sg = max(sigma, row[2] / STRIDE / 6) if len(row) > 2 else sigma
            g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg ** 2)))
            Hm[i, 0] = np.maximum(Hm[i, 0], g)
            Hm[i, 0, int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] = 1.0       # пік = 1
    return Hm


def channel_stats(X):
    """Статистики нормалізації — з train, окремо для кожного каналу (не ImageNet)."""
    X = X.astype(np.float32) / (255.0 if X.dtype == np.uint8 else 1.0)
    return X.mean((0, 2, 3)), X.std((0, 2, 3))


# =====================================================================
# Моделі
# =====================================================================
def cbr(ci, co, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """stem (stride 1, «перший conv») → s2 → s4 [→ s8, якщо w має 4 ширини]. Повертає ознаки stride 2, 4 [, 8]."""
    def __init__(self, c_in, w=(8, 24, 32)):
        super().__init__()
        self.stem = cbr(c_in, w[0])
        self.s2 = cbr(w[0], w[1], 2)
        self.s4 = nn.Sequential(cbr(w[1], w[2], 2), cbr(w[2], w[2]))
        if len(w) > 3:
            self.s8 = nn.Sequential(cbr(w[2], w[3], 2), cbr(w[3], w[3]))
        self.out_ch = list(w[1:])

    def forward(self, x):
        f = [self.s2(self.stem(x))]
        f.append(self.s4(f[-1]))
        if hasattr(self, 's8'):
            f.append(self.s8(f[-1]))
        return f


class Head(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(sum(chs), 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                  cbr(16, 16))
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.out.bias, -2.19)                       # prior 0.1, як у CenterNet

    def forward(self, *f):
        size = f[0].shape[-2:]
        up = [f[0]] + [F.interpolate(x, size=size, mode='bilinear', align_corners=False) for x in f[1:]]
        return self.out(self.fuse(torch.cat(up, 1)))


class SingleNet(nn.Module):
    """Один потік: channels = [0, 1, 2] (RGB), [3] (T) або [0, 1, 2, 3] (early fusion)."""
    def __init__(self, channels, w=(8, 24, 32)):
        super().__init__()
        self.channels = list(channels)
        self.enc = Encoder(len(self.channels), w)
        self.head = Head(self.enc.out_ch)

    def forward(self, x):
        return self.head(*self.enc(x[:, self.channels]))


class MidNet(nn.Module):
    """Mid fusion: два енкодери, злиття ознак concat + 1×1 на кожному рівні."""
    def __init__(self, w=(6, 16, 24), out=(24, 32, 48)):
        super().__init__()
        self.er, self.et = Encoder(3, w), Encoder(1, w)
        names = ('m2', 'm4', 'm8')
        for name, c, o in zip(names, self.er.out_ch, out):
            setattr(self, name, nn.Sequential(nn.Conv2d(2 * c, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)))
        self.levels = names[:len(self.er.out_ch)]
        self.head = Head(list(out[:len(self.levels)]))

    def forward(self, x):
        fr, ft = self.er(x[:, :3]), self.et(x[:, 3:4])
        return self.head(*[getattr(self, n)(torch.cat([a, b], 1)) for n, a, b in zip(self.levels, fr, ft)])


def widen_first_conv(conv: nn.Conv2d, n_new=1, init='mean', seed=0):
    """Хірургія першого conv: in_channels 3 → 3 + n_new зі збереженням RGB-фільтрів.
    init: 'zero' — мережа на старті рахує рівно те саме, що й RGB-модель;
          'mean' — нові канали = середнє RGB-фільтрів (як «сірий» вхід);
          'random' — kaiming, як для щойно створеного шару (у Ultralytics 8.4 — дефолт для 4-го каналу).
    seed — для відтворюваності 'random'."""
    w = conv.weight.data
    torch.manual_seed(seed)
    new = nn.Conv2d(w.shape[1] + n_new, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :w.shape[1]] = w
        if init == 'zero':
            new.weight[:, w.shape[1]:] = 0
        elif init == 'mean':
            new.weight[:, w.shape[1]:] = w.mean(1, keepdim=True)
        elif init != 'random':
            raise ValueError(init)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def conv_macs(c_in, c_out, k, H, W, stride):
    return (H // stride) * (W // stride) * c_out * c_in * k * k


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# =====================================================================
# Навчання
# =====================================================================
def focal_loss(logit, gt, alpha=2, beta=4):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    lp = -torch.log(p) * (1 - p) ** alpha * pos
    ln = -torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * neg
    npos = pos.sum().clamp(min=1)
    return (lp.sum() + ln.sum()) / npos


def modality_dropout(x, p_rgb=0.15, p_t=0.15):
    """x: [B, 4, H, W] уже нормалізований (0 = середнє train). Цілу модальність гасимо в 0, але не обидві разом."""
    r = torch.rand(x.shape[0], device=x.device)
    drop_rgb = r < p_rgb
    drop_t = (r >= p_rgb) & (r < p_rgb + p_t)
    x = x.clone()
    x[drop_rgb, :3] = 0
    x[drop_t, 3:] = 0
    return x


def shift_T(x, dx, dy=0):
    """Зсув T-каналу на (dx, dy) px — штучна похибка реєстрації (край заповнюємо нулем = середнє, «немає T»)."""
    if dx == 0 and dy == 0:
        return x
    x = x.clone(); t = torch.zeros_like(x[:, 3:])
    H, W = x.shape[-2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    t[..., yd, xd] = x[:, 3:, ys, xs]
    x[:, 3:] = t
    return x


def shift_aug(x, max_px=3):
    """Аугментація: кожен кадр батча — свій випадковий зсув T у ±max_px по обох осях (як modality dropout, але для реєстрації)."""
    out = x.clone()
    d = torch.randint(-max_px, max_px + 1, (x.shape[0], 2))
    for i, (dx, dy) in enumerate(d.tolist()):
        out[i:i + 1] = shift_T(x[i:i + 1], dx, dy)
    return out


class Data:
    """Синтетика: train / test з однаковою нормалізацією (статистики train)."""
    tag, conds, size = 'v4', CONDS, (S, S)

    def __init__(self, n_train=4000, n_test=1200, seed=0):
        Xtr, Ctr, _ = make_dataset(n_train, seed)
        Xte, Cte, cond = make_dataset(n_test, seed + 1000)
        self.mu, self.sd = channel_stats(Xtr)
        nrm = lambda X: (X - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        self.Xtr = torch.from_numpy(nrm(Xtr)); self.Htr = torch.from_numpy(heatmaps(Ctr))
        self.Xte = torch.from_numpy(nrm(Xte)); self.Cte = Cte; self.cond = cond
        self.Ite = [np.zeros((0, 2), np.float32)] * len(Cte)
        self.Hte = torch.from_numpy(heatmaps(Cte))
        self.raw_te = Xte


class RealData:
    """LLVIP + FLIR-aligned на сітці 160×128. Спліти — авторські (LLVIP — за сценами, FLIR — за поїздками).
    Xtr / Xte зберігаються у float16 (≈ 0,6 ГБ), у float32 переводимо батч під час навчання."""
    tag, size = 'real1', REAL_SIZE[::-1]
    conds = ('llvip_night', 'llvip_day', 'flir_night', 'flir_day')

    def __init__(self, verbose=True):
        import rgbt12 as R
        R.require()
        t0 = time.time()
        tr = R.llvip_pairs('train') + R.flir_pairs('train')
        te = R.llvip_pairs('test') + R.flir_pairs('validation')
        Xtr, Ctr, Itr, self.cond_tr, self.luma_tr = R.load_arrays(tr, REAL_SIZE)
        Xte, self.Cte, self.Ite, self.cond, self.luma = R.load_arrays(te, REAL_SIZE)
        self.src_tr = np.array([p['src'] for p in tr]); self.src_te = np.array([p['src'] for p in te])
        self.pairs_te = te
        self.mu, self.sd = channel_stats(Xtr)
        nrm = lambda X: torch.from_numpy(((X.astype(np.float32) / 255 - self.mu[None, :, None, None])
                                          / self.sd[None, :, None, None]).astype(np.float16))
        self.Xtr, self.Xte = nrm(Xtr), nrm(Xte)
        self.Htr = torch.from_numpy(heatmaps(Ctr, self.size)); self.Hte = torch.from_numpy(heatmaps(self.Cte, self.size))
        self.raw_te = Xte
        if verbose:
            n = lambda c: sum(len(x) for x in c)
            print(f'train {len(tr)} пар ({(self.src_tr == "llvip").sum()} LLVIP, {(self.src_tr == "flir").sum()} FLIR), '
                  f'test {len(te)} пар; людей train {n(Ctr)}, test {n(self.Cte)} (+{n(self.Ite)} у зоні ігнору); '
                  f'{time.time() - t0:.0f} с')


def train(model, data, epochs=8, bs=64, lr=3e-3, dropout=False, shift=0, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(data.Xtr); steps = epochs * (n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    g = torch.Generator().manual_seed(seed)
    hist = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(n // bs):
            idx = perm[i * bs:(i + 1) * bs]
            x = data.Xtr[idx].float()
            if dropout:
                x = modality_dropout(x)
            if shift:
                x = shift_aug(x, shift)
            loss = focal_loss(model(x), data.Htr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            hist.append(loss.item())
    model.eval()
    return hist


def val_loss(model, data, n=600):
    with torch.no_grad():
        return focal_loss(model(data.Xte[:n].float()), data.Hte[:n]).item()


# =====================================================================
# Декодування, WBF, AP
# =====================================================================
@torch.no_grad()
def predict_heat(model, X, bs=200):
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs].float())))
    return torch.cat(out)


def decode(heat, thr=0.05, k=20):
    """heat: [N,1,h,w] → список масивів (x, y, score) у px кадру."""
    mx = F.max_pool2d(heat, 3, 1, 1)
    peaks = (heat == mx) & (heat > thr)
    dets = []
    for i in range(len(heat)):
        ys, xs = torch.nonzero(peaks[i, 0], as_tuple=True)
        s = heat[i, 0, ys, xs]
        o = torch.argsort(s, descending=True)[:k]
        dets.append(np.stack([(xs[o].numpy() + 0.5) * STRIDE - 0.5, (ys[o].numpy() + 0.5) * STRIDE - 0.5,
                              s[o].numpy()], 1) if len(o) else np.zeros((0, 3)))
    return dets


def wbf_points(det_lists, r=3.0, n_models=None):
    """Weighted Boxes Fusion для точок: кластер за відстанню ≤ r, координати — зважене середнє за score,
    score = середній score кластера · min(n, N) / N (детекція лише однієї з двох моделей — ×0.5)."""
    n_models = n_models or len(det_lists)
    allp = np.concatenate([d for d in det_lists if len(d)], 0) if any(len(d) for d in det_lists) else np.zeros((0, 3))
    allp = allp[np.argsort(-allp[:, 2])]
    clusters = []
    for p in allp:
        for c in clusters:
            cx = np.average(np.array(c)[:, :2], axis=0, weights=np.array(c)[:, 2])
            if np.hypot(*(p[:2] - cx)) <= r:
                c.append(p); break
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        c = np.array(c)
        xy = np.average(c[:, :2], axis=0, weights=c[:, 2])
        out.append([xy[0], xy[1], c[:, 2].sum() / n_models * min(len(c), n_models) / len(c)])
    return np.array(out).reshape(-1, 3)


def nms_points(det_lists, r=3.0):
    allp = np.concatenate([d for d in det_lists if len(d)], 0) if any(len(d) for d in det_lists) else np.zeros((0, 3))
    allp = allp[np.argsort(-allp[:, 2])]
    keep = []
    for p in allp:
        if all(np.hypot(*(p[:2] - q[:2])) > r for q in keep):
            keep.append(p)
    return np.array(keep).reshape(-1, 3)


def _radius(g, r):
    return np.maximum(2.0, 0.2 * g[:, 2]) if g.shape[1] > 2 else np.full(len(g), r)


def match_frames(dets, gts, ign=None, r=3.0):
    """Попереднє зіставлення: для кожного кадру — (scores, tp) з урахуванням зони ігнору. Радіус TP:
    для GT з висотою h — max(2, 0.2·h) px, інакше r. Детекція біля «ігнор»-людини, що не стала TP, відкидається."""
    out = []
    for i in range(len(gts)):
        d, g = dets[i], gts[i]
        ig = ign[i] if ign is not None else np.zeros((0, 2))
        used = np.zeros(len(g), bool); rad = _radius(g, r) if len(g) else np.zeros(0)
        sc, tp = [], []
        for x, y, s in (d[np.argsort(-d[:, 2])] if len(d) else []):
            hit = False
            if len(g):
                dist = np.hypot(g[:, 0] - x, g[:, 1] - y); dist[used] = np.inf
                j = np.argmin(dist)
                if dist[j] <= rad[j]:
                    used[j] = True; hit = True
            if not hit and len(ig) and np.min(np.hypot(ig[:, 0] - x, ig[:, 1] - y) - _radius(ig, r)) <= 0:
                continue
            sc.append(s); tp.append(hit)
        out.append((np.array(sc, float), np.array(tp, float), len(g)))
    return out


def ap_from_matches(m, idx):
    sc = np.concatenate([m[i][0] for i in idx]) if len(idx) else np.zeros(0)
    tp = np.concatenate([m[i][1] for i in idx]) if len(idx) else np.zeros(0)
    npos = sum(m[i][2] for i in idx)
    if npos == 0:
        return np.nan
    o = np.argsort(-sc, kind='stable'); tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / npos; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    k = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[k + 1] - mrec[k]) * mpre[k + 1]))


def average_precision(dets, gts, idx=None, r=3.0, ign=None):
    """AP (VOC, усі точки) для точкових детекцій."""
    idx = np.arange(len(gts)) if idx is None else np.asarray(idx)
    return ap_from_matches(match_frames(dets, gts, ign, r), idx)


def bootstrap_ap(dets, gts, idx, n_boot=300, seed=0, ign=None):
    rng = np.random.default_rng(seed)
    m = match_frames(dets, gts, ign)
    vals = [ap_from_matches(m, rng.choice(idx, len(idx))) for _ in range(n_boot)]
    return np.nanpercentile(vals, [2.5, 97.5])


def paired_bootstrap(dets_a, dets_b, gts, idx, n_boot=300, seed=0, ign=None):
    """Різниця AP(a) − AP(b) на ТИХ САМИХ перевибраних кадрах → 95 % CI різниці.
    Правильний тест «чи є приріст», на відміну від порівняння двох окремих CI на око."""
    rng = np.random.default_rng(seed)
    ma, mb = match_frames(dets_a, gts, ign), match_frames(dets_b, gts, ign)
    diff = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        diff.append(ap_from_matches(ma, s) - ap_from_matches(mb, s))
    return ap_from_matches(ma, idx) - ap_from_matches(mb, idx), np.nanpercentile(diff, [2.5, 97.5])


# =====================================================================
# Зоопарк моделей з кешем (demo 3 навчає, demo 4 бере з кешу)
# =====================================================================
W_REAL = (8, 24, 32, 48)          # реальні кадри більші — додаємо рівень stride 8
SPECS = {
    'RGB':             dict(make=lambda w: SingleNet([0, 1, 2], w)),
    'T':               dict(make=lambda w: SingleNet([3], w)),
    'early 4-ch':      dict(make=lambda w: SingleNet([0, 1, 2, 3], w)),
    'early + dropout': dict(make=lambda w: SingleNet([0, 1, 2, 3], w), dropout=True),
    'early + shift':   dict(make=lambda w: SingleNet([0, 1, 2, 3], w), shift=2),
    'mid two-stream':  dict(make=lambda w: MidNet((6, 16, 24, 32) if len(w) > 3 else (6, 16, 24))),
}
EPOCHS = {'v4': 8, 'real1': 12}


def make_model(name, data):
    return SPECS[name]['make'](W_REAL if data.tag != 'v4' else (8, 24, 32))


def get_model(name, data, epochs=None, verbose=True, retrain=False):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f'{data.tag}_{name.replace(" ", "_").replace("+", "p")}.pt'
    spec = SPECS[name]
    m = make_model(name, data)
    if f.exists() and not retrain:
        m.load_state_dict(torch.load(f, weights_only=True)); m.eval()
        if verbose:
            print(f'{name:16s} — з кешу {f.name}')
        return m
    t0 = time.time()
    train(m, data, epochs=epochs or EPOCHS[data.tag], dropout=spec.get('dropout', False), shift=spec.get('shift', 0))
    torch.save(m.state_dict(), f)
    if verbose:
        print(f'{name:16s} — навчено за {time.time() - t0:5.1f} с, параметрів {n_params(m):,}')
    return m


def evaluate(models, data, shift=(0, 0), drop=None, late=True):
    """AP за умовами. shift — (dx, dy) зсув T у px на тесті; drop — 'rgb' / 't' (модальність = 0 на тесті)."""
    dx, dy = (shift, 0) if np.isscalar(shift) else shift
    X = shift_T(data.Xte, dx, dy)
    if drop == 'rgb':
        X = X.clone(); X[:, :3] = 0
    elif drop == 't':
        X = X.clone(); X[:, 3:] = 0
    dets = {k: decode(predict_heat(m, X)) for k, m in models.items()}
    if late and 'RGB' in dets and 'T' in dets:
        dets['late WBF'] = [wbf_points([a, b]) for a, b in zip(dets['RGB'], dets['T'])]
    res = {}
    for k, d in dets.items():
        m = match_frames(d, data.Cte, data.Ite)
        row = {c: ap_from_matches(m, np.where(data.cond == c)[0]) for c in data.conds}
        row['all'] = ap_from_matches(m, np.arange(len(data.Cte)))
        res[k] = row
    return res, dets


def load_data(prefer_real=True):
    """Реальні дані, якщо завантажені; інакше — синтетика з поясненням."""
    import rgbt12 as R
    if prefer_real and R.available():
        return RealData()
    print(f'Реальних даних немає ({R.ROOT}) — запасний варіант: синтетика 64×64. Завантажити: {R.FETCH}')
    return Data()
