"""mtmc.py — спільний код ДЗ 9 (крос-камерний трекер на MDMT).

Дві «камери» — два дрони MDMT, що одночасно знімають ту саму сцену; об'єкт має той самий GT-ID в обох потоках.
Тут — усе, що не є суттю ДЗ: трекер на камері (той самий, що в демо 1), симульований детектор, треклети,
кропи, гомографія між дронами, метрики й малювання. Суть — часове вікно, геометрія, злиття доказів,
три зони порогів — студент пише в ноутбуці.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from mdmt import FPS, H, W, frame_list, frame_path, frames_dir, load_boxes, run_trackeval

REPO = Path(__file__).resolve().parents[2]

# ───────────────────────────── детектор ─────────────────────────────


def detect_mdmt(seq: str, drone: int, p_vis=0.95, p_occ=0.8, occ_score=(0.15, 0.45), shrink=0.7,
                fp_rate=3.0, jitter=1.5, classes=("car",), seed=0):
    """Симульований детектор на реальних траєкторіях MDMT (як §7 демо 1): видимий об'єкт — сильний score,
    перекритий або обрізаний краєм кадру (прапорець MDMT) — слабкий score і обрізаний бокс; плюс хибні спрацювання.
    Повертає (gt, dets): gt — для кожного кадру [(id, cx, cy, w, h)], dets — масиви (cx, cy, w, h, score)."""
    rng = np.random.default_rng(seed)
    n = len(frame_list(seq, drone))
    gt: List[list] = [[] for _ in range(n)]
    out: List[list] = [[] for _ in range(n)]
    for t, tid, x, y, w, h, _, occ in load_boxes(seq, drone, classes):
        if t >= n:
            continue
        cx, cy = x + w / 2, y + h / 2
        gt[t].append((tid, cx, cy, w, h))
        if rng.random() > (p_occ if occ else p_vis):
            continue
        s = rng.uniform(*occ_score) if occ else rng.uniform(0.6, 0.95)
        bw, bh = w, h
        if occ:
            bw, bh = (w * shrink, h) if w >= h else (w, h * shrink)
        out[t].append([cx + rng.normal(0, jitter), cy + rng.normal(0, jitter),
                       bw * (1 + rng.normal(0, 0.03)), bh * (1 + rng.normal(0, 0.03)), s])
    for t in range(n):
        for _ in range(rng.poisson(fp_rate)):
            out[t].append([rng.uniform(0, W), rng.uniform(0, H), rng.uniform(25, 70),
                           rng.uniform(25, 70), rng.uniform(0.1, 0.5)])
    return gt, [np.array(d, float).reshape(-1, 5) for d in out]


# ───────────────────────────── трекер на камері ─────────────────────────────


def iou_matrix(A, B):
    """IoU між боксами у форматі (cx, cy, w, h)."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A, B = np.asarray(A, float), np.asarray(B, float)
    ax1, ay1, ax2, ay2 = A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2, A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2
    bx1, by1, bx2, by2 = B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2, B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    ua = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None] - inter
    return inter / np.maximum(ua, 1e-9)


class Track:
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    Hm = np.eye(4, 8)
    Q = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.05, 0.05])
    R = np.diag([4, 4, 9, 9])

    def __init__(self, tid, z):
        self.id = tid
        self.x = np.r_[z[:4], 0, 0, 0, 0].astype(float)
        self.P = np.diag([10, 10, 10, 10, 100, 100, 10, 10]).astype(float)
        self.hits, self.since, self.confirmed = 1, 0, False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.since += 1

    def update(self, z, n_init):
        y = z[:4] - self.Hm @ self.x
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.Hm) @ self.P
        self.hits += 1
        self.since = 0
        self.confirmed |= self.hits >= n_init

    def box(self):
        return self.x[:4].copy()


def _associate(tracks, dets, max_cost):
    if len(tracks) == 0 or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))
    cost = 1 - iou_matrix([t.box() for t in tracks], dets[:, :4])
    r, c = linear_sum_assignment(cost)
    m = [(i, j) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    mt, md = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def run_tracker(dets, n_init=3, max_age=30, high=0.6, low=0.1, new_thr=0.7, match1=0.8, match2=0.5):
    """ByteTrack з оновленням лише центру слабким боксом (BYTE (центр) з демо 1).
    n_init і max_age — у кадрах. Повертає [(t, id, cx, cy, w, h)] лише для підтверджених треків."""
    tracks, nid, out = [], 1, []
    for t, D in enumerate(dets):
        for tr in tracks:
            tr.predict()
        hi = D[D[:, 4] >= high]
        lo = D[(D[:, 4] >= low) & (D[:, 4] < high)]
        m1, ut, ud = _associate(tracks, hi, match1)
        for i, j in m1:
            tracks[i].update(hi[j], n_init)
        active = [tracks[i] for i in ut if tracks[i].since == 1 and tracks[i].confirmed]
        m2, _, _ = _associate(active, lo, match2)
        for i, j in m2:
            z = lo[j].copy()
            z[2:4] = active[i].box()[2:4]
            active[i].update(z, n_init)
        tracks = [tr for tr in tracks if tr.confirmed or tr.since == 0]
        for j in ud:
            if hi[j, 4] >= new_thr:
                tracks.append(Track(nid, hi[j]))
                nid += 1
        tracks = [tr for tr in tracks if tr.since <= max_age]
        out += [(t, tr.id, *tr.box()) for tr in tracks if tr.confirmed and tr.since == 0]
    return out


# ───────────────────────────── треклети ─────────────────────────────


@dataclass
class Tracklet:
    cam: int                      # 1 або 2 — номер дрона
    tid: int                      # ID трекера на своїй камері
    t: np.ndarray                 # кадри
    boxes: np.ndarray             # (n, 4): cx, cy, w, h
    gt: Optional[int] = None      # GT-ID за більшістю кадрів — лише для оцінки, у зшивці не використовується
    crops: list = field(default_factory=list)
    emb: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def t0(self):
        return self.t[0] / FPS

    @property
    def t1(self):
        return self.t[-1] / FPS

    @property
    def key(self):
        return f"{self.cam}:{self.tid}"


def build_tracklets(hyp, cam: int, gt_frames=None, min_len: int = 5, iou_thr: float = 0.5) -> List[Tracklet]:
    """Вихід трекера -> треклети (≥ min_len кадрів). Якщо дано gt_frames, кожному треклету
    ставимо GT-ID, з яким він збігається (IoU ≥ iou_thr) щонайменше в половині своїх кадрів."""
    by_id: Dict[int, list] = {}
    for t, tid, cx, cy, w, h in hyp:
        by_id.setdefault(tid, []).append((t, cx, cy, w, h))
    out = []
    for tid, rows in by_id.items():
        if len(rows) < min_len:
            continue
        a = np.array(sorted(rows))
        tr = Tracklet(cam, tid, a[:, 0].astype(int), a[:, 1:5])
        if gt_frames is not None:
            votes: Dict[int, int] = {}
            for t, b in zip(tr.t, tr.boxes):
                g = gt_frames[t]
                if not g:
                    continue
                M = iou_matrix([b], [x[1:] for x in g])[0]
                k = int(np.argmax(M))
                if M[k] >= iou_thr:
                    votes[g[k][0]] = votes.get(g[k][0], 0) + 1
            if votes:
                g, n = max(votes.items(), key=lambda kv: kv[1])
                tr.gt = g if n >= len(tr.t) / 2 else None
        out.append(tr)
    return out


def read_crops(seq: str, tracklets: Sequence[Tracklet], rows: Dict[str, np.ndarray]):
    """Кропи по боксу без поля, повернуті довшою стороною вертикально (дрони бачать ту саму машину
    під різними кутами, а DINO не інваріантний до повороту). rows = {tracklet.key: індекси рядків tr.boxes}.
    Кожен кадр читаємо один раз."""
    from PIL import Image
    need: Dict[Tuple[int, int], list] = {}
    for tr in tracklets:
        tr.crops = []
        for i in rows.get(tr.key, []):
            need.setdefault((tr.cam, int(tr.t[i])), []).append((tr, int(i)))
    for (cam, t), items in sorted(need.items()):
        im = Image.open(frame_path(seq, cam, t)).convert("RGB")
        for tr, i in items:
            cx, cy, w, h = tr.boxes[i]
            c = im.crop((int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)))
            tr.crops.append(c.rotate(90, expand=True) if c.width > c.height else c)


def hsv_hist(crops, bins=(8, 4, 4)) -> np.ndarray:
    """Колірна гістограма HSV кожного кропа (корінь — щоб косинус поводився як відстань Гелінгера)."""
    out = []
    for c in crops:
        a = np.asarray(c.convert("HSV").resize((32, 64)), float).reshape(-1, 3)
        h, _ = np.histogramdd(a, bins=bins, range=((0, 256),) * 3)
        v = np.sqrt(h.ravel() / max(h.sum(), 1))
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.array(out)


def dino_encoder(size: str = "base", device: Optional[str] = None):
    """DINOv2 із заняття 5 (ваги — tools/fetch_dino_models.py). panels.py заняття 5 вантажимо за шляхом,
    бо в заняття 9 є власний panels.py."""
    import sys
    D = sys.modules.get("panels_l05")
    if D is None:
        spec = importlib.util.spec_from_file_location("panels_l05", REPO / "lessons" / "05-vit-multimodal" / "panels.py")
        D = importlib.util.module_from_spec(spec)
        sys.modules["panels_l05"] = D   # dataclasses шукають модуль у sys.modules ще під час виконання
        spec.loader.exec_module(D)
    D.setup_hf(dino=["dinov2"], size=size)
    return D.load_dino(["dinov2"], size=size, device=device)["dinov2"]


# ───────────────────────────── час і простір ─────────────────────────────


def time_overlap(A: Sequence[Tracklet], B: Sequence[Tracklet]) -> np.ndarray:
    """Перекриття інтервалів існування треклетів, с: > 0 — бачили одночасно, < 0 — розрив між ними."""
    a0, a1 = np.array([a.t0 for a in A]), np.array([a.t1 for a in A])
    b0, b1 = np.array([b.t0 for b in B]), np.array([b.t1 for b in B])
    return np.minimum(a1[:, None], b1[None]) - np.maximum(a0[:, None], b0[None])


def inter_drone_homographies(seq: str, step: int = 10, min_inliers: int = 15, scale: float = 0.5):
    """Гомографія «кадр дрона 1 → кадр дрона 2» з SIFT + RANSAC (заняття 2–3), без жодної розмітки.
    Рахуємо кожен step-й кадр і беремо найближчу до кадру t; якщо узгоджених точок < min_inliers —
    геометрії на цьому відрізку немає. Повертає ({t: H 3×3}, кадри замірів, кількість inliers)."""
    import cv2
    sift = cv2.SIFT_create(4000)
    n = len(frame_list(seq, 1))
    ts = np.arange(0, n, step)
    Hs, inliers = {}, []
    for t in ts:
        ims = [cv2.resize(cv2.imread(str(frame_path(seq, d, int(t))), cv2.IMREAD_GRAYSCALE), None, fx=scale, fy=scale)
               for d in (1, 2)]
        (k1, d1), (k2, d2) = [sift.detectAndCompute(im, None) for im in ims]
        good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance] if d1 is not None and d2 is not None else []
        if len(good) < 12:
            inliers.append(0)
            continue
        src = np.float32([k1[a.queryIdx].pt for a in good]) / scale
        dst = np.float32([k2[a.trainIdx].pt for a in good]) / scale
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
        ni = int(inl.sum()) if inl is not None else 0
        inliers.append(ni)
        if ni >= min_inliers:
            Hs[int(t)] = Hm
    keys = np.array(sorted(Hs))
    full = {}
    if len(keys):
        for t in range(n):
            k = keys[np.argmin(np.abs(keys - t))]
            if abs(k - t) <= step:
                full[t] = Hs[int(k)]
    return full, ts, np.array(inliers)


def geo_matrix(A: Sequence[Tracklet], B: Sequence[Tracklet], proj: Dict[str, Dict[int, np.ndarray]],
               min_common: int = 5) -> np.ndarray:
    """Медіанна відстань (у довжинах машини) між проєкцією треклета з дрона 1 і треклетом дрона 2
    за спільними кадрами. proj = {a.key: {t: (x, y) у кадрі дрона 2}}. NaN — замало спільних кадрів із геометрією."""
    Bt = [dict(zip(b.t.tolist(), b.boxes)) for b in B]
    D = np.full((len(A), len(B)), np.nan)
    for i, a in enumerate(A):
        pa = proj.get(a.key, {})
        for j, bt in enumerate(Bt):
            common = [t for t in pa if t in bt]
            if len(common) >= min_common:
                d = [np.hypot(*(pa[t] - bt[t][:2])) / max(bt[t][2], bt[t][3], 1) for t in common]
                D[i, j] = float(np.median(d))
    return D


# ───────────────────────────── оцінка ─────────────────────────────


def same_target(A: Sequence[Tracklet], B: Sequence[Tracklet]) -> np.ndarray:
    """Матриця «та сама ціль» за GT — лише для оцінки й гістограм."""
    return np.array([[a.gt is not None and a.gt == b.gt for b in B] for a in A], bool).reshape(len(A), len(B))


def app_quality(S: np.ndarray, G: np.ndarray, cand: np.ndarray) -> Dict[str, float]:
    """Якість візуального опису серед кандидатів: ROC AUC «та сама / різні» і Rank-1 (найсхожіший кандидат — та сама ціль)."""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(G[cand], S[cand]) if G[cand].any() and (~G[cand]).any() else np.nan
    r1 = [G[i, np.where(cand[i])[0][np.argmax(S[i, cand[i]])]] for i in range(len(S)) if (G[i] & cand[i]).any()]
    return dict(AUC=float(auc), Rank1=float(np.mean(r1)) if r1 else np.nan, n=len(r1), cand_per_track=float(cand.sum(1).mean()))


def zones(S: np.ndarray, pairs: Sequence[Tuple[int, int]], t_low: float, t_high: float):
    """Три зони: auto (s ≥ t_high) — зшиваємо, operator (t_low ≤ s < t_high) — на перевірку, new (s < t_low) — нова ціль."""
    z = {"auto": [], "operator": [], "new": []}
    for i, j in pairs:
        z["auto" if S[i, j] >= t_high else "operator" if S[i, j] >= t_low else "new"].append((i, j))
    return z


def resolve_operator(A, B, z) -> List[Tuple[int, int]]:
    """Оператор переглядає зону operator і підтверджує лише правильні пари (симулюємо через GT)."""
    G = same_target(A, B)
    return list(z["auto"]) + [(i, j) for i, j in z["operator"] if G[i, j]]


def link_report(A, B, links) -> Dict[str, float]:
    """ID persistence — частка GT-цілей, що мають треклети на обох камерах, для яких є правильна зшивка.
    Хибні зшивки — частка зшивок між різними цілями (або з треклетом-фантомом без GT)."""
    G = same_target(A, B)
    shared = {a.gt for a in A if a.gt is not None} & {b.gt for b in B if b.gt is not None}
    ok_ids = {A[i].gt for i, j in links if G[i, j]}
    wrong = sum(1 for i, j in links if not G[i, j])
    return dict(persistence=len(ok_ids & shared) / max(len(shared), 1), false_rate=wrong / max(len(links), 1),
                links=len(links), wrong=wrong, shared=len(shared))


def sweep_thresholds(S, assign: Callable, A, B, lows, highs, clip_s: float, sec_per_pair: float = 10.0):
    """Таблиця для вибору T_low / T_high. assign(S, gate) — призначення з гейтом: T_low одночасно і гейт
    перед угорським алгоритмом, і нижня межа зони оператора, тож для кожного T_low призначення рахуємо заново."""
    import pandas as pd
    rows = []
    for lo in lows:
        pairs = assign(S, lo)
        for hi in highs:
            if hi < lo:
                continue
            z = zones(S, pairs, lo, hi)
            auto = link_report(A, B, z["auto"])
            final = link_report(A, B, resolve_operator(A, B, z))
            rows.append(dict(T_low=lo, T_high=hi, auto=len(z["auto"]), false_auto=auto["false_rate"],
                             operator=len(z["operator"]), operator_share=len(z["operator"]) / max(len(pairs), 1),
                             operator_h_per_day=operator_hours(len(z["operator"]), clip_s, sec_per_pair),
                             persistence_auto=auto["persistence"], persistence_final=final["persistence"]))
    return pd.DataFrame(rows)


def operator_hours(n_pairs: int, clip_s: float, sec_per_pair: float = 10.0) -> float:
    """Години оператора на добу, якщо такий потік пар іде цілодобово (слайд «три зони порогів»)."""
    return n_pairs * sec_per_pair / clip_s * 86400 / 3600


def global_ids(A, B, links) -> Dict[str, int]:
    """Глобальні ID: зшиті треклети отримують спільний ID, решта — власний."""
    gid = {tr.key: k for k, tr in enumerate(list(A) + list(B), 1)}
    for i, j in links:
        gid[B[j].key] = gid[A[i].key]
    return gid


def mtmc_trackeval(gt1, gt2, A, B, links, run="mtmc"):
    """HOTA / IDF1 по мережі з двох камер: кадри камери 2 дописуємо після кадрів камери 1, ID — глобальні.
    Тоді зшивка між камерами — це асоціація в часі, і TrackEval її рахує (AssA, IDF1)."""
    gid = global_ids(A, B, links)
    T1 = len(gt1)
    gt = [list(f) for f in gt1] + [list(f) for f in gt2]
    hyp = [(int(t), gid[tr.key], *b) for tr in A for t, b in zip(tr.t, tr.boxes)]
    hyp += [(int(t) + T1, gid[tr.key], *b) for tr in B for t, b in zip(tr.t, tr.boxes)]
    return run_trackeval({run: gt}, {"net": {run: hyp}}, run=run)["net"]["COMBINED_SEQ"]


class Timer:
    """Сумарний час по етапах: with timer('етап'): ..."""

    def __init__(self):
        self.ms: Dict[str, float] = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                timer.ms[name] = timer.ms.get(name, 0) + 1000 * (time.perf_counter() - self.t)
        return _Ctx()


# ───────────────────────────── малювання ─────────────────────────────


def show_pairs(A, B, pairs, S=None, title="", n=6):
    """Галерея пар: верхній ряд — кроп із дрона 1, нижній — з дрона 2; підпис — схожість і чи це та сама ціль."""
    import matplotlib.pyplot as plt
    pairs = list(pairs)[:n]
    if not pairs:
        print(f"{title}: немає пар")
        return
    fig, axs = plt.subplots(2, len(pairs), figsize=(1.9 * len(pairs) + 1, 4.4), squeeze=False)
    for k, (i, j) in enumerate(pairs):
        for r, tr in enumerate((A[i], B[j])):
            ax = axs[r, k]
            if tr.crops:
                ax.imshow(tr.crops[len(tr.crops) // 2])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"д{tr.cam} id{tr.tid} · {tr.t0:.1f}–{tr.t1:.1f} с", fontsize=8)
        same = A[i].gt is not None and A[i].gt == B[j].gt
        axs[1, k].set_xlabel((f"s={S[i, j]:.2f} " if S is not None else "") + ("та сама" if same else "різні"), fontsize=9,
                             color="#4ADE80" if same else "#F87171")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_projection(seq: str, t: int, A, B, proj, ax=None):
    """Кадр дрона 2: бокси його треклетів і проєкції центрів треклетів дрона 1 (хрестики)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    ax = ax or plt.subplots(figsize=(12, 6.8))[1]
    ax.imshow(Image.open(frame_path(seq, 2, t)))
    for b in B:
        k = np.where(b.t == t)[0]
        if len(k):
            cx, cy, w, h = b.boxes[k[0]]
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="#38BDF8", lw=1.2))
    pts = np.array([proj[a.key][t] for a in A if t in proj.get(a.key, {})]).reshape(-1, 2)
    ax.scatter(pts[:, 0], pts[:, 1], marker="x", s=40, c="#F97316", lw=2)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"{seq}-2, кадр {t}: бокси дрона 2 (сині) і проєкції треклетів дрона 1 (×)")
