"""KITTI object для заняття 11 і ДЗ 11: калібрування, хмара й кадр, розмітка, GT дальності і статуси відмов.

Дані — python tools/fetch_kitti.py --object (перші 200 кадрів training/ в artifacts/kitti/object/training)
або будь-яка тека зі змінної середовища KITTI_OBJECT_ROOT.
"""
import os
from functools import lru_cache
from pathlib import Path
import numpy as np

import geom11 as G

ROOT = Path(os.environ.get('KITTI_OBJECT_ROOT',
                           Path(__file__).resolve().parents[2] / 'artifacts' / 'kitti' / 'object' / 'training'))
NO_KITTI = 'синтетична сцена (KITTI не знайдено: python tools/fetch_kitti.py --object)'
CLASS_H = {'Car': 1.53, 'Van': 2.2, 'Truck': 3.2, 'Pedestrian': 1.75, 'Cyclist': 1.74}   # апріорна висота класу, м
PARTS = {'velodyne': '.bin', 'image_2': '.png', 'calib': '.txt', 'label_2': '.txt'}


def stem(idx):
    return idx if isinstance(idx, str) else f'{idx:06d}'


def available(idx=0):
    s = stem(idx)
    return all((ROOT / d / f'{s}{ext}').exists() for d, ext in PARTS.items())


@lru_cache(maxsize=None)
def calib(idx):
    """calib/*.txt → (P2 3×4, R0_rect 4×4, Tr_velo_to_cam 4×4). Слайд «Рамки · KITTI»: y = P2 · R0_rect · Tr_velo_to_cam · X."""
    d = {}
    for ln in open(ROOT / 'calib' / f'{stem(idx)}.txt'):
        if ':' in ln:
            k, v = ln.split(':', 1)
            d[k.strip()] = np.array(v.split(), dtype=float)
    R0 = np.eye(4); R0[:3, :3] = d['R0_rect'].reshape(3, 3)
    Tv = np.eye(4); Tv[:3] = d['Tr_velo_to_cam'].reshape(3, 4)
    return d['P2'].reshape(3, 4), R0, Tv


def P(idx):
    """Одна матриця 3×4: velo → пікселі камери 2."""
    P2, R0, Tv = calib(idx)
    return P2 @ R0 @ Tv


def cam(idx):
    """K (3×3) і T_cam2_velo (4×4): P2 = K·[I | K⁻¹·P2[:, 3]], тож T = [I | K⁻¹·P2[:, 3]] · R0_rect · Tr_velo_to_cam.
    Так проєкція й збурення калібрування працюють з тією самою парою (T, K), що й на синтетиці."""
    P2, R0, Tv = calib(idx)
    K = P2[:, :3]
    return K, G.se3(np.eye(3), np.linalg.solve(K, P2[:, 3])) @ R0 @ Tv


@lru_cache(maxsize=8)
def points_image(idx):
    """Хмара (N×3, м, рамка Velodyne) і кадр RGB у [0, 1]. Кеш: слайдери не перечитують файли на кожен рух;
    масиви лише для читання, бо спільні між викликами."""
    import cv2
    s = stem(idx)
    pts = np.fromfile(ROOT / 'velodyne' / f'{s}.bin', dtype=np.float32).reshape(-1, 4)
    img = cv2.cvtColor(cv2.imread(str(ROOT / 'image_2' / f'{s}.png')), cv2.COLOR_BGR2RGB)
    xyz, img = pts[:, :3].astype(float), img / 255.0
    xyz.flags.writeable = img.flags.writeable = False
    return xyz, img


def labels(idx):
    """label_2 → об'єкти класів з CLASS_H: клас, truncated (0…1), occluded (0…3), 2D-бокс у пікселях image_2,
    розміри h, w, l (м), location — центр нижньої грані 3D-боксу в rect cam0, поворот ry навколо вертикалі."""
    objs = []
    for ln in open(ROOT / 'label_2' / f'{stem(idx)}.txt'):
        f = ln.split()
        if not f or f[0] not in CLASS_H:              # DontCare, Misc, Tram, Person_sitting — пропускаємо
            continue
        objs.append(dict(cls=f[0], trunc=float(f[1]), occl=int(f[2]), box=np.array(f[4:8], float),
                         hwl=np.array(f[8:11], float), loc=np.array(f[11:14], float), ry=float(f[14])))
    return objs


def box_frame(o):
    """Поворот і центр нижньої грані 3D-боксу в rect cam0 (вісь x об'єкта — довжина, y — вниз, z — ширина)."""
    c, s = np.cos(o['ry']), np.sin(o['ry'])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), o['loc']


def box_corners(o):
    h, w, l = o['hwl']
    R, t = box_frame(o)
    xs, zs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2, np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    ys = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    return (R @ np.vstack([xs, ys, zs])).T + t


def visible_depth(o, P2, n=40):
    """GT дальності: медіанна глибина видимої поверхні 3D-боксу — промені з камери 2 через сітку n × n пікселів
    2D-боксу, перша точка входу в бокс. LiDAR бачить поверхню, а location — центр нижньої грані: для машини задом
    різниця ≈ 2 м. Сітка рівномірна в пікселях, як і точки LiDAR, тож довгий бік машини не перетягує медіану."""
    h, w, l = o['hwl']
    R, t = box_frame(o)
    half = np.array([l / 2, h / 2, w / 2]); mid = np.array([0, -h / 2, 0])
    c = -np.linalg.solve(P2[:, :3], P2[:, 3])                 # центр камери 2 у rect cam0
    x0, y0, x1, y1 = o['box']
    u, v = [a.ravel() for a in np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))]
    rays = np.linalg.solve(P2[:, :3], np.vstack([u, v, np.ones_like(u)])).T
    org, d = R.T @ (c - t) - mid, rays @ R                    # у рамці боксу з центром у його середині
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (-half - org) / d, (half - org) / d
    t_in = np.nanmax(np.minimum(t1, t2), 1); t_out = np.nanmin(np.maximum(t1, t2), 1)
    hit = (t_in <= t_out) & (t_in > 0)
    z = (c + rays[hit] * t_in[hit, None]) @ P2[2, :3] + P2[2, 3]
    return float(np.median(z)) if hit.any() else np.nan


def status(o):
    """Чому дальність об'єкта не можна брати в метрику (поріг висоти — KITTI moderate). 'ok' — можна;
    '< 5 точок' додає той, хто рахує дальність LiDAR."""
    h_px = o['box'][3] - o['box'][1]
    return 'оклюзія' if o['occl'] >= 2 else 'обрізаний' if o['trunc'] > 0.3 else 'малий бокс' if h_px < 25 else 'ok'


def motion_scan(v, w_deg, idx, t_world=0.05):
    """Справжня хмара KITTI, «знята» сенсором, що рухається: v м/с вперед і w_deg °/с ліворуч (demo 2).

    Час точки — з азимуту (оберт 100 мс від −180°). Світ — рамка сенсора в момент t_world: саме в ній хмара KITTI
    узгоджена з кадром камери. Рух змодельований: у KITTI object немає поз і міток часу (вони є лише в KITTI raw)."""
    xyz, img = points_image(idx)
    t = G.point_times(xyz)
    return dict(xyz=G.skew(xyz, t, v, w_deg, t_world), t=t, xyz_world=xyz, t_world=t_world, img=img)
