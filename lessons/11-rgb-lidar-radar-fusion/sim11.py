"""Синтетична сцена для заняття 11: ray-cast LiDAR і камери з відомою T_cam_lidar. Геометрія — у geom11.py.

Рамка LiDAR: x — вперед, y — ліворуч, z — вгору (FLU, як Velodyne у KITTI).
Рамка камери: x — праворуч, y — вниз, z — вперед (RDF, OpenCV).
"""
import numpy as np

from geom11 import se3, inv, R_AXES, pose

# як у KITTI (Tr_velo_to_cam, t ≈ (0, −0.08, −0.27)): камера ~0.27 м попереду і ~0.08 м нижче LiDAR
T_CAM_LIDAR = se3(R_AXES, np.array([0.0, -0.08, -0.27]))

# ---------- камера ----------
class Camera:
    def __init__(self, W=1242, H=375, f=721.5, cx=609.6, cy=172.9):
        self.W, self.H = W, H
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])

KITTI_CAM = Camera()

# ---------- сцена: AABB у рамці LiDAR ----------
GROUND_Z = -1.73
SCENE = [  # (назва, min xyz, max xyz, колір RGB 0..1, відбивна здатність)
    ('стіна', (38, -25, GROUND_Z), (39, 25, 6), (0.42, 0.40, 0.38), 0.3),
    ('авто 1', (12, 1.2, GROUND_Z), (16.4, 3.0, -0.25), (0.20, 0.35, 0.60), 0.6),
    ('авто 2', (22, -4.8, GROUND_Z), (26.4, -3.0, -0.3), (0.65, 0.18, 0.15), 0.5),
    ('людина', (9.0, -1.9, GROUND_Z), (9.5, -1.4, 0.02), (0.85, 0.65, 0.20), 0.2),
    ('стовп', (7.0, 4.6, GROUND_Z), (7.25, 4.85, 4.5), (0.55, 0.55, 0.55), 0.4),
    ('стовп 2', (7.0, -7.4, GROUND_Z), (7.25, -7.15, 4.5), (0.55, 0.55, 0.55), 0.4),
    ('дріт', (7.12, -7.3, 3.90), (7.135, 4.75, 3.915), (0.12, 0.12, 0.12), 0.1),  # дріт Ø 1.5 см між стовпами
    ('будівля', (14, 7.0, GROUND_Z), (34, 12, 5.0), (0.35, 0.33, 0.30), 0.3),
    ('кущі', (18, -12, GROUND_Z), (30, -9.5, -0.6), (0.18, 0.35, 0.16), 0.2),
]

def raycast(O, D, scene=SCENE, ground=True, tmax=120.):
    """O: (3,) або (N,3), D: (N,3) одиничні. Повертає t (N,), id (N,), -1 = земля, -2 = промах."""
    D = np.asarray(D, float); N = len(D)
    O = np.broadcast_to(np.asarray(O, float), (N, 3))
    best = np.full(N, np.inf); ids = np.full(N, -2)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d = 1.0 / D
        for k, (_, lo, hi, _, _) in enumerate(scene):
            t1 = (np.array(lo) - O) * inv_d; t2 = (np.array(hi) - O) * inv_d
            tn = np.nanmax(np.minimum(t1, t2), 1); tf = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tf >= np.maximum(tn, 0)) & (tn > 0.05) & (tn < best)
            best[hit] = tn[hit]; ids[hit] = k
        if ground:
            tg = (GROUND_Z - O[:, 2]) / D[:, 2]
            hit = (tg > 0.05) & (tg < best)
            best[hit] = tg[hit]; ids[hit] = -1
    miss = best > tmax
    best[miss] = np.inf; ids[miss] = -2
    return best, ids

# ---------- LiDAR ----------
def hdl64_dirs(az_step=0.18, fov=(-24.8, 2.0), n=64, az_range=(-180, 180)):
    """HDL-64E на 10 Гц: ~0.18° за азимутом (0.09° — лише на 5 Гц), 64 кільця, 26.8° верт."""
    el = np.radians(np.linspace(fov[1], fov[0], n))
    az = np.radians(np.arange(az_range[0], az_range[1], az_step))
    A, E = np.meshgrid(az, el)
    D = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    return D, A.ravel(), E.ravel()

def lidar_scan(az_step=0.18, n=64, fov=(-24.8, 2.0), noise=0.02, seed=0, az_range=(-180, 180),
               rotation_hz=10.0, ego_v=0.0, ego_w_deg=0.0, scene=None):
    """Скан із часом кожної точки. ego_v (м/с вперед) і ego_w (°/с рискання) — рух під час оберту:
    точки записані в рамці LiDAR у момент свого виміру (тобто «з перекосом»)."""
    rng = np.random.default_rng(seed)
    D, A, E = hdl64_dirs(az_step, fov, n, az_range)
    # час точки: оберт починається ззаду (az = -180) і йде проти год. стрілки
    t = ((A - np.radians(az_range[0])) / (2 * np.pi)) / rotation_hz
    # поза сенсора в момент t (рамка світу = рамка LiDAR у t=0.05, середина оберту)
    Rz, pos = pose(t, ego_v, ego_w_deg, t_mid=0.5 / rotation_hz)
    yaw = np.radians(ego_w_deg) * (t - 0.5 / rotation_hz)
    Dw = np.einsum('nij,nj->ni', Rz, D)
    sc = SCENE if scene is None else scene
    r, ids = raycast(pos, Dw, sc)
    ok = np.isfinite(r)
    r = r[ok] + rng.normal(0, noise, ok.sum())
    Xw = pos[ok] + Dw[ok] * r[:, None]
    # у рамку сенсора на момент виміру
    X = np.einsum('nji,nj->ni', Rz[ok], Xw - pos[ok])
    refl = np.array([sc[i][4] if i >= 0 else 0.25 for i in ids[ok]])
    return dict(xyz=X, t=t[ok], ids=ids[ok], refl=refl, xyz_world=Xw, yaw=yaw[ok], pos=pos[ok])

# ---------- рендер камери ----------
def render_rgb(cam=KITTI_CAM, T_cam_lidar=T_CAM_LIDAR, scene=SCENE):
    """Справжній кадр сцени з тієї ж геометрії (ray-cast кожного пікселя)."""
    W, H, K = cam.W, cam.H, cam.K
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)
    T_lc = inv(T_cam_lidar)
    O = T_lc[:3, 3]; Dl = rays_c @ T_lc[:3, :3].T
    t, ids = raycast(O, Dl, scene)
    img = np.zeros((W * H, 3))
    sky = ids == -2
    img[sky] = np.array([0.55, 0.65, 0.78]) * (0.75 + 0.25 * (np.arange(W * H)[sky] // W) / H)[:, None]
    g = ids == -1
    P = O + Dl[g] * t[g, None]
    check = ((np.floor(P[:, 0] / 2) + np.floor(P[:, 1] / 2)) % 2) * 0.04
    img[g] = (np.array([0.30, 0.30, 0.31]) + check[:, None]) * np.clip(1.2 - t[g, None] / 80, 0.4, 1)
    for k, (_, lo, hi, col, _) in enumerate(scene):
        m = ids == k
        if not m.any():
            continue
        P = O + Dl[m] * t[m, None]
        # затінення за нормаллю грані
        lo_, hi_ = np.array(lo), np.array(hi)
        dmin = np.minimum(np.abs(P - lo_), np.abs(P - hi_))
        ax = np.argmin(dmin, 1)
        shade = np.array([0.95, 0.75, 1.0])[ax]
        img[m] = np.array(col) * shade[:, None] * np.clip(1.15 - t[m, None] / 90, 0.5, 1)
    depth = t.reshape(H, W)
    return np.clip(img.reshape(H, W, 3), 0, 1), depth, ids.reshape(H, W)

ROOM = [  # «двір» для демо deskew: стіни навколо
    ('стіна +x', (30, -20, GROUND_Z), (30.5, 20, 3), (0.4, 0.4, 0.4), 0.3),
    ('стіна -x', (-30.5, -20, GROUND_Z), (-30, 20, 3), (0.4, 0.4, 0.4), 0.3),
    ('стіна +y', (-30, 20, GROUND_Z), (30, 20.5, 3), (0.4, 0.4, 0.4), 0.3),
    ('стіна -y', (-30, -20.5, GROUND_Z), (30, -20, 3), (0.4, 0.4, 0.4), 0.3),
    ('стовп', (8, 5, GROUND_Z), (8.4, 5.4, 3), (0.5, 0.5, 0.5), 0.4),
    ('авто', (-14, -6, GROUND_Z), (-9.6, -4.2, -0.3), (0.2, 0.3, 0.6), 0.5),
]
