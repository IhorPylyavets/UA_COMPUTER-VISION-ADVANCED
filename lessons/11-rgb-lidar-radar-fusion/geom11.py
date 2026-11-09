"""Геометрія заняття 11: без даних і без графіки — спільна для демо, сторінки demo 1 і ДЗ 11.

Рамка LiDAR — FLU (x вперед, y ліворуч, z вгору), рамка камери — RDF (OpenCV). Імена трансформів — T_to_from
(слайд «Рамки · SE(3)»): T_cam_lidar переводить точки з рамки LiDAR у рамку камери.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ---------- SE(3) і осі (слайди «Рамки · SE(3)», «Рамки · осі») ----------
def se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def inv(T):
    """Аналітичне обернення: [Rᵀ | −Rᵀ·t] — те саме C = −Rᵀ·t, що в заняття 2."""
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


# перестановка осей FLU → RDF: x_cam = −y_l, y_cam = −z_l, z_cam = x_l
R_AXES = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])


def perturb(T, yaw=0., pitch=0., roll=0., dt=(0., 0., 0.)):
    """Помилка калібрування: малі кути (°) навколо осей камери і зсув (м) у рамці камери — зліва від T."""
    dR = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=True).as_matrix()   # yaw — навколо y_cam (вниз)
    return se3(dR, np.asarray(dt, float)) @ T


def predicted_shift(yaw_deg, tx, Z, K, u0=None):
    """Прогноз зсуву по u (слайд «Калібрування · валідація»). На осі: f·δθ + f·δt/Z; поза віссю поворот дає
    f·[tan(θ + δθ) − tan θ] ≈ f·δθ / cos²θ."""
    f, cx = K[0, 0], K[0, 2]
    th = 0.0 if u0 is None else np.arctan((u0 - cx) / f)
    return f * (np.tan(th + np.radians(yaw_deg)) - np.tan(th)) + f * tx / Z


# ---------- проєкція і z-buffer (слайд «Проєкція · конвеєр») ----------
def _to_pixels(Y, W, H, zmin):
    z = Y[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = Y[:, 0] / z; v = Y[:, 1] / z
    m = (z > zmin) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, m


def project(xyz, T_cam_lidar, K, W, H, zmin=0.5):
    """Точки LiDAR (N×3) → u, v, z і маска: перед камерою (z > zmin) і в кадрі."""
    return _to_pixels((xyz @ T_cam_lidar[:3, :3].T + T_cam_lidar[:3, 3]) @ K.T, W, H, zmin)


def project_P(xyz, P, W, H, zmin=0.5):
    """Те саме через одну матрицю 3×4, як у KITTI: P = P2 · R0_rect · Tr_velo_to_cam."""
    return _to_pixels(np.c_[xyz, np.ones(len(xyz))] @ P.T, W, H, zmin)


def zbuffer(u, v, z, W, H):
    """Найближча точка на піксель. np.minimum.at — не залежить від порядку запису."""
    depth = np.full((H, W), np.inf)
    np.minimum.at(depth, (v.astype(int), u.astype(int)), z)
    return depth


# ---------- час і deskew (слайди «Час · бюджет», «Час · deskew») ----------
ROT_HZ = 10.0     # HDL-64E у KITTI: оберт 100 мс, починається ззаду (азимут −180°), вперед дивиться в середині


def point_times(xyz, rot_hz=ROT_HZ):
    """Час точки в оберті з її азимуту, с: від 0 (ззаду, −180°) до 1 / rot_hz."""
    return (np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi) / rot_hz


def pose(t, v, w_deg, t_mid=0.05):
    """Поза сенсора у світі для руху v м/с вперед і w_deg °/с ліворуч; світ — рамка LiDAR у момент t_mid.
    t — скаляр або масив: R (…×3×3), p (…×3)."""
    tc = np.asarray(t, float) - t_mid
    a = np.radians(w_deg) * tc
    c, s, o, i = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    R = np.stack([np.stack([c, -s, o], -1), np.stack([s, c, o], -1), np.stack([o, o, i], -1)], -2)
    return R, np.stack([v * tc, o, o], -1)


def skew(xyz_world, t, v, w_deg, t_world=0.05):
    """Кожна точка — у рамці сенсора в момент свого виміру t: так її віддає драйвер без deskew. R(t)ᵀ · (X − p(t))."""
    R, p = pose(t, v, w_deg, t_world)
    return np.einsum('nji,nj->ni', R, xyz_world - p)


def to_frame_at(scan, t_ref, v, w_deg):
    """Ідеальний deskew (пози відомі точно): кожна точка → рамка сенсора в момент t_ref."""
    Rr, pr = pose(t_ref, v, w_deg, scan.get('t_world', 0.05))
    return (scan['xyz_world'] - pr) @ Rr


def deskew(xyz, t, poses_t, poses_R, poses_p, t_ref):
    """Deskew за дискретними позами (як з IMU): Slerp для обертання, лінійно для позиції; точка з рамки сенсора
    в момент t_i → рамка сенсора в момент t_ref."""
    sl = Slerp(poses_t, Rotation.from_matrix(poses_R))
    Ri = sl(np.clip(t, poses_t[0], poses_t[-1])).as_matrix()
    pi = np.stack([np.interp(t, poses_t, poses_p[:, k]) for k in range(3)], 1)
    Xw = np.einsum('nij,nj->ni', Ri, xyz) + pi
    Rr = sl([t_ref]).as_matrix()[0]
    pr = np.array([np.interp(t_ref, poses_t, poses_p[:, k]) for k in range(3)])
    return (Xw - pr) @ Rr


# ---------- вирівнювання ребер (слайди «Калібрування · валідація», «Калібрування · поле») ----------
def depth_edges(xyz, jump=0.5):
    """Точки ближнього боку розриву глибини вздовж кільця LiDAR (у KITTI і в sim11 точки йдуть кільцями, азимут
    зростає). Повертає маску і вагу — величину стрибка, м."""
    r = np.linalg.norm(xyz[:, :2], axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    same = np.abs(np.diff(az)) < np.radians(0.6)            # сусід у тому ж кільці, без пропуску
    w = np.maximum(np.where(np.r_[False, same], np.r_[r[0], r[:-1]] - r, 0),
                   np.where(np.r_[same, False], np.r_[r[1:], r[-1]] - r, 0))
    return w > jump, w


def edge_map(img, sigma=3.0):
    """Ребра Canny → exp(−відстань / σ): 1 на ребрі, ≈ 0 далі за кілька σ пікселів. img — RGB у [0, 1]."""
    import cv2
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 50, 150)
    return np.exp(-cv2.distanceTransform(255 - e, cv2.DIST_L2, 3) / sigma)


def edge_score(xyz, w, E, T, K):
    """Середнє значення карти ребер у проєкціях розривів глибини, зважене величиною розриву (Levinson & Thrun)."""
    H, W = E.shape
    u, v, z, m = project(xyz, T, K, W, H)
    if not m.any():
        return np.nan
    ww = np.clip(w[m], 0, 5)
    return float(np.sum(ww * E[v[m].astype(int), u[m].astype(int)]) / ww.sum())


# ---------- дальність до цілі (слайд «Чесне порівняння») ----------
def lidar_range(u, v, z, box, bin_m=0.5, min_pts=5, cluster_m=2.0):
    """Frustum-точки в боксі → перший щільний кластер за глибиною → медіана. (None, n), якщо точок менше min_pts."""
    x0, y0, x1, y1 = box
    zz = np.sort(z[(u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)])
    if len(zz) < min_pts:
        return None, len(zz)
    hist, edges = np.histogram(zz, bins=np.arange(zz[0], zz[-1] + 2 * bin_m, bin_m))
    k = int(np.argmax(hist >= max(min_pts, 0.15 * hist.max())))
    return float(np.median(zz[(zz >= edges[k]) & (zz < edges[k] + cluster_m)])), len(zz)
