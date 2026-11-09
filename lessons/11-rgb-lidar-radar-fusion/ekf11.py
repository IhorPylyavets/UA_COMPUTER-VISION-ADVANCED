"""EKF для злиття радара і камери: стан цілі [p, v] у рамці щогли (x — північ/вперед, y — ліворуч, z — вгору).

Радар: R, az, el, v_r (10 Гц). Камера: пікселі (u, v) (30 Гц), лише напрямок.
Усі вимірювання обробляються за їхніми мітками часу: predict до t_meas → update.
"""
import numpy as np

from geom11 import R_AXES as C_AXES                         # рамка щогли FLU → камера RDF


def F_Q(dt, q=1.0):
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    G = np.vstack([np.eye(3) * dt ** 2 / 2, np.eye(3) * dt])
    return F, q ** 2 * G @ G.T          # білий шум прискорення, σ_a = q м/с²


def h_radar(x):
    p, v = x[:3], x[3:]
    r = np.linalg.norm(p)
    return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(p[2] / r), p @ v / r])


def H_radar(x):
    p, v = x[:3], x[3:]
    r = np.linalg.norm(p); rxy2 = p[0] ** 2 + p[1] ** 2; rxy = np.sqrt(rxy2)
    H = np.zeros((4, 6))
    H[0, :3] = p / r
    H[1, :2] = [-p[1] / rxy2, p[0] / rxy2]
    H[2, :3] = [-p[0] * p[2] / (r ** 2 * rxy), -p[1] * p[2] / (r ** 2 * rxy), rxy / r ** 2]
    H[3, :3] = v / r - (p @ v) * p / r ** 3
    H[3, 3:] = p / r
    return H


def h_cam(x, f=2000.0, c=(960., 540.)):
    q = C_AXES @ x[:3]
    return np.array([f * q[0] / q[2] + c[0], f * q[1] / q[2] + c[1]])


def H_cam(x, f=2000.0):
    q = C_AXES @ x[:3]
    J = np.array([[f / q[2], 0, -f * q[0] / q[2] ** 2], [0, f / q[2], -f * q[1] / q[2] ** 2]])
    H = np.zeros((2, 6)); H[:, :3] = J @ C_AXES
    return H


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, x0, P0, q=1.0):
        self.x, self.P, self.q, self.t = x0.astype(float), P0.astype(float), q, None

    def predict(self, t):
        if self.t is not None and t > self.t:
            F, Q = F_Q(t - self.t, self.q)
            self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z, kind, R, gate=None):
        """gate — поріг χ² для NIS: вимір з d2 > gate відкидається без оновлення. Повертає (d2, прийнято)."""
        h, Hf = (h_radar, H_radar) if kind == 'radar' else (h_cam, H_cam)
        H = Hf(self.x)                                 # один якобіан у прогнозі: і для K, і для форми Джозефа
        y = z - h(self.x)
        if kind == 'radar':
            y[1] = wrap(y[1])
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))          # NIS: для gate і перевірки узгодженості
        if gate is not None and d2 > gate:
            return d2, False
        K = np.linalg.solve(S, H @ self.P).T           # = P Hᵀ S⁻¹ без явного обернення (P і S симетричні)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # форма Джозефа — стабільніша
        return d2, True


# ---------- сценарій: дрон перетинає поле зору щогли ----------
SIG_R, SIG_AZ, SIG_EL, SIG_VR = 2.0, np.radians(1.0), np.radians(1.5), 0.15
SIG_PX = 1.5


def truth(T=20.0, dt=0.005):
    t = np.arange(0, T, dt)
    p0 = np.array([350.0, 150.0, 60.0]); v0 = np.array([-4.0, -18.0, 0.0])
    a = np.zeros((len(t), 3)); a[(t > 8) & (t < 12), 0] = 2.0      # м'який маневр
    v = v0 + np.cumsum(a, 0) * dt
    p = p0 + np.cumsum(v, 0) * dt
    return t, p, v


def measurements(seed=0, cam_time_bias=0.0, radar=True, cam=True):
    rng = np.random.default_rng(seed)
    t, p, v = truth()
    Z = []
    if radar:
        for k in range(0, len(t), 20):                               # 10 Гц
            x = np.r_[p[k], v[k]]
            z = h_radar(x) + rng.normal(0, [SIG_R, SIG_AZ, SIG_EL, SIG_VR])
            Z.append((t[k], 'radar', z))
    if cam:
        for k in range(0, len(t), 7):                                # ~30 Гц
            x = np.r_[p[k], v[k]]
            z = h_cam(x) + rng.normal(0, SIG_PX, 2)
            Z.append((t[k] + cam_time_bias, 'cam', z))               # хибна мітка часу
    Z.sort(key=lambda m: m[0])
    return t, p, v, Z


CHI2_99 = {'radar': 13.28, 'cam': 9.21}                  # χ², 99 %: 4 і 2 ступені свободи


def run(variant='fused', seed=0, cam_time_bias=0.0, q=1.0, gate=False):
    """Стовпці виходу: t, |помилка позиції|, NEES позиції, |помилка швидкості|, вимір прийнято (1/0)."""
    radar = variant in ('fused', 'radar')
    cam = variant in ('fused', 'cam')
    t, p, v, Z = measurements(seed, cam_time_bias, radar, cam)
    # ініціалізація: з першого радарного виміру; «лише камера» — з грубої апріорної дальності 500 м
    x0 = np.r_[p[0] + np.random.default_rng(seed + 99).normal(0, 5, 3), 0, 0, 0]
    if variant == 'cam':
        x0[:3] = p[0] / np.linalg.norm(p[0]) * 500.0
    P0 = np.diag([30 ** 2] * 3 + [20 ** 2] * 3)
    if variant == 'cam':
        P0[:3, :3] = np.eye(3) * 200 ** 2
    f = EKF(x0, P0, q)
    Rr = np.diag([SIG_R, SIG_AZ, SIG_EL, SIG_VR]) ** 2
    Rc = np.eye(2) * SIG_PX ** 2
    out = []
    for tm, kind, z in Z:
        f.predict(tm)
        _, acc = f.update(z, kind, Rr if kind == 'radar' else Rc, gate=CHI2_99[kind] if gate else None)
        k = min(int(round(tm / 0.005)), len(t) - 1)
        e = f.x[:3] - p[k]
        nees = float(e @ np.linalg.solve(f.P[:3, :3], e))
        out.append((tm, np.linalg.norm(e), nees, np.linalg.norm(f.x[3:] - v[k]), float(acc)))
    return np.array(out)


def summary(cam_bias=0.04, seeds=range(20), t_from=5.0):
    res = {}
    for name, var, bias in (('лише радар', 'radar', 0), ('лише камера', 'cam', 0), ('радар + камера', 'fused', 0),
                            (f'радар + камера, мітка камери +{cam_bias * 1000:.0f} мс', 'fused', cam_bias)):
        E, N, V = [], [], []
        for s in seeds:
            o = run(var, s, bias)
            m = o[:, 0] > t_from
            E.append(np.sqrt(np.mean(o[m, 1] ** 2))); N.append(np.mean(o[m, 2])); V.append(np.sqrt(np.mean(o[m, 3] ** 2)))
        res[name] = (np.mean(E), np.mean(N), np.mean(V))
    return res
