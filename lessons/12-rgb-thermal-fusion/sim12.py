"""Синтетична RGB + LWIR сцена для заняття 12 з відомим ground truth.

Що моделюємо:
  * радіометрію LWIR 8–14 мкм: Планк, емісивність, відбиття неба / оточення, атмосфера;
  * сигнальний ланцюг мікроболометра: counts 14 біт, FPN, NETD, тепловий час τ, FFC, AGC;
  * dual EO/IR-установку side-by-side: дві камери з різними K, роздільністю і FOV, baseline B,
    малий boresight-поворот між осями — T_rgb_t відомо точно;
  * калібрувальну мішень з підігрітими колами (висока ε) на алюмінії (низька ε).

Рамка установки (body): FLU — x вперед, y ліворуч, z вгору; земля z = GROUND_Z.
Рамки камер: RDF (OpenCV) — x праворуч, y вниз, z вперед. Конвенція імен — як на занятті 11: T_to_from.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

# =====================================================================
# Фізика: Планк, смугова радіація
# =====================================================================
H_PL, C_L, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23
SIGMA_SB = 5.670374419e-8
WIEN_B = 2.897771955e-3            # м·K


def planck(lam_m, T):
    """Спектральна яскравість чорного тіла, Вт / (м² · ср · м)."""
    lam_m = np.asarray(lam_m, float)
    x = H_PL * C_L / (lam_m * K_B * np.asarray(T, float))
    return 2 * H_PL * C_L ** 2 / lam_m ** 5 / np.expm1(x)


_LUT = {}


def band_radiance(T, lo_um=8.0, hi_um=14.0, n=600):
    """Інтеграл Планка у смузі [lo, hi] мкм, Вт / (м² · ср). T — скаляр або масив.
    Для великих масивів — таблиця 150–500 K з кроком 0.01 K (похибка інтерполяції < 1e-6 відносно)."""
    T = np.asarray(T, float)
    if T.size > 2000:
        key = (lo_um, hi_um)
        if key not in _LUT:
            Tg = np.arange(150.0, 500.0, 0.01)
            _LUT[key] = (Tg, np.concatenate([band_radiance(c, lo_um, hi_um, n) for c in np.array_split(Tg, 40)]))
        Tg, Lg = _LUT[key]
        return np.interp(T, Tg, Lg)
    lam = np.linspace(lo_um, hi_um, n) * 1e-6
    x = H_PL * C_L / (lam[:, None] * K_B * T.reshape(1, -1))
    L = 2 * H_PL * C_L ** 2 / lam[:, None] ** 5 / np.expm1(x)
    return np.trapezoid(L, lam, axis=0).reshape(T.shape)


def band_fraction(T, lo_um, hi_um):
    """Частка повної потужності чорного тіла (σT⁴) у смузі."""
    return band_radiance(T, lo_um, hi_um, n=4000) * np.pi / (SIGMA_SB * np.asarray(T, float) ** 4)


def dL_dT(T, lo_um=8.0, hi_um=14.0, dT=0.01):
    """Контраст на 1 K: d(смугова яскравість)/dT, Вт / (м² · ср · K)."""
    return (band_radiance(T + dT, lo_um, hi_um) - band_radiance(T - dT, lo_um, hi_um)) / (2 * dT)


def apparent_radiance(T_obj, eps, T_refl, tau=1.0, T_atm=None):
    """L = τ · [ε·L(T) + (1−ε)·L(T_відбите)] + (1−τ)·L(T_атм) — непрозорий об'єкт."""
    T_atm = T_refl if T_atm is None else T_atm
    return tau * (eps * band_radiance(T_obj) + (1 - eps) * band_radiance(T_refl)) + (1 - tau) * band_radiance(T_atm)


def T_from_radiance(L, lo=150.0, hi=450.0):
    """Обернення band_radiance (яскравісна температура чорного тіла), за таблицею."""
    Ts = np.linspace(lo, hi, 3001)
    return np.interp(L, band_radiance(Ts), Ts)


# =====================================================================
# Добовий цикл: повітря, сонце, поверхні (ілюстративна модель з тепловою інерцією)
# =====================================================================
def air_temp(h):
    """°C → K. Мінімум перед світанком, максимум ~15:00."""
    return 273.15 + 12.0 + 7.0 * np.sin(2 * np.pi * (np.asarray(h, float) - 9.0) / 24.0)


def sun(h):
    """Відносна сонячна опроміненість 0..1 (схід 6:00, захід 18:00)."""
    return np.clip(np.sin(np.pi * (np.asarray(h, float) - 6.0) / 12.0), 0, None)


def sky_temp(h, clear=True):
    """Яскравісна температура ясного неба в LWIR — на десятки K холодніша за повітря."""
    return air_temp(h) - (30.0 if clear else 5.0)


# матеріали: ε (LWIR), поглинання сонця α, теплова інерція (год), внутрішнє джерело ΔT (K), RGB-альбедо
MATERIALS = {
    'трава':      dict(eps=0.97, alpha=0.75, lag=1.5, heat=0.0,  rgb=(0.20, 0.33, 0.14), cool=4.0),
    'асфальт':    dict(eps=0.95, alpha=0.92, lag=2.5, heat=0.0,  rgb=(0.22, 0.22, 0.23), cool=3.0),
    'бетон':      dict(eps=0.92, alpha=0.65, lag=3.5, heat=0.0,  rgb=(0.55, 0.54, 0.50), cool=2.0),
    'людина':     dict(eps=0.98, alpha=0.60, lag=0.3, heat=None, rgb=(0.30, 0.34, 0.26), cool=0.0),
    'фарба авто': dict(eps=0.90, alpha=0.80, lag=1.0, heat=0.0,  rgb=(0.48, 0.10, 0.10), cool=2.0),
    'двигун':     dict(eps=0.90, alpha=0.80, lag=1.0, heat=35.0, rgb=(0.48, 0.10, 0.10), cool=0.0),
    'скло':       dict(eps=0.90, alpha=0.10, lag=0.5, heat=0.0,  rgb=(0.15, 0.20, 0.25), cool=1.0),
    'метал':      dict(eps=0.07, alpha=0.30, lag=0.5, heat=0.0,  rgb=(0.75, 0.76, 0.78), cool=0.0),
    'дрон':       dict(eps=0.85, alpha=0.70, lag=0.3, heat=12.0, rgb=(0.08, 0.08, 0.09), cool=0.0),
    'мішень':     dict(eps=0.95, alpha=0.50, lag=0.1, heat=0.0,  rgb=(0.05, 0.05, 0.05), cool=0.0),
}
T_PERSON_SURFACE = 273.15 + 31.0     # поверхня одягу/шкіри, усереднено


def surface_temp(material, h, up=False):
    """Температура поверхні в момент h (K). up — горизонтальна поверхня (більше сонця і більше втрат у небо)."""
    m = MATERIALS[material]
    if m['heat'] is None:                                   # людина: терморегуляція + невеликий solar loading
        return T_PERSON_SURFACE + 3.0 * sun(h)
    solar = 18.0 if up else 9.0
    s_lag = sun(np.asarray(h, float) - m['lag'])            # інерція: пік нагріву зсунутий
    night_cool = m['cool'] * (1.0 - np.clip(sun(h) * 3, 0, 1)) * (1.6 if up else 1.0)
    return air_temp(np.asarray(h, float) - 0.5 * m['lag']) + m['alpha'] * solar * s_lag - night_cool + m['heat']


# =====================================================================
# Камери і установка
# =====================================================================
class Camera:
    def __init__(self, W, H, hfov_deg, name):
        self.W, self.H, self.name = W, H, name
        f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
        self.K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
        self.f = f
        self.hfov = hfov_deg

    @property
    def ifov_deg(self):
        return self.hfov / self.W


RGB_CAM = Camera(1280, 720, 60.0, 'RGB')       # f ≈ 1108.5 px
T_CAM = Camera(640, 512, 45.0, 'LWIR')         # f ≈ 772.5 px, як 640×512 з об'єктивом ~13.6 мм при 17 мкм


def se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


R_AXES = np.array([[0., -1, 0], [0, 0, -1], [1, 0, 0]])     # FLU → RDF
CAM_HEIGHT = 2.0
GROUND_Z = -CAM_HEIGHT
# RGB-камера в початку body; тепловізор на 5 см праворуч і 1 см нижче, вісь трохи розвернута (boresight)
BASELINE = np.array([0.05, 0.01, 0.0])                       # у рамці RGB-камери (RDF): праворуч, вниз
BORESIGHT_DEG = dict(yaw=0.40, pitch=-0.25, roll=0.10)
T_RGB_BODY = se3(R_AXES, np.zeros(3))
_R_bs = Rot.from_euler('yxz', [BORESIGHT_DEG['yaw'], BORESIGHT_DEG['pitch'], BORESIGHT_DEG['roll']],
                       degrees=True).as_matrix()
T_RGB_T = se3(_R_bs, BASELINE)          # точки з рамки тепловізора → у рамку RGB
T_T_RGB = inv(T_RGB_T)
T_T_BODY = T_T_RGB @ T_RGB_BODY


def project(X_cam, K, W, H, zmin=0.1):
    """X_cam: N×3 у рамці камери → u, v, z, mask (у кадрі і перед камерою)."""
    z = X_cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0] * X_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * X_cam[:, 1] / z + K[1, 2]
    m = (z > zmin) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, m


def transform(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


# =====================================================================
# Сцена: AABB у body (FLU)
# =====================================================================
G = GROUND_Z
SCENE = [  # назва, min xyz, max xyz, матеріал
    ('стіна', (75, -40, G), (76, 40, 9.0), 'бетон'),
    ('вікно', (74.9, -6, 1.0), (75.0, 2, 5.0), 'скло'),
    ('людина 8 м', (8.0, 0.9, G), (8.35, 1.4, G + 1.75), 'людина'),
    ('людина 18 м', (18.0, -2.6, G), (18.35, -2.1, G + 1.75), 'людина'),
    ('людина 45 м', (45.0, 3.0, G), (45.35, 3.5, G + 1.75), 'людина'),
    ('авто', (25.0, 5.5, G + 0.15), (29.3, 7.3, G + 1.45), 'фарба авто'),
    ('двигун авто', (24.6, 5.6, G + 0.3), (25.0, 7.2, G + 1.0), 'двигун'),
    ('лист металу', (11.0, -6.0, G), (14.0, -3.5, G + 0.02), 'метал'),
    ('бетонні блоки', (32.0, -9.0, G), (34.0, -5.0, G + 1.0), 'бетон'),
    ('дрон', (60.0, -8.0, 6.0), (60.35, -7.65, 6.12), 'дрон'),
]
ASPHALT_Y = 2.5     # смуга асфальту |y| < 2.5 м, решта — трава


def raycast(O, D, scene=SCENE, tmax=400.0):
    """O: (3,), D: (N,3) одиничні → t, id, normal. id: -1 земля, -2 небо."""
    N = len(D)
    best = np.full(N, np.inf); ids = np.full(N, -2); nrm = np.zeros((N, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        invd = 1.0 / D
        for k, (_, lo, hi, _) in enumerate(scene):
            lo, hi = np.array(lo, float), np.array(hi, float)
            t1 = (lo - O) * invd; t2 = (hi - O) * invd
            tn = np.nanmax(np.minimum(t1, t2), 1); tf = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tf >= np.maximum(tn, 0)) & (tn > 0.05) & (tn < best)
            if hit.any():
                best[hit] = tn[hit]; ids[hit] = k
                ax = np.argmax(np.minimum(t1, t2)[hit], 1)        # грань входу
                n = np.zeros((hit.sum(), 3)); n[np.arange(hit.sum()), ax] = -np.sign(D[hit, ax])
                nrm[hit] = n
        tg = (GROUND_Z - O[2]) / D[:, 2]
        g = (D[:, 2] < 0) & (tg > 0) & (tg < best) & (tg < tmax)
    best[g] = tg[g]; ids[g] = -1; nrm[g] = (0, 0, 1)
    best[ids == -2] = np.inf
    return best, ids, nrm


def camera_rays(cam, T_cam_body):
    """Промені всіх пікселів у body. Повертає O (3,), D (H·W, 3)."""
    uu, vv = np.meshgrid(np.arange(cam.W), np.arange(cam.H))
    d_cam = np.stack([(uu - cam.K[0, 2]) / cam.K[0, 0], (vv - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(uu, float)], -1)
    d_cam = d_cam.reshape(-1, 3); d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    T_body_cam = inv(T_cam_body)
    D = d_cam @ T_body_cam[:3, :3].T
    return T_body_cam[:3, 3], D


_CACHE = {}


def geometry(cam, T_cam_body):
    key = (cam.name, cam.W, T_cam_body.tobytes())
    if key not in _CACHE:
        O, D = camera_rays(cam, T_cam_body)
        t, ids, n = raycast(O, D)
        P = O + D * np.where(np.isfinite(t), t, 0)[:, None]
        # глибина вздовж оптичної осі (z у рамці камери)
        z = transform(T_cam_body, P)[:, 2]
        z[~np.isfinite(t)] = np.inf
        _CACHE[key] = dict(t=t, ids=ids, n=n, P=P, z=z, D=D)
    return _CACHE[key]


def material_map(ids, P):
    mats = np.empty(len(ids), dtype=object)
    for k, (_, _, _, mat) in enumerate(SCENE):
        mats[ids == k] = mat
    g = ids == -1
    mats[g] = np.where(np.abs(P[g, 1]) < ASPHALT_Y, 'асфальт', 'трава')
    return mats


# =====================================================================
# Рендер RGB і LWIR
# =====================================================================
def render_rgb(h=13.0, smoke=0.0, cam=RGB_CAM, T_cam_body=T_RGB_BODY, seed=0):
    """RGB 0..1, лінійна модель освітлення + шум сенсора. Ніч — темно і шумно; smoke — серпанок з дальністю."""
    rng = np.random.default_rng(seed)
    g = geometry(cam, T_cam_body)
    ids, n, P, t = g['ids'], g['n'], g['P'], g['t']
    mats = material_map(ids, P)
    alb = np.zeros((len(ids), 3))
    for name, m in MATERIALS.items():
        alb[mats == name] = m['rgb']
    sun_dir = np.array([0.3, -0.5, 0.81]); sun_dir /= np.linalg.norm(sun_dir)
    E_sun = sun(h); E_amb = 0.25 * np.clip(sun(h) * 2.5, 0, 1) + 0.004
    lam = np.clip(n @ sun_dir, 0, None)
    img = alb * (E_sun * lam + E_amb)[:, None]
    sky = ids == -2
    sky_rgb = np.array([0.45, 0.62, 0.85]) * (0.01 + 0.99 * np.clip(sun(h) * 2.5, 0, 1)) + 0.002
    img[sky] = sky_rgb
    if smoke > 0:
        a = np.exp(-smoke * np.where(np.isfinite(t), t, 400) / 50.0)[:, None]
        img = img * a + (1 - a) * np.array([0.55, 0.55, 0.55]) * (E_amb + 0.6 * E_sun)
    img = img.reshape(cam.H, cam.W, 3)
    # експозиція: автоекспозиція до середнього 0.35, але з обмеженням підсилення (ніч — шум)
    gain = np.clip(0.35 / max(img.mean(), 1e-6), 0.5, 8.0)
    photons = img * gain * 4000.0
    noisy = rng.poisson(np.clip(photons, 0, None)) + rng.normal(0, 25.0, photons.shape)
    out = np.clip(noisy / 4000.0, 0, 1)
    return out ** (1 / 2.2)


def thermal_radiance(h=13.0, cam=T_CAM, T_cam_body=T_T_BODY, humidity=0.5, smoke=0.0):
    """Ідеальна (без сенсора) смугова яскравість LWIR у кожному пікселі + GT температура поверхні."""
    g = geometry(cam, T_cam_body)
    ids, n, P, t = g['ids'], g['n'], g['P'], g['t']
    mats = material_map(ids, P)
    Ts = np.full(len(ids), sky_temp(h)); eps = np.ones(len(ids))
    up = n[:, 2] > 0.5
    for name, m in MATERIALS.items():
        sel = mats == name
        if sel.any():
            Ts[sel & up] = surface_temp(name, h, up=True)
            Ts[sel & ~up] = surface_temp(name, h, up=False)
            eps[sel] = m['eps']
    # відбита температура: горизонталь «дивиться» в небо, вертикаль — наполовину в небо, наполовину на землю
    T_refl = np.where(up, sky_temp(h), 0.5 * sky_temp(h) + 0.5 * air_temp(h))
    L = eps * band_radiance(Ts) + (1 - eps) * band_radiance(T_refl)
    sky = ids == -2
    L[sky] = band_radiance(np.full(sky.sum(), sky_temp(h)))
    # атмосфера: τ = exp(−α R), α — 1/км, росте з вологістю; дим у LWIR майже прозорий
    alpha = (0.05 + 0.25 * humidity) + 0.3 * smoke
    R = np.where(np.isfinite(t), t, 0.0)
    tau = np.exp(-alpha * R / 1000.0)
    L = np.where(sky, L, tau * L + (1 - tau) * band_radiance(air_temp(h)))
    return L.reshape(cam.H, cam.W), Ts.reshape(cam.H, cam.W), eps.reshape(cam.H, cam.W)


# ---------- мікроболометр ----------
class Core:
    """Модель неохолоджуваного ядра: 14 біт, лінійний відгук, FPN, NETD, дрейф offset між FFC."""
    BITS = 14

    def __init__(self, cam=T_CAM, netd_mK=50.0, T_lo=233.15, T_hi=343.15, seed=1):
        rng = np.random.default_rng(seed)
        self.cam = cam
        self.L_lo, self.L_hi = band_radiance(T_lo), band_radiance(T_hi)
        self.G = 15000.0 / (self.L_hi - self.L_lo)               # counts на Вт/(м²·ср)
        self.O = 800.0 - self.G * self.L_lo
        self.sigma = self.G * dL_dT(300.0) * netd_mK / 1000.0     # σ шуму в counts
        H, W = cam.H, cam.W
        # FPN до NUC: попіксельні gain/offset і стовпчикові смуги
        self.g_px = 1 + rng.normal(0, 0.03, (H, W))
        self.o_px = rng.normal(0, 120.0, (H, W)) + rng.normal(0, 60.0, (1, W))
        self.dead = rng.random((H, W)) < 2e-4
        # мертві пікселі «залипають»: кожен — назавжди гарячий або холодний (окремий генератор, щоб не зсувати шум)
        self.dead_val = np.where(np.random.default_rng(seed + 100).random((H, W)) < 0.5, 2 ** self.BITS - 1, 0)
        self.drift_pattern = (np.linspace(-1, 1, W)[None, :] ** 2 + np.linspace(-1, 1, H)[:, None] ** 2)  # віньєт-подібно
        self.rng = rng

    def counts_per_K(self, T=300.0):
        return self.G * dL_dT(T)

    def raw(self, L, nuc=True, drift=0.0, tau_blur_px=0.0, psf_sigma=0.6):
        """L → uint16 counts (14 біт). drift — дрейф offset (counts) від нагріву корпусу після останнього FFC.
        nuc=False — «сирий» кадр до NUC: FPN і залиплі биті пікселі; nuc=True — залишок FPN 5 % і биті пікселі замінено."""
        img = self.G * L + self.O
        if psf_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), psf_sigma)
        if tau_blur_px > 0:                                   # шлейф від теплового часу τ при обертанні
            k = int(np.ceil(tau_blur_px * 4)) | 1
            x = np.arange(k) - k // 2
            ker = np.where(x <= 0, np.exp(x / max(tau_blur_px, 1e-3)), 0.0)[None, :]
            ker /= ker.sum()
            img = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REPLICATE)
        if not nuc:
            img = img * self.g_px + self.o_px
        else:                                                 # залишок після NUC: 5 % від FPN
            img = img * (1 + (self.g_px - 1) * 0.05) + self.o_px * 0.05
        img = img + drift * (0.4 + self.drift_pattern)
        img = img + self.rng.normal(0, self.sigma, img.shape)
        img[self.dead] = self.dead_val[self.dead]
        if nuc:                                               # карта битих пікселів — частина NUC: заміна медіаною сусідів
            med = cv2.medianBlur(np.clip(img, 0, 2 ** self.BITS - 1).astype(np.float32), 3)
            img[self.dead] = med[self.dead]
        return np.clip(np.round(img), 0, 2 ** self.BITS - 1).astype(np.uint16)

    def to_kelvin(self, raw):
        """Радіометрична конверсія (ідеальна калібровка): counts → яскравісна температура."""
        return T_from_radiance((raw.astype(float) - self.O) / self.G)


def render_thermal(h=13.0, core=None, humidity=0.5, smoke=0.0, **kw):
    core = core or Core()
    L, Ts, eps = thermal_radiance(h, humidity=humidity, smoke=smoke)
    return core.raw(L, **kw), Ts, eps


# ---------- AGC: 14 біт → 8 біт ----------
def agc(raw, mode='percentile', lo_hi=None, p=(0.5, 99.9), clip_limit=2.0):
    x = raw.astype(np.float32)
    if mode == 'minmax':
        lo, hi = x.min(), x.max()
    elif mode == 'percentile':
        lo, hi = np.percentile(x, p)
    elif mode == 'fixed':
        lo, hi = lo_hi
    elif mode == 'histeq':
        flat = np.sort(x.ravel())
        cdf = np.searchsorted(flat, x.ravel(), side='right') / flat.size
        return (cdf.reshape(x.shape) * 255).astype(np.uint8)
    elif mode == 'clahe':
        lo, hi = np.percentile(x, (0.1, 99.9))
        x16 = np.clip((x - lo) / max(hi - lo, 1) * 65535, 0, 65535).astype(np.uint16)
        cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(x16)
        return (cl >> 8).astype(np.uint8)
    else:
        raise ValueError(mode)
    return np.clip((x - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


# ---------- FFC-потік ----------
def thermal_stream(n=450, fps=30.0, ffc_every_s=6.0, ffc_len_s=0.4, drift_rate=15.0, core=None, h=21.0, seed=3):
    """Потік (t, raw, telem): між FFC offset повзе (drift_rate counts/с), під час FFC ядро віддає замерзлий кадр
    біт-у-біт, після FFC offset обнуляється (стрибок). Сцена статична, щоб бачити саме ефекти сенсора."""
    core = core or Core(seed=seed)
    L, _, _ = thermal_radiance(h)
    out = []; last = None; t_ffc = 0.0; frozen_until = -1.0
    for i in range(n):
        t = i / fps
        if t >= t_ffc + ffc_every_s:
            t_ffc = t; frozen_until = t + ffc_len_s
        in_ffc = t < frozen_until
        if in_ffc and last is not None:
            raw = last.copy()
        else:
            raw = core.raw(L, drift=drift_rate * (t - t_ffc - (ffc_len_s if t_ffc > 0 else 0)))
        out.append((t, raw, dict(ffc_in_progress=bool(in_ffc), t_since_ffc=t - t_ffc)))
        last = raw
    return out


# =====================================================================
# Реєстрація
# =====================================================================
def H_inf(Kd=RGB_CAM.K, Ks=T_CAM.K, T_d_s=T_RGB_T):
    """Гомографія «на нескінченності»: пікселі src (T) → dst (RGB). Ігнорує трансляцію."""
    return Kd @ T_d_s[:3, :3] @ np.linalg.inv(Ks)


def H_plane(Z0, Kd=RGB_CAM.K, Ks=T_CAM.K, T_d_s=T_RGB_T, n_s=np.array([0, 0, 1.0])):
    """Гомографія, індукована площиною n·X = Z0 у рамці src (фронтальна площина на глибині Z0):
    H = Kd (R + t nᵀ / Z0) Ks⁻¹. Точна лише для точок на цій площині."""
    R, t = T_d_s[:3, :3], T_d_s[:3, 3]
    return Kd @ (R + np.outer(t, n_s) / Z0) @ np.linalg.inv(Ks)


def warp_t_to_rgb(t_img, H, interp=cv2.INTER_LINEAR):
    out = cv2.warpPerspective(t_img.astype(np.float32), H, (RGB_CAM.W, RGB_CAM.H), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid = np.isfinite(out)
    return np.nan_to_num(out, nan=0.0), valid


def warp_t_to_rgb_depth(t_img, depth_rgb, T_t_rgb=T_T_RGB):
    """Точна реєстрація з глибиною в RGB-кадрі (LiDAR / стерео): кожен RGB-піксель → 3D → у кадр T."""
    Hh, Ww = depth_rgb.shape
    uu, vv = np.meshgrid(np.arange(Ww), np.arange(Hh))
    z = np.where(np.isfinite(depth_rgb), depth_rgb, 1e4)
    X = np.stack([(uu - RGB_CAM.K[0, 2]) / RGB_CAM.K[0, 0] * z, (vv - RGB_CAM.K[1, 2]) / RGB_CAM.K[1, 1] * z, z], -1)
    Xt = transform(T_t_rgb, X.reshape(-1, 3))
    u = T_CAM.K[0, 0] * Xt[:, 0] / Xt[:, 2] + T_CAM.K[0, 2]
    v = T_CAM.K[1, 1] * Xt[:, 1] / Xt[:, 2] + T_CAM.K[1, 2]
    mapx = u.reshape(Hh, Ww).astype(np.float32); mapy = v.reshape(Hh, Ww).astype(np.float32)
    out = cv2.remap(t_img.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=np.nan)
    valid = np.isfinite(out)
    return np.nan_to_num(out, nan=0.0), valid


def gt_correspondence(depth_rgb, T_t_rgb=T_T_RGB):
    """GT: де в кадрі T лежить кожен RGB-піксель (для вимірювання залишку реєстрації)."""
    Hh, Ww = depth_rgb.shape
    uu, vv = np.meshgrid(np.arange(Ww), np.arange(Hh))
    z = np.where(np.isfinite(depth_rgb), depth_rgb, 1e6)
    X = np.stack([(uu - RGB_CAM.K[0, 2]) / RGB_CAM.K[0, 0] * z, (vv - RGB_CAM.K[1, 2]) / RGB_CAM.K[1, 1] * z, z], -1)
    Xt = transform(T_t_rgb, X.reshape(-1, 3))
    return (T_CAM.K[0, 0] * Xt[:, 0] / Xt[:, 2] + T_CAM.K[0, 2]).reshape(Hh, Ww), \
           (T_CAM.K[1, 1] * Xt[:, 1] / Xt[:, 2] + T_CAM.K[1, 2]).reshape(Hh, Ww)


def registration_residual(H, depth_rgb):
    """Залишок гомографії H (T → RGB) у px RGB-сітки для кожного RGB-пікселя зі скінченною глибиною."""
    ut, vt = gt_correspondence(depth_rgb)
    ok = np.isfinite(depth_rgb)
    pts = np.stack([ut[ok], vt[ok], np.ones(ok.sum())], 1) @ H.T
    up, vp = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
    uu, vv = np.meshgrid(np.arange(depth_rgb.shape[1]), np.arange(depth_rgb.shape[0]))
    res = np.full(depth_rgb.shape, np.nan)
    res[ok] = np.hypot(up - uu[ok], vp - vv[ok])
    return res


def parallax_px(f_px, B_m, Z_m):
    return f_px * B_m / np.asarray(Z_m, float)


# ---------- калібрувальна мішень ----------
def target_points(rows=5, cols=7, pitch=0.08):
    """Центри кіл у рамці мішені (x праворуч, y вниз, z = 0)."""
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([gx.ravel() * pitch, gy.ravel() * pitch, np.zeros(rows * cols)], 1).astype(np.float32)


def render_target(cam, T_cam_target, modality, rows=5, cols=7, pitch=0.08, radius=0.022, h=21.0,
                  T_heat=318.0, rng=None):
    """Плоска мішень (алюміній, низька ε) з колами чорної фарби (висока ε), підігрітими до T_heat.
    RGB: чорні кола на світлому металі. LWIR: кола гарячі й «яскраві», алюміній відбиває холодне небо."""
    rng = rng or np.random.default_rng(0)
    uu, vv = np.meshgrid(np.arange(cam.W), np.arange(cam.H))
    d = np.stack([(uu - cam.K[0, 2]) / cam.K[0, 0], (vv - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(uu, float)], -1)
    T_tc = inv(T_cam_target)
    Rtc, ttc = T_tc[:3, :3], T_tc[:3, 3]
    O = ttc; D = d.reshape(-1, 3) @ Rtc.T                       # промені в рамці мішені
    s = -O[2] / D[:, 2]
    X = O + D * s[:, None]
    w, hgt = (cols + 1) * pitch, (rows + 1) * pitch
    on = (s > 0) & (X[:, 0] > -pitch) & (X[:, 0] < w - pitch) & (X[:, 1] > -pitch) & (X[:, 1] < hgt - pitch)
    cx = np.round(X[:, 0] / pitch).clip(0, cols - 1); cy = np.round(X[:, 1] / pitch).clip(0, rows - 1)
    circ = on & (np.hypot(X[:, 0] - cx * pitch, X[:, 1] - cy * pitch) < radius)
    if modality == 'rgb':
        bg = 0.25
        img = np.full(len(s), bg); img[on] = 0.80; img[circ] = 0.06
        img = img.reshape(cam.H, cam.W)
        img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 0.01, img.shape)
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    # LWIR: яскравісна температура
    T_bg = air_temp(h)
    Lm = apparent_radiance(air_temp(h), MATERIALS['метал']['eps'], sky_temp(h) * 0.5 + air_temp(h) * 0.5)
    L = np.full(len(s), band_radiance(T_bg)); L[on] = Lm; L[circ] = apparent_radiance(T_heat, 0.95, air_temp(h))
    img = T_from_radiance(L.reshape(cam.H, cam.W))
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 0.05, img.shape)     # LWIR м'якший
    lo, hi = np.percentile(img, (1, 99.8))
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def random_target_pose(rng, dist=(1.6, 3.2), tilt=25.0):
    """Поза мішені перед RGB-камерою: T_rgb_target. tilt — максимальний нахил навколо x і y, градуси
    (поворот навколо осі камери — 0.3·tilt). tilt ≈ 0 — фронтальні пози."""
    Z = rng.uniform(*dist)
    ang = rng.uniform(-tilt, tilt, 3) * np.array([1, 1, 0.3])
    R = Rot.from_euler('xyz', ang, degrees=True).as_matrix()
    c = np.array([rng.uniform(-0.25, 0.25) * Z, rng.uniform(-0.12, 0.12) * Z, Z])
    off = R @ np.array([3 * 0.08, 2 * 0.08, 0])          # центр мішені → у c
    return se3(R, c - off)


def find_circles(img, rows=5, cols=7, dark=True):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor = True; p.blobColor = 0 if dark else 255
    p.filterByArea = True; p.minArea = 15; p.maxArea = 20000
    p.filterByCircularity = False; p.filterByConvexity = False; p.filterByInertia = False
    det = cv2.SimpleBlobDetector_create(p)
    ok, c = cv2.findCirclesGrid(img, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=det)
    return ok, c


# ---------- крос-модальна схожість ----------
def mutual_information(a, b, bins=32, mask=None):
    a = a.ravel(); b = b.ravel()
    if mask is not None:
        a, b = a[mask.ravel()], b[mask.ravel()]
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / h.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def ncc(a, b, mask=None):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    if mask is not None:
        a, b = a[mask.ravel()], b[mask.ravel()]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img.astype(np.float32), M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


# ---------- час ----------
def sync_shift_px(f_px, omega_dps, dt_s):
    """Зсув у кадрі від розсинхрону Δt при обертанні ω: f · ω · Δt (рад)."""
    return f_px * np.radians(omega_dps) * dt_s


def pair_nearest(t_a, t_b):
    """Для кожного кадру a — найближчий за часом кадр b. Повертає індекси і |Δt|."""
    t_b = np.asarray(t_b)
    j = np.clip(np.searchsorted(t_b, t_a), 1, len(t_b) - 1)
    j = np.where(np.abs(t_b[j - 1] - t_a) < np.abs(t_b[j] - t_a), j - 1, j)
    return j, np.abs(t_b[j] - t_a)


def rgb_depth():
    return geometry(RGB_CAM, T_RGB_BODY)['z'].reshape(RGB_CAM.H, RGB_CAM.W)


def rgb_ids():
    return geometry(RGB_CAM, T_RGB_BODY)['ids'].reshape(RGB_CAM.H, RGB_CAM.W)


def t_ids():
    return geometry(T_CAM, T_T_BODY)['ids'].reshape(T_CAM.H, T_CAM.W)


def object_px(name, cam='rgb'):
    """Маска пікселів об'єкта сцени за назвою."""
    k = [s[0] for s in SCENE].index(name)
    return (rgb_ids() if cam == 'rgb' else t_ids()) == k
