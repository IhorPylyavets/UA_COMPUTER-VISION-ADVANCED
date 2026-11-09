"""Графіки й інтерактивні панелі заняття 11: demo 2 (час і deskew) і demo 3 (дальність LiDAR vs моно vs GT).

Геометрія — у geom11.py, дані KITTI object — у kitti11.py, синтетична сцена — у sim11.py. Обидва демо працюють
на KITTI object, якщо дані є (python tools/fetch_kitti.py --object), інакше — на синтетиці з відомим T_cam_lidar.
У demo 2 рух змодельований поверх справжньої хмари: у KITTI object немає поз і міток часу точок.
"""
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import ipywidgets as W

import geom11 as G, kitti11 as K11, sim11 as S

CAM = S.KITTI_CAM
# Okabe–Ito: розрізняються при дейтеранопії й протанопії і в чорно-білому друку
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
DEPTH_NORM = LogNorm(3, 60)                 # глибина, м: та сама log-шкала, що на слайді «Проєкція · результат»
DEPTH_TICKS = (3, 5, 10, 20, 40, 60)
OUTLINE = [pe.withStroke(linewidth=3, foreground='k')]   # білі підписи читаються і на небі, і на асфальті
SLIDER = dict(continuous_update=False, style={'description_width': '9em'}, layout=W.Layout(width='32em'))


def lecture_style():
    """Стиль для проєктора: більший шрифт, товстіші лінії, палітра, безпечна для дальтоніків."""
    plt.rcParams.update({'figure.dpi': 90, 'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
                         'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'lines.linewidth': 2,
                         'axes.prop_cycle': plt.cycler(color=OKABE_ITO)})


@lru_cache(maxsize=1)
def scene_image():
    return S.render_rgb()


def overlay(ax, img, u, v, z, s=2.0, title=None, colorbar=True):
    """Точки LiDAR на кадрі, колір — глибина в log-шкалі, з колорбаром у метрах."""
    ax.imshow(img)
    sc = ax.scatter(u, v, c=z, s=s, cmap='turbo', norm=DEPTH_NORM, lw=0)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.set_axis_off()
    if title:
        ax.set_title(title, loc='left')
    if colorbar:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, ticks=DEPTH_TICKS)
        cb.ax.set_yticklabels([str(t) for t in DEPTH_TICKS]); cb.ax.minorticks_off(); cb.set_label('глибина, м')
    return sc


# =====================================================================
# DEMO 2 — час і deskew
# =====================================================================
DEMO2_FRAME = 12            # той самий кадр, що на сторінці demo 1: стовп світлофора на ≈ 7 м, стовпи й будинки за ним
SEAM_FRAME = 52             # стіна у формі «V» прямо за машиною перетинає шов оберту
T_FRAME = 0.05              # мітка кадру, яку вважаємо правильною: камеру тригерить LiDAR, коли дивиться вперед


def demo2_scan(v, w_deg):
    """Скан для власного deskew: KITTI зі змодельованим рухом або синтетична сцена (світ — сенсор у t = 50 мс)."""
    if K11.available(DEMO2_FRAME):
        return K11.motion_scan(v, w_deg, DEMO2_FRAME)
    return S.lidar_scan(ego_v=v, ego_w_deg=w_deg)


def time_panel(static=None):
    """Розсинхрон і deskew на кадрі: кадр камери знято в T_FRAME + Δt, а ми вважаємо, що в T_FRAME."""
    cache = {}
    use_kitti = K11.available(DEMO2_FRAME)
    if use_kitti:
        K, T = K11.cam(DEMO2_FRAME)
        source = f'KITTI object {K11.stem(DEMO2_FRAME)}, рух змодельований'
    else:
        K, T, source = CAM.K, S.T_CAM_LIDAR, 'синтетична сцена'

    def scan_and_image(v, yaw_rate, dt_ms):
        t_true = T_FRAME + dt_ms / 1000            # кадр насправді знятий пізніше
        if use_kitti:
            # справжній кадр KITTI не перемалюєш, тож світ прив'язуємо до моменту кадру: у t_true хмара лягає ідеально
            key = (round(v, 2), round(yaw_rate, 1), round(dt_ms, 1))
            if key not in cache:
                cache[key] = K11.motion_scan(v, yaw_rate, DEMO2_FRAME, t_world=t_true)
            return cache[key], cache[key]['img']
        key = (round(v, 2), round(yaw_rate, 1))
        if key not in cache:
            cache[key] = S.lidar_scan(ego_v=v, ego_w_deg=yaw_rate)
        R, p = G.pose(t_true, v, yaw_rate)
        return cache[key], S.render_rgb(T_cam_lidar=S.T_CAM_LIDAR @ G.inv(G.se3(R, p)))[0]

    def show(v=15.0, yaw_rate=90.0, dt_ms=10.0, deskew=True):
        scan, img = scan_and_image(v, yaw_rate, dt_ms)
        H, W_ = img.shape[:2]
        raw = G.project(scan['xyz'], T, K, W_, H)
        fixed = G.project(G.to_frame_at(scan, T_FRAME, v, yaw_rate), T, K, W_, H)
        u, vv, z, m = fixed if deskew else raw
        fig, ax = plt.subplots(figsize=(13, 4.6))
        overlay(ax, img, u[m], vv[m], z[m], title=f'{source} · v = {v:.0f} м/с, рискання {yaw_rate:.0f}°/с, '
                                                  f'розсинхрон {dt_ms:.0f} мс, deskew: {"так" if deskew else "ні"}')
        plt.show()
        both = raw[3] & fixed[3]
        du = np.abs(raw[0][both] - fixed[0][both])
        edge = np.abs(raw[0][both] - K[0, 2]) > 0.35 * W_
        print(f'прогноз зсуву від обертання f·ω·Δt = {K[0, 0] * np.radians(yaw_rate) * dt_ms / 1000:.1f} px — однаковий на '
              f'будь-якій дальності; поступальна частина v·Δt дає зсув, що спадає з дальністю')
        print(f'deskew зсуває точки в кадрі: медіана {np.median(du):.1f} px, на краях кадру '
              f'{np.median(du[edge]) if edge.any() else 0:.1f} px (в центрі майже 0: камеру тригерить LiDAR, коли дивиться '
              f'вперед, як у KITTI)')

    if static is not None:
        return show(**static)
    return W.interact(show, v=W.FloatSlider(min=0, max=30, step=1, value=15, description='швидкість, м/с', **SLIDER),
                      yaw_rate=W.FloatSlider(min=0, max=300, step=10, value=90, description='рискання, °/с', **SLIDER),
                      dt_ms=W.FloatSlider(min=0, max=50, step=1, value=10, description='розсинхрон Δt, мс', **SLIDER),
                      deskew=W.Checkbox(value=True, description='deskew', indent=False))


def seam_plot(v=20.0):
    """Вид згори на те, що позаду машини: «шов» оберту (стик початку й кінця, азимут ±180°) до і після deskew."""
    if K11.available(SEAM_FRAME):
        scan = K11.motion_scan(v, 0.0, SEAM_FRAME)
        title = f'KITTI object {K11.stem(SEAM_FRAME)}, рух {v:.0f} м/с змодельований'
        sel_fn = lambda X: (X[:, 2] > -1.3) & (X[:, 0] < -2) & (np.abs(X[:, 1]) < 15) & (X[:, 0] > -30)
        lims = ((15, -15), (-30, -2))
    else:
        scan = S.lidar_scan(ego_v=v, scene=S.ROOM)
        title = 'синтетичний двір, задня стіна на x = −30 м'
        sel_fn = lambda X: (X[:, 2] > -1.2) & (X[:, 2] < 1.0) & (X[:, 0] < -25)
        lims = ((12, -12), (-33, -27))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, X, lab in ((axs[0], scan['xyz'], 'без deskew'),
                       (axs[1], G.to_frame_at(scan, T_FRAME, v, 0.0), f'deskew до t = {T_FRAME * 1000:.0f} мс')):
        sel = sel_fn(X)
        sc = ax.scatter(X[sel, 1], X[sel, 0], c=scan['t'][sel] * 1000, s=1.5, cmap='coolwarm', vmin=0, vmax=1000 / G.ROT_HZ)
        ax.set_title(lab); ax.set_xlabel('y, м (ліворуч +)'); ax.grid(alpha=0.3)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect('equal', adjustable='box')
    axs[0].set_ylabel('x, м (позаду машини)')
    fig.colorbar(sc, ax=axs, fraction=0.025, label='час точки в оберті, мс')
    fig.suptitle(f'{title}: синій — початок оберту, червоний — кінець; шов — уздовж осі x позаду', fontsize=12, x=0.45)
    plt.show()


# =====================================================================
# DEMO 3 — дальність: LiDAR vs моно vs GT
# =====================================================================
def kitti_frame(idx):
    """Кадр KITTI для demo 3: хмара, кадр, P = P2 · R0_rect · Tr_velo_to_cam і об'єкти з GT видимої поверхні й статусом."""
    pts, img = K11.points_image(idx)
    P2, _, _ = K11.calib(idx)
    objs = [dict(cls=o['cls'], box=o['box'], d_gt=K11.visible_depth(o, P2), d_center=o['loc'][2],
                 H=K11.CLASS_H[o['cls']], status=K11.status(o)) for o in K11.labels(idx)]
    return dict(pts=pts, img=img, P=K11.P(idx), f=P2[0, 0], objs=objs, source=f'KITTI object {K11.stem(idx)}')


def zdepth_map(cam=CAM):
    """Глибина z (а не довжина променя) у кожному пікселі синтетичного кадру — щоб порівнювати з project()."""
    img, t, ids = scene_image()
    u, v = np.meshgrid(np.arange(cam.W) + 0.5, np.arange(cam.H) + 0.5)
    n = np.linalg.norm(np.stack([(u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(u)], -1),
                       axis=-1)
    return img, t / n, ids


def synthetic_frame(occl_thr=0.25):
    """Та сама логіка статусів, що й на KITTI: GT — медіана глибини видимої поверхні,
    оклюзія — частка пікселів бокса, зайнятих ближчим об'єктом."""
    img, zmap, ids = zdepth_map()
    scan = S.lidar_scan()
    K = CAM.K; P = np.hstack([K, np.zeros((3, 1))]) @ S.T_CAM_LIDAR
    objs = []
    for k, (name, lo, hi, _, _) in enumerate(S.SCENE):
        if name not in ('авто 1', 'авто 2', 'людина'):
            continue
        vv, uu = np.nonzero(ids == k)
        if len(uu) < 30:
            continue
        box = np.array([uu.min(), vv.min(), uu.max(), vv.max()], float)
        d_gt = float(np.median(zmap[ids == k]))                       # видима поверхня, як visible_depth на KITTI
        sub_i, sub_z = ids[vv.min():vv.max() + 1, uu.min():uu.max() + 1], zmap[vv.min():vv.max() + 1, uu.min():uu.max() + 1]
        occl = float(((sub_i != k) & (sub_z < d_gt - 0.5)).mean())    # бокс закритий ближчим об'єктом
        H_prior = 1.75 if name == 'людина' else 1.53                  # апріорний розмір класу, не справжній
        objs.append(dict(cls=name, box=box, d_gt=d_gt, H=H_prior, H_true=hi[2] - lo[2], occl=occl,
                         status='оклюзія' if occl > occl_thr else 'малий бокс' if box[3] - box[1] < 25 else 'ok'))
    return dict(pts=scan['xyz'], img=img, P=P, f=K[0, 0], objs=objs, source=K11.NO_KITTI)


def range_table(frame, range_fn=None):
    """Рядок на об'єкт: клас, GT, дальність LiDAR (range_fn, за замовчуванням geom11.lidar_range), моно f·H/h, точок, статус."""
    range_fn = range_fn or G.lidar_range
    H, W_ = frame['img'].shape[:2]
    u, v, z, m = G.project_P(frame['pts'], frame['P'], W_, H)
    rows = []
    for o in frame['objs']:
        d_l, n = range_fn(u[m], v[m], z[m], o['box'])
        d_mono = frame['f'] * o['H'] / (o['box'][3] - o['box'][1])
        status = o['status'] if o['status'] != 'ok' or d_l is not None else '< 5 точок'
        rows.append((o['cls'], o['d_gt'], d_l, d_mono, n, status))
    return rows, (u, v, z, m)


def range_report(frame, range_fn=None):
    rows, (u, v, z, m) = range_table(frame, range_fn)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    overlay(ax, frame['img'], u[m], v[m], z[m], s=1.5,
            title=f"{frame['source']} · суцільний бокс — ok, пунктир — відмова; номер — рядок таблиці")
    for k, (o, r) in enumerate(zip(frame['objs'], rows)):
        x0, y0, x1, y1 = o['box']
        ok = r[5] == 'ok'
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='w' if ok else OKABE_ITO[1],
                                   lw=1.5 if ok else 2.0, ls='-' if ok else '--'))
        ax.text(x0, y0 - 2, str(k), color='w', fontsize=10, va='bottom', bbox=dict(fc='k', alpha=0.7, pad=1))
    plt.show()
    print(f"{'#':>2s} {'клас':12s} {'статус':11s} {'GT, м':>7s} {'LiDAR':>7s} {'моно':>7s} {'точок':>6s} {'|ΔLiDAR|':>9s} {'|Δмоно|':>8s}")
    for k, (c, g, l, mo, n, st) in enumerate(rows):
        print(f"{k:2d} {c:12s} {st:11s} {g:7.2f} {('—' if l is None else f'{l:.2f}'):>7s} {mo:7.2f} {n:6d} "
              f"{('—' if l is None else f'{abs(l - g):.2f}'):>9s} {abs(mo - g):8.2f}")
    return rows


def points_vs_range(az=0.18, el=0.425, W_t=0.5, H_t=1.75):
    """Скільки точок HDL-64E на 10 Гц падає на людину з дальністю (слайд «Чесне порівняння»)."""
    Z = np.linspace(5, 100, 200)
    n = (W_t / Z / np.radians(az)) * (H_t / Z / np.radians(el))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    z5 = Z[np.argmax(n < 5)]
    ax.semilogy(Z, n, label='точок на цілі')
    ax.axhline(5, color=OKABE_ITO[3], ls='--', lw=1.5, label='поріг: 5 точок')
    ax.axvline(z5, color='0.4', ls=':', lw=1.5, label=f'≈ {z5:.0f} м')
    ax.set_xlabel('дальність, м'); ax.set_ylabel('точок на цілі'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f'людина {W_t} × {H_t} м, крок HDL-64E {az}° × {el}°')
    plt.show()
    return z5
