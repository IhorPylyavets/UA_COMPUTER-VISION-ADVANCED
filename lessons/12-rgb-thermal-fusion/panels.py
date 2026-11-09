"""Інтерактивні панелі заняття 12 (RGB + Thermal).

Усі демо працюють на синтетиці з відомим ground truth: sim12.py — радіометрія LWIR, мікроболометр і dual EO/IR-
установка; fuse12.py — процедурний RGB-T датасет і моделі early / mid / late fusion.
Кожна панель має параметр static=dict(...) — тоді малює один кадр без віджетів (для слайдів і nbconvert).
"""
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
import cv2

import sim12 as S

C_RGB, C_T, C_F, C_W = '#38BDF8', '#F97316', '#4FD1C5', '#9AA3AD'


def _interact(fn, static, **widgets):
    if static is not None:
        return fn(**static)
    return W.interact(fn, **widgets)


# =====================================================================
# DEMO 1 — фізика і сигнал
# =====================================================================
def planck_figure(ax=None):
    ax = ax or plt.subplots(figsize=(9, 3.8))[1]
    lam = np.linspace(0.5, 25, 800)
    for T, c in ((250, '#64748B'), (300, C_F), (350, '#F59E0B'), (600, C_T)):
        L = S.planck(lam * 1e-6, T) * 1e-6          # Вт/(м²·ср·мкм)
        ax.semilogy(lam, L, color=c, lw=2, label=f'{T} K, пік {S.WIEN_B / T * 1e6:.1f} мкм')
    for lo, hi, n, c in ((3, 5, 'MWIR', '#C084FC'), (8, 14, 'LWIR', C_F)):
        ax.axvspan(lo, hi, color=c, alpha=0.12); ax.text((lo + hi) / 2, 2e3, n, ha='center', color=c, fontsize=10)
    ax.set_ylim(1e-2, 1e4); ax.set_xlabel('довжина хвилі, мкм'); ax.set_ylabel('L_λ, Вт/(м²·ср·мкм)')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3, which='both')
    return ax


def diurnal_curves(hours=np.linspace(0, 24, 241)):
    out = {
        'людина': S.surface_temp('людина', hours),
        'трава': S.surface_temp('трава', hours, up=True),
        'асфальт (горизонт.)': S.surface_temp('асфальт', hours, up=True),
        'бетонні блоки (верт.)': S.surface_temp('бетон', hours, up=False),
        'повітря': S.air_temp(hours),
    }
    return hours, {k: v - 273.15 for k, v in out.items()}


def crossovers(h, a, b):
    """Моменти, коли криві a і b перетинаються (ΔT змінює знак)."""
    d = a - b
    i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    return h[i] - d[i] * (h[i + 1] - h[i]) / (d[i + 1] - d[i])


def diurnal_figure(ax=None):
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    h, c = diurnal_curves()
    cols = dict(zip(c, (C_T, '#22C55E', '#94A3B8', '#F59E0B', '#64748B')))
    for k, v in c.items():
        ax.plot(h, v, lw=2 if k != 'повітря' else 1, ls='-' if k != 'повітря' else '--', color=cols[k], label=k)
    for t in crossovers(h, c['бетонні блоки (верт.)'], c['трава']):
        ax.axvline(t, color='#F59E0B', alpha=0.5, lw=1); ax.text(t, 36, f'{t:.1f} год', color='#F59E0B', fontsize=8, ha='center')
    for t in crossovers(h, c['людина'], c['асфальт (горизонт.)']):
        ax.axvline(t, color=C_T, alpha=0.5, lw=1, ls=':')
    ax.set_xlabel('година доби'); ax.set_ylabel('T поверхні, °C'); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=8, ncol=3, loc='upper left'); ax.grid(alpha=0.3)
    return ax


def scene_panel(static=None):
    """Година доби → RGB і LWIR тієї самої сцени (кожна камера зі свого місця, різні FOV)."""
    core = S.Core()

    def show(hour=21.0, smoke=0.0, agc_mode='percentile'):
        rgb = S.render_rgb(hour, smoke=smoke)
        raw, Ts, eps = S.render_thermal(hour, core, smoke=smoke)
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
        ax[0].imshow(rgb); ax[0].set_title(f'RGB 1280×720, HFOV 60°   {hour:04.1f} год, дим {smoke:.1f}', loc='left', fontsize=9)
        ax[1].imshow(S.agc(raw, agc_mode), cmap='gray'); ax[1].set_title(f'LWIR 640×512, HFOV 45°, AGC = {agc_mode}', loc='left', fontsize=9)
        for a in ax:
            a.set_axis_off()
        plt.tight_layout(); plt.show()
        TK = core.to_kelvin(raw)
        for name in ('людина 18 м', 'лист металу', 'вікно', 'бетонні блоки'):
            m = S.object_px(name, 't')
            if m.any():
                print(f'{name:14s}: T поверхні {np.median(Ts[m]) - 273.15:6.1f} °C, ε = {np.median(eps[m]):.2f}, '
                      f'яскравісна T {np.median(TK[m]) - 273.15:6.1f} °C')
        print(f'повітря {S.air_temp(hour) - 273.15:.1f} °C, ясне небо {S.sky_temp(hour) - 273.15:.1f} °C')

    return _interact(show, static, hour=W.FloatSlider(min=0, max=24, step=0.5, value=21, continuous_update=False),
                     smoke=W.FloatSlider(min=0, max=3, step=0.25, value=0, continuous_update=False),
                     agc_mode=['percentile', 'minmax', 'histeq', 'clahe'])


def agc_hot_source(core=None, hour=21.0, hot_K=600.0):
    """Кадр без і з гарячим джерелом (вихлоп / ракетниця) у кутку: як AGC змінює рівень людини."""
    core = core or S.Core()
    L, _, _ = S.thermal_radiance(hour)
    raw0 = core.raw(L)
    L1 = L.copy(); L1[40:52, 560:572] = S.band_radiance(np.full((12, 12), hot_K))
    raw1 = core.raw(L1)
    lohi = np.percentile(raw0, (0.5, 99.9))
    m = S.object_px('людина 18 м', 't')
    rows = []
    for mode in ('minmax', 'percentile', 'histeq', 'fixed'):
        a = S.agc(raw0, mode, lo_hi=lohi); b = S.agc(raw1, mode, lo_hi=lohi)
        rows.append((mode, float(np.median(a[m])), float(np.median(b[m])), a, b))
    return raw0, raw1, rows


# =====================================================================
# DEMO 2 — реєстрація
# =====================================================================
def overlay(rgb, t_on_rgb, valid, alpha=0.5):
    t = t_on_rgb.copy(); t = (t - np.percentile(t[valid], 1)) / (np.ptp(np.percentile(t[valid], (1, 99))) + 1e-6)
    heat = plt.get_cmap('inferno')(np.clip(t, 0, 1))[..., :3]
    out = rgb.copy(); out[valid] = (1 - alpha) * rgb[valid] + alpha * heat[valid]
    return out


def edges_overlay(rgb, t_on_rgb, valid):
    """Червоні ребра T поверх сірого RGB: так найкраще видно зсув реєстрації."""
    g = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    t8 = np.clip(t_on_rgb, 0, 255).astype(np.uint8)
    e = cv2.dilate(cv2.Canny(t8, 40, 100), np.ones((2, 2), np.uint8)) > 0
    e &= valid
    out = np.repeat(g[..., None], 3, 2) / 255.0 * 0.8
    out[e] = (1.0, 0.25, 0.1)
    return out


def registration_panel(static=None, hour=9.0):
    rgb = S.render_rgb(hour); core = S.Core(); raw, _, _ = S.render_thermal(hour, core)
    t8 = S.agc(raw, 'percentile').astype(np.float32)
    D = S.rgb_depth()

    def show(method='H∞', Z0=18.0, crop='людина 8 м'):
        if method == 'H∞':
            H = S.H_inf(); tw, v = S.warp_t_to_rgb(t8, H); res = S.registration_residual(H, D)
        elif method == 'H площини Z0':
            H = S.H_plane(Z0); tw, v = S.warp_t_to_rgb(t8, H); res = S.registration_residual(H, D)
        else:
            tw, v = S.warp_t_to_rgb_depth(t8, D); res = np.zeros_like(D); res[~np.isfinite(D)] = np.nan
        boxes = {'людина 8 м': (330, 260, 180, 330), 'людина 45 м': (690, 330, 100, 80), 'стіна': (380, 120, 300, 200)}
        x0, y0, w, h = boxes[crop]
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1, 1.2]))
        ax[0].imshow(edges_overlay(rgb, tw, v)); ax[0].add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=C_F, lw=1.5))
        ax[0].set_title(f'ребра T (червоні) на RGB · {method}' + (f', Z0 = {Z0:.0f} м' if method == 'H площини Z0' else ''),
                        loc='left', fontsize=9)
        ax[1].imshow(edges_overlay(rgb, tw, v)[y0:y0 + h, x0:x0 + w]); ax[1].set_title(crop, loc='left', fontsize=9)
        im = ax[2].imshow(res, cmap='magma', vmin=0, vmax=8); plt.colorbar(im, ax=ax[2], fraction=0.035, label='px')
        ax[2].set_title('залишок реєстрації (GT), px RGB', loc='left', fontsize=9)
        for a in ax:
            a.set_axis_off()
        plt.tight_layout(); plt.show()
        for name in ('людина 8 м', 'людина 18 м', 'людина 45 м', 'дрон', 'стіна'):
            m = S.object_px(name) & np.isfinite(res)
            print(f'{name:12s} Z = {np.median(D[m]):5.1f} м: залишок {np.median(res[m]):5.2f} px, '
                  f'f·B/Z = {S.parallax_px(S.RGB_CAM.f, np.linalg.norm(S.BASELINE[:2]), np.median(D[m])):5.2f} px')

    return _interact(show, static, method=['H∞', 'H площини Z0', 'глибина (LiDAR / стерео)'],
                     Z0=W.FloatSlider(min=5, max=80, step=1, value=18, continuous_update=False),
                     crop=['людина 8 м', 'людина 45 м', 'стіна'])


def sync_panel(static=None):
    def show(omega_dps=30.0, dt_ms=16.7, f_px=1108.5, target_px=12):
        d = S.sync_shift_px(f_px, omega_dps, dt_ms / 1000)
        iou = max(target_px - d, 0) / (target_px + d)
        print(f'f·ω·Δt = {f_px:.0f} px × {np.radians(omega_dps):.3f} рад/с × {dt_ms / 1000:.4f} с = {d:.1f} px; '
              f'IoU боксу {target_px} px після такого зсуву = {iou:.2f}')
    return _interact(show, static, omega_dps=W.FloatSlider(min=0, max=300, step=5, value=30, continuous_update=False),
                     dt_ms=W.FloatSlider(min=0, max=50, step=0.5, value=16.7, continuous_update=False),
                     f_px=W.FloatSlider(min=400, max=4000, step=50, value=1108.5, continuous_update=False),
                     target_px=W.IntSlider(min=4, max=64, value=12, continuous_update=False))


# =====================================================================
# DEMO 4 — абляція зсуву
# =====================================================================
def shift_panel(models, data, static=None):
    import fuse12 as F

    def show(dx=0, dy=0):
        res, _ = F.evaluate(models, data, shift=(dx, dy))
        for k, v in res.items():
            print(f'{k:16s} AP all {v["all"]:.3f} · ' + ' · '.join(f'{F.label(c)} {v[c]:.3f}' for c in data.conds))
    return _interact(show, static, dx=W.IntSlider(min=-6, max=6, value=0, continuous_update=False),
                     dy=W.IntSlider(min=-6, max=6, value=0, continuous_update=False))
