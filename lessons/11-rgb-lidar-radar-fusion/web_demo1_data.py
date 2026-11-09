"""Дані для веб-сторінки lesson11_demo1_extrinsics.html: KITTI object 000012 (кадр, хмара, K, T, опорні точки)
і криві вирівнювання ребер на перших 40 кадрах KITTI (geom11.edge_score). Monte Carlo слайда 26 сторінка рахує
сама — йому потрібні лише K і T.

Запуск з кореня репозиторію (потрібні дані KITTI object: python tools/fetch_kitti.py --object):
    python lessons/11-rgb-lidar-radar-fusion/web_demo1_data.py
Скрипт переписує JSON між /*DATA-BEGIN*/ і /*DATA-END*/ у HTML — решту сторінки не чіпає.
"""
import base64, itertools, json, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geom11 as G, kitti11 as K11

HTML = HERE / 'lesson11_demo1_extrinsics.html'
FRAME = 12                     # світлофор зі стовпом на ≈ 7 м і стовпи / будинки на 30–50 м
ANG, SHIFT = 2.0, 0.3          # межі слайдерів сторінки: ±2° і ±0.3 м
MARGIN = 20                    # px запасу поза кадром — між «кутами» перебору проєкція майже лінійна
EDGE_SWEEPS = {'yaw, °': ('yaw', np.linspace(-2, 2, 17)), 'pitch, °': ('pitch', np.linspace(-2, 2, 17)),
               'зсув x, м': ('tx', np.linspace(-0.4, 0.4, 17))}


def ref_points(xyz, T, K, W, H):
    """Дві опорні точки для звірки зсуву з прогнозом: розрив глибини вище сенсора на 5–10 м і на 25–50 м,
    найближчий до центру кадру по горизонталі."""
    e, _ = G.depth_edges(xyz)
    u, v, z, m = G.project(xyz, T, K, W, H)
    refs = {}
    for name, (lo, hi) in (('близько', (5, 10)), ('далеко', (25, 50))):
        c = np.nonzero(e & m & (xyz[:, 2] > 0.3) & (z > lo) & (z < hi))[0]
        if len(c):
            i = c[np.argmin(np.abs(u[c] - K[0, 2]))]
            refs[f'{name}, {z[i]:.0f} м'] = xyz[i]
    return refs


def demo1_frame(idx=FRAME):
    assert K11.available(idx), 'потрібні дані KITTI object: python tools/fetch_kitti.py --object'
    xyz, img = K11.points_image(idx)
    K, T = K11.cam(idx)
    H, W = img.shape[:2]
    return dict(xyz=xyz, img=img, K=K, T=T, W=W, H=H, source=f'KITTI object {K11.stem(idx)}',
                refs=ref_points(xyz, T, K, W, H))


def keep_mask(xyz, T0, K, W, H):
    """Точки, які можуть потрапити в кадр за будь-яких помилок у межах слайдерів і за обох багів зі сторінки.
    Перебираємо кожен параметр у {−max, 0, +max} (3⁶ = 729 комбінацій) для трьох варіантів T: правильна,
    без перестановки осей і T замість inv(T)."""
    keep = np.zeros(len(xyz), bool)
    T_noperm = G.se3(G.R_AXES.T @ T0[:3, :3], T0[:3, 3])
    for yaw, pitch, roll in itertools.product((-ANG, 0, ANG), repeat=3):
        for dt in itertools.product((-SHIFT, 0, SHIFT), repeat=3):
            Tp = G.perturb(T0, yaw, pitch, roll, dt)
            for T in (Tp, G.perturb(T_noperm, yaw, pitch, roll, dt), G.inv(Tp)):
                u, v, z, _ = G.project(xyz, T, K, W, H)
                keep |= (z > 0.5) & (u > -MARGIN) & (u < W + MARGIN) & (v > -MARGIN) & (v < H + MARGIN)
    return keep


def edge_frames(n=40):
    """Для кожного з перших n кадрів KITTI — розриви глибини з вагами, карта ребер, T і K."""
    out = []
    for i in range(n):
        if K11.available(i):
            xyz, img = K11.points_image(i)
            K, T = K11.cam(i)
            e, w = G.depth_edges(xyz)
            out.append(dict(stem=K11.stem(i), xyz=xyz[e], w=w[e], E=G.edge_map(img), T=T, K=K))
    return out


def edge_curves(n=40):
    """Криві вирівнювання ребер: для кожного параметра з EDGE_SWEEPS — метрика кожного з n кадрів на сітці штучних
    помилок і два числа під графіком: де максимум середнього і в якій частці кадрів максимум у межах кроку від 0."""
    frames = edge_frames(n)
    assert len(frames) > 1, 'потрібні кадри KITTI object'
    out = []
    for label, (key, vals) in EDGE_SWEEPS.items():
        pert = (lambda T, a: G.perturb(T, dt=(a, 0, 0))) if key == 'tx' else (lambda T, a, k=key: G.perturb(T, **{k: a}))
        curves = np.array([[G.edge_score(fr['xyz'], fr['w'], fr['E'], pert(fr['T'], a), fr['K']) for a in vals]
                           for fr in frames])
        best = vals[np.nanargmax(curves, 1)]
        stat = (float(vals[np.nanargmax(np.nanmean(curves, 0))]), float(np.mean(np.abs(best) <= (vals[1] - vals[0]))))
        print(f'{label:10s}: максимум середнього — при {stat[0]:+.2f}; у {stat[1]:.0%} кадрів максимум у межах одного кроку від 0')
        out.append(dict(label=label, key=key, vals=np.round(vals, 4).tolist(), curves=np.round(curves, 5).tolist(),
                        bestMean=stat[0], share=stat[1]))
    return dict(stems=[fr['stem'] for fr in frames], sweeps=out)


def main():
    F = demo1_frame()
    xyz, K, T, W, H = F['xyz'], F['K'], F['T'], F['W'], F['H']
    m = keep_mask(xyz, T, K, W, H)
    q = xyz[m].astype('<f4')                              # Float32, як у .bin KITTI: без втрат
    ok, jpg = cv2.imencode('.jpg', cv2.cvtColor((F['img'] * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = dict(
        source=F['source'], W=W, H=H,
        K=np.round(K, 4).tolist(), T=np.round(T, 8).tolist(),                 # T_cam2_velo, як у kitti11.cam
        refs=[dict(name=k, xyz=np.round(v, 4).tolist()) for k, v in F['refs'].items()],
        nAll=len(xyz), n=int(m.sum()),
        pts=base64.b64encode(q.tobytes()).decode(),                          # x, y, z підряд, м, Float32 little-endian
        jpg='data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode(),
        edges=edge_curves(40),
    )
    s = HTML.read_text(encoding='utf-8')
    a, b = s.index('/*DATA-BEGIN*/') + len('/*DATA-BEGIN*/'), s.index('/*DATA-END*/')
    with open(HTML, 'w', encoding='utf-8', newline='\n') as fh:      # LF на будь-якій ОС: повторний запуск дає той самий файл
        fh.write(s[:a] + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + s[b:])
    print(f"{F['source']}: лишили {m.sum()} з {len(xyz)} точок, JPEG {len(jpg) / 1e3:.0f} КБ, "
          f'HTML {HTML.stat().st_size / 1e6:.2f} МБ')


if __name__ == '__main__':
    main()
