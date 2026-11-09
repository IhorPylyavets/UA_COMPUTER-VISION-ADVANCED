#!/usr/bin/env python
"""
Реальні RGB + LWIR датасети для заняття 12 — в artifacts/rgbt/.

    python tools/fetch_rgbt.py                    # LLVIP + FLIR-aligned, підмножини для демо, ≈ 1,3 ГБ
    python tools/fetch_rgbt.py --full             # повні LLVIP (15 488 пар) і FLIR-aligned (5 142 пари), ≈ 5 ГБ
    python tools/fetch_rgbt.py --only llvip       # лише один датасет
    python tools/fetch_rgbt.py --raw16 <zip|тека> # 16-біт TIFF тепловізора (FLIR ADAS v2 або власні) для Demo 1
    python tools/fetch_rgbt.py --info             # KAIST, M3FD, MS2, DroneVehicle, DVTOD: де брати вручну

Що качається автоматично (з офіційних посилань авторів на Google Drive; якщо Drive вичерпав денну квоту —
з байт-ідентичних дзеркал на Hugging Face):

  llvip         LLVIP (Jia et al., ICCVW 2021): 1280×1024, RGB і LWIR вирівняні апаратно, 26 сцен, майже все — ніч,
                лише клас person. Спліт авторів — за сценами: train 01–18, 25; test 19–24, 26.
                Підмножина: кожен 6-й кадр train і кожен 4-й test (сусідні кадри майже однакові).
  flir-aligned  FLIR-aligned (Zhang et al., ICIP 2020): 5 142 пари FLIR ADAS v1, вручну вирівняні, 640×512, день і ніч,
                класи person / bicycle / car. Спліт авторів align_train / align_validation — різні поїздки.
                Підмножина: кожен 2-й кадр train і весь validation.

Скрипт читає лише потрібні файли з архівів через HTTP Range (як tools/fetch_mdmt.py), тож повні архіви
(3,7 і 1,3 ГБ) не качаються. Уже завантажені файли пропускаються — скрипт можна перезапускати.

Ліцензії: LLVIP — лише некомерційне використання (README авторів); FLIR — умови Teledyne FLIR
(https://www.oem.flir.com/solutions/automotive/adas-dataset-form/). Для курсу — академічне використання з посиланням на статті.

Структура на диску (як в архівах):
    artifacts/rgbt/llvip/LLVIP/{visible,infrared}/{train,test}/*.jpg, LLVIP/Annotations/*.xml
    artifacts/rgbt/flir_aligned/align/JPEGImages/FLIR_xxxxx_{RGB.jpg,PreviewData.jpeg}, align/Annotations/*.xml,
                                align/align_{train,validation}.txt
    artifacts/rgbt/raw16/*.tiff

Запуск з кореня репозиторію.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mdmt import fetch_member  # noqa: E402  (Range-читання zip — спільний код)
import fetch_llvip  # noqa: E402  (LLVIP: джерела, вибір кадрів, паралельне завантаження — спільний код)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "rgbt"
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

SOURCES = {
    "llvip": dict(dir="llvip", size=fetch_llvip.LLVIP_SIZE, urls=fetch_llvip.LLVIP_URLS),
    "flir-aligned": dict(
        dir="flir_aligned", size=1358345083,
        urls=[GDRIVE.format("1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y"),
              "https://huggingface.co/datasets/jsonhash/FLIR_aligned/resolve/main/aligned.zip"],
    ),
}

INFO = """
Інші RGB-T датасети (завантаження вручну, умови — на сторінках проєктів):

  FLIR ADAS v2   26 442 кадри, 15 класів; 14/16-біт TIFF без AGC і 8-біт JPEG з AGC; RGB і T — різні камери,
                 невирівняні. Форма: https://www.oem.flir.com/solutions/automotive/adas-dataset-form/
                 Для Demo 1:  python tools/fetch_flir_adas.py --raw16   (усе про v2 — tools/fetch_flir_adas.py --help)
                 З уже завантаженого: python tools/fetch_rgbt.py --raw16 <FLIR_ADAS_v2.zip або розпакована тека>
  KAIST          95 328 вирівняних пар (beam splitter), офіційний поділ day / night, пішоходи.
                 https://github.com/soonminhwang/rgbt-ped-detection (≈ 36 ГБ); виправлена розмітка:
                 sanitized (Li et al., BMVC 2018), KAIST-Paired — окремі бокси для RGB і T (AR-CNN, ICCV 2019,
                 https://github.com/luzhang16/AR-CNN)
  M3FD           4 177 вирівняних пар, 6 класів, дим / туман / ніч (TarDAL, CVPR 2022): https://github.com/dlut-dimt/TarDAL
  MS2            стерео RGB / NIR / thermal (uint16) + LiDAR + GNSS/IMU, день / ніч / дощ; глибина вже спроєктована
                 в площину тепловізора (Shin et al., CVPR 2023). Посилання — за запитом:
                 https://sites.google.com/view/multi-spectral-stereo-dataset
  DroneVehicle   28 439 пар RGB-IR з дрона, oriented boxes (TCSVT 2022): https://github.com/VisDrone/DroneVehicle
  DVTOD          2 179 навмисно невирівняних пар з дрона — реєстрація на реальних даних: https://github.com/VDT-2048/DVTOD
"""


session = fetch_llvip.session


def open_index(s, src):
    """Каталог архіву через Range. Перше джерело, що віддає zip очікуваного розміру."""
    try:
        return fetch_llvip.open_index(s, src["urls"], src["size"])
    except fetch_llvip.SourceError as e:
        sys.exit(f"Жодне джерело не відповіло: {e}")


def pick_llvip(infos, full):
    """Кожен 6-й кадр train і 4-й test (або всі) — див. tools/fetch_llvip.py."""
    return fetch_llvip.select(infos, "all", 1 if full else None)


def pick_flir(infos, full, s, url):
    names = {i.filename: i for i in infos}
    lists = [names["align/align_train.txt"], names["align/align_validation.txt"]]
    out = list(lists)
    for info, step in zip(lists, (2, 1)):
        txt = fetch_bytes(s, url, info).decode().split()
        for stem in (txt if full else txt[::step]):
            fid = stem.replace("_PreviewData", "")
            for m in (f"align/JPEGImages/{fid}_PreviewData.jpeg", f"align/JPEGImages/{fid}_RGB.jpg",
                      f"align/Annotations/{fid}_PreviewData.xml"):
                if m in names:
                    out.append(names[m])
    return out


def fetch_bytes(s, url, info):
    tmp = OUT / "_tmp" / Path(info.filename).name
    fetch_member(s, url, info, tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


def download(s, url, members, root: Path, workers=16):
    """Паралельне завантаження з повторами — спільне з tools/fetch_llvip.py."""
    return fetch_llvip.download(s, url, members, root, workers=workers)


def raw16(src: Path, n=60):
    """16-біт TIFF тепловізора: зі zip або теки беремо до n кадрів, рівномірно за іменами."""
    import cv2
    dst = OUT / "raw16"
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
        pick = files[:: max(1, len(files) // n)][:n]
        for p in pick:
            shutil.copy2(p, dst / p.name)
    else:
        with zipfile.ZipFile(src) as z:
            files = sorted((i for i in z.infolist() if i.filename.lower().endswith((".tif", ".tiff"))),
                           key=lambda i: i.filename)
            pick = files[:: max(1, len(files) // n)][:n]
            for i in pick:
                (dst / Path(i.filename).name).write_bytes(z.read(i))
    ok = 0
    for p in sorted(dst.glob("*.tif*")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.dtype != np.uint16:
            p.unlink()
        else:
            ok += 1
    print(f"raw16: {ok} кадрів uint16 -> {dst}" if ok else "raw16: серед TIFF немає uint16 — перевірте джерело")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=list(SOURCES), help="лише один датасет")
    ap.add_argument("--full", action="store_true", help="усі кадри, а не підмножини для демо")
    ap.add_argument("--raw16", type=Path, help="zip або тека з 16-біт TIFF тепловізора (FLIR ADAS v2 або власні)")
    ap.add_argument("--info", action="store_true", help="інші RGB-T датасети: де брати вручну")
    args = ap.parse_args()
    if args.info:
        print(INFO)
        return
    if args.raw16:
        print("Порада: 16-біт TIFF FLIR ADAS v2 без ручного завантаження — python tools/fetch_flir_adas.py --raw16")
        raw16(args.raw16)
        return
    s = session()
    for name, src in SOURCES.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}] читаю каталог архіву…")
        url, infos = open_index(s, src)
        members = pick_llvip(infos, args.full) if name == "llvip" else pick_flir(infos, args.full, s, url)
        failed = download(s, url, members, OUT / src["dir"])
        if failed:
            for f, e in failed[:10]:
                print(f"[fail] {f}: {e}")
            sys.exit(f"Не завантажено {len(failed)} файлів — перезапустіть скрипт, готові файли буде пропущено")
    shutil.rmtree(OUT / "_tmp", ignore_errors=True)
    print(f"Готово -> {OUT}")


if __name__ == "__main__":
    main()
