#!/usr/bin/env python
"""
LLVIP (Jia et al., «LLVIP: A Visible-infrared Paired Dataset for Low-light Vision», ICCVW 2021) — в artifacts/rgbt/llvip/.

    python tools/fetch_llvip.py --list                   # сцени, train/test, день/ніч, розміри (нічого не качає)
    python tools/fetch_llvip.py                          # підмножина для демо, як у fetch_rgbt.py: кожен 6-й train, 4-й test
    python tools/fetch_llvip.py --split test --step 1    # увесь test (3 463 пари)
    python tools/fetch_llvip.py --scenes 25 26           # лише денні сцени (з тим самим проріджуванням)
    python tools/fetch_llvip.py --annotations-only       # лише VOC-розмітка (усі кадри вибраних сцен, ≈ 15 МБ на все)
    python tools/fetch_llvip.py --full                   # увесь архів 4,0 ГБ одним потоком з докачуванням, потім розпаковка
    python tools/fetch_llvip.py --yolo                   # додатково YOLO-мітки й yolo/{visible,infrared}.yaml для ultralytics
    python tools/fetch_llvip.py --raw --list             # сирі невирівняні дані: що є в архіві 10,3 ГБ
    python tools/fetch_llvip.py --raw --scenes 08        # сирі пари 1920×1080 / 1280×720 (кожна 20-та), в artifacts/rgbt/llvip_raw/
    python tools/fetch_llvip.py --raw --videos 01 --step 0   # лише одна пара відео (без кадрів)

Що це: 15 488 пар RGB + LWIR 1280×1024, вирівняних апаратно, 26 сцен, клас person (VOC xml, одна розмітка на пару).
Сцена — перші дві цифри номера кадру. Спліт авторів — за сценами: train 01–18, 25; test 19–24, 26.
Денні — сцени 25 і 26, решта — ніч. У архіві вже виправлена розмітка 2023 року.
Тепловий канал — 8-біт JPEG після AGC; 16-біт сирих даних у LLVIP немає (для них — FLIR ADAS v2).

Джерела: офіційний Google Drive авторів (https://github.com/bupt-ai-cz/LLVIP), а якщо Drive вичерпав денну квоту —
байт-ідентичні дзеркала на Hugging Face (jsonhash/LLVIP, McCheng/LLVIP). Приймається лише файл розміром 4 003 858 172 Б.
Підмножини читаються з архіву по одному файлу через HTTP Range (як tools/fetch_mdmt.py) з перевіркою CRC;
--full качає zip цілком (обірване завантаження продовжується з місця зупинки) і розпаковує.

Сирі дані (--raw): 13 975 невирівняних пар (visible 1920×1080, infrared 1280×720) і 15 пар відео — лише з Google Drive.
Спліт сирих даних інший, ніж у вирівняних: train — сцени 02, 04, 06, 08, 10, 12, 15, 17, 19, 21, 23; test — решта;
сцен 13, 25, 26 немає. Розмітки немає.

Ліцензія: лише некомерційне використання (наукова робота, навчання, публікації) — «Term of Use and License.md»
у репозиторії авторів. Ліцензії, вказані на дзеркалах, її не змінюють.

Структура на диску (як в архіві):
    artifacts/rgbt/llvip/LLVIP/{visible,infrared}/{train,test}/NNNNNN.jpg, LLVIP/Annotations/NNNNNN.xml
    artifacts/rgbt/llvip/yolo/{visible,infrared}/{train,test}/{images,labels}/, yolo/{visible,infrared}.yaml   (--yolo)
    artifacts/rgbt/llvip_raw/LLVIP_raw_images/{train,test}/{infrared,visible}/NNNNNN.jpg,
                             LLVIP_raw_videos/{infrared,visible}/NN.mp4                                       (--raw)

Уже завантажені файли (з правильним розміром) пропускаються — скрипт можна перезапускати. Повторний --full
спершу читає каталог архіву: якщо бракує менше чверті обсягу, докачує лише відсутні файли, а не весь zip.
Запуск з кореня репозиторію.
"""
import argparse
import io
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mdmt import HttpFile, fetch_member  # noqa: E402  (Range-читання zip — спільний код)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "rgbt" / "llvip"
OUT_RAW = ARTIFACTS / "rgbt" / "llvip_raw"
GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

LLVIP_SIZE = 4003858172
LLVIP_URLS = [
    GDRIVE.format("1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"),
    "https://huggingface.co/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip",
    "https://huggingface.co/datasets/McCheng/LLVIP/resolve/main/LLVIP.zip",
]
RAW_ID = "1a0zNvj1mBh1v_HFWJ43LFbNEq8YLXB9-"
RAW_SIZE = 10330888692
RAW_PREFIX = "LLVIP_raw_data/"
RAW_MANUAL = f"""
Сирі дані LLVIP (LLVIP_raw_data.zip, 10,3 ГБ) зараз недоступні для скрипта — найімовірніше, Google Drive вичерпав
денну квоту завантажень. Дзеркал на Hugging Face немає. Варіанти:
  1) спробувати завтра:  python tools/fetch_llvip.py --raw ...
  2) завантажити вручну з Google Drive:  https://drive.google.com/file/d/{RAW_ID}/view
  3) Baidu Yun:  https://pan.baidu.com/s/1VXMiWH1EJGVGoCEV7AMF6Q   (пароль cpfh)
Розпакуйте так, щоб вийшло artifacts/rgbt/llvip_raw/LLVIP_raw_images/... (без верхньої теки LLVIP_raw_data/).
Посилання авторів: https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
"""
LICENCE = ("LLVIP: лише некомерційне використання (див. «Term of Use and License.md» на "
           "https://github.com/bupt-ai-cz/LLVIP); у публікаціях — посилання на Jia et al., ICCVW 2021.")

SPLITS = ("train", "test")
DEMO_STEP = {"train": 6, "test": 4}          # як у fetch_rgbt.py: сусідні кадри майже однакові
DAY_SCENES = {"25", "26"}
RAW_STEP = 20


class SourceError(Exception):
    pass


def session(workers=16):
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers, max_retries=3))
    return s


def probe(s, url):
    """Розмір файлу й кінцевий URL одним GET з Range: bytes=0-0. HEAD тут не годиться: Drive для великих файлів
    інколи відповідає на HEAD порожньою HTML-сторінкою (Content-Length 0), хоча GET з Range працює."""
    with s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        if r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
            return r.url, int(r.headers["Content-Range"].rsplit("/", 1)[1])
        return r.url, int(r.headers.get("Content-Length", 0))


class RangeFile(HttpFile):
    """HttpFile з fetch_mdmt.py, але розмір визначаємо через probe(), а не HEAD."""

    def __init__(self, url, session):  # noqa: D107  (батьківський __init__ свідомо не викликаємо — він робить HEAD)
        io.RawIOBase.__init__(self)
        self.s, self.pos = session, 0
        self.url, self.size = probe(session, url)


def skip_msg(url, err, more):
    print(f"  {url[:70]}…: {str(err)[:120]}" + (" — пробую наступне джерело" if more else ""))


def open_index(s, urls, size):
    """Каталог архіву через Range з першого джерела, що віддає zip очікуваного розміру -> (url, [ZipInfo]).
    Повертаємо початковий URL, а не кінцевий після редиректу: підписані посилання CDN Hugging Face живуть
    лише годину, а requests проходить редирект заново на кожному запиті."""
    errors = []
    for k, url in enumerate(urls):
        try:
            f = RangeFile(url, s)
            if f.size != size:
                raise IOError(f"розмір {f.size} замість {size} (квота Drive або інший файл)")
            with zipfile.ZipFile(io.BufferedReader(f, 1 << 20)) as z:
                return url, [i for i in z.infolist() if not i.is_dir()]
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            skip_msg(url, e, k < len(urls) - 1)
    raise SourceError(f"жодне джерело не відповіло: {errors[-1] if errors else '-'}")


def safe_dest(root: Path, name: str) -> Path:
    """root/name, але лише якщо шлях лишається всередині root (захист від «../» і абсолютних імен у чужому zip)."""
    if not name or name.startswith(("/", "\\")) or ":" in name or ".." in Path(name).parts or \
            not (root / name).resolve().is_relative_to(root.resolve()):
        raise IOError(f"небезпечне ім'я в архіві: {name!r}")
    return root / name


def scene_of(name):
    return name.rsplit("/", 1)[1][:2]


def select(infos, split="all", step=None, scenes=None, annotations_only=False):
    """Пари вирівняного LLVIP: сортуємо infrared/<split>/*.jpg, беремо кожен step-й (за замовчуванням 6 для train,
    4 для test), потім фільтр за сценами. Для кожного кадру — infrared, visible і xml. Порядок і вибір
    при split='all', step=None, scenes=None збігаються з fetch_rgbt.py."""
    names = {i.filename: i for i in infos}
    out = []
    for sp in (SPLITS if split == "all" else (split,)):
        st = step or DEMO_STEP[sp]
        ir = sorted(n for n in names if n.startswith(f"LLVIP/infrared/{sp}/") and n.endswith(".jpg"))
        for n in ir[::st]:
            if scenes and scene_of(n) not in scenes:
                continue
            stem = n.rsplit("/", 1)[1][:-4]
            ms = [f"LLVIP/Annotations/{stem}.xml"] if annotations_only else \
                 [n, f"LLVIP/visible/{sp}/{stem}.jpg", f"LLVIP/Annotations/{stem}.xml"]
            out += [names[m] for m in ms if m in names]
    return out


def select_raw(infos, split="all", step=RAW_STEP, scenes=None, videos=()):
    """Сирі невирівняні пари: кожна step-та за відсортованим списком infrared (step=0 — без кадрів) + вибрані відео."""
    names = {i.filename: i for i in infos}
    out = [names[n] for n in names if n.endswith("README.md")]
    if step:
        for sp in (SPLITS if split == "all" else (split,)):
            base = f"{RAW_PREFIX}LLVIP_raw_images/{sp}/"
            ir = sorted(n for n in names if n.startswith(base + "infrared/") and n.endswith(".jpg"))
            for n in ir[::step]:
                if scenes and scene_of(n) not in scenes:
                    continue
                out += [names[m] for m in (n, n.replace("/infrared/", "/visible/")) if m in names]
    for v in videos:
        for mod in ("infrared", "visible"):
            m = f"{RAW_PREFIX}LLVIP_raw_videos/{mod}/{v}.mp4"
            if m not in names:
                raise SystemExit(f"Відео {v}.mp4 немає в архіві (є 01–15). Див. --raw --list")
            out.append(names[m])
    return out


def fmt_size(b):
    return f"{b / 2**30:.2f} ГБ" if b >= 2**30 else f"{b / 2**20:.1f} МБ"


def fetch_retry(s, url, info, dest, tries=4):
    for k in range(tries):
        try:
            return fetch_member(s, url, info, dest)
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def download(s, url, members, root: Path, strip="", workers=16):
    """Паралельно качає члени архіву в root/<ім'я без префікса strip>; готові файли пропускає. -> [(ім'я, помилка)]"""
    failed, ok = [], []
    for i in members:
        try:
            ok.append((i, safe_dest(root, i.filename[len(strip):])))
        except IOError as e:
            failed.append((i.filename, str(e)))
    todo = [(i, d) for i, d in ok if not (d.exists() and d.stat().st_size == i.file_size)]
    total = sum(i.file_size for i, _ in todo)
    print(f"  {len(members)} файлів, з них качати {len(todo)} ({fmt_size(total)}) -> {root}")
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_retry, s, url, i, d): i for i, d in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut].filename, str(e)[:160]))
            done += futs[fut].file_size
            if k % 100 == 0 or k == len(todo):
                mbps = done / 2**20 / max(time.time() - t0, 1e-3)
                print(f"\r  {k}/{len(todo)} файлів, {fmt_size(done)}/{fmt_size(total)}, {mbps:.2f} МБ/с",
                      end="", flush=True)
    if todo:
        print()
    return failed


def stream_download(s, urls, size, dest: Path, tries=10):
    """Увесь файл одним потоком у dest.part з докачуванням (Range: bytes=N-); приймає лише файл розміру size."""
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == size:
        return dest
    for n, url in enumerate(urls):
        more = n < len(urls) - 1
        try:
            got_size = probe(s, url)[1]
        except Exception as e:  # noqa: BLE001
            skip_msg(url, e, more)
            continue
        if got_size != size:
            skip_msg(url, f"розмір {got_size} замість {size} (квота Drive або інший файл)", more)
            continue
        for k in range(tries):
            have = part.stat().st_size if part.exists() else 0
            if have > size:   # чужий або зіпсований .part: інакше кожен запит дає 416 назавжди
                print(f"  {part.name} більший за архів ({have} > {size} Б) — видаляю й качаю заново")
                part.unlink()
                have = 0
            if have == size:
                part.replace(dest)
                return dest
            try:
                hdr = {"Range": f"bytes={have}-"} if have else {}
                with s.get(url, headers=hdr, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    full = int(r.headers.get("Content-Range", "/0").rsplit("/", 1)[1] or 0) \
                        if r.status_code == 206 else int(r.headers.get("Content-Length", 0))
                    if full != size:
                        raise SourceError(f"розмір {full} замість {size} (квота Drive або інший файл)")
                    if have and r.status_code != 206:  # сервер ігнорує Range — починаємо спочатку
                        have = 0
                    t0, got, shown = time.time(), 0, 0
                    with open(part, "ab" if have else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                            if got - shown >= 64 << 20 or have + got == size:
                                shown = got
                                mbps = got / 2**20 / max(time.time() - t0, 1e-3)
                                print(f"\r  {fmt_size(have + got)}/{fmt_size(size)}, {mbps:.1f} МБ/с",
                                      end="", flush=True)
                print()
            except (SourceError, requests.HTTPError) as e:
                code = e.response.status_code if isinstance(e, requests.HTTPError) and e.response is not None else 0
                if code >= 500 or code == 429:
                    print(f"\n  HTTP {code}, повторюю…")
                    time.sleep(min(2 ** k, 60))
                    continue
                skip_msg(url, e, more)
                break
            except Exception as e:  # noqa: BLE001
                saved = part.stat().st_size if part.exists() else 0
                print(f"\n  обрив ({str(e)[:100]}), продовжую з {fmt_size(saved)}…")
                time.sleep(min(2 ** k, 60))
        if part.exists() and part.stat().st_size == size:
            part.replace(dest)
            return dest
    raise SourceError("жодне джерело не віддало повний файл; перезапустіть — завантаження продовжиться з місця зупинки")


class BadArchive(Exception):
    pass


def extract(zpath: Path, root: Path):
    """Розпаковка з перевіркою CRC (zipfile перевіряє сам); готові файли правильного розміру пропускає.
    Зіпсований архів -> BadArchive (недописаний *.part прибираємо)."""
    tmp = None
    try:
        with zipfile.ZipFile(zpath) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            for k, i in enumerate(infos, 1):
                try:
                    dest = safe_dest(root, i.filename)
                except IOError as e:
                    print(f"\n  пропускаю: {e}")
                    continue
                if not (dest.exists() and dest.stat().st_size == i.file_size):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with z.open(i) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    tmp.replace(dest)
                    tmp = None
                if k % 1000 == 0 or k == len(infos):
                    print(f"\r  розпаковано {k}/{len(infos)}", end="", flush=True)
    except (zipfile.BadZipFile, zlib.error, EOFError) as e:   # CRC теж -> BadZipFile; помилки диска — не сюди
        if tmp is not None and tmp.exists():
            tmp.unlink()
        print()
        raise BadArchive(str(e)[:160]) from e
    print()


def missing_members(infos, root: Path):
    """Члени архіву, яких ще немає в root з правильним розміром."""
    return [i for i in infos
            if not ((root / i.filename).exists() and (root / i.filename).stat().st_size == i.file_size)]


def voc_to_yolo(xml_path: Path):
    """VOC xml -> рядки YOLO «0 cx cy w h» (нормовані до розміру з xml; клас person)."""
    r = ET.parse(xml_path).getroot()
    w = float(r.findtext("size/width") or 1280)
    h = float(r.findtext("size/height") or 1024)
    lines = []
    for o in r.iter("object"):
        if (o.findtext("name") or "").strip().lower() != "person":
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def link_or_copy(src: Path, dst: Path):
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # жорстке посилання: місце на диску не дублюється
    except OSError:
        shutil.copy2(src, dst)


def make_yolo(root: Path, split="all", scenes=None):
    """Для всіх уже завантажених пар у root/LLVIP: root/yolo/<mod>/<split>/{images,labels} + root/yolo/<mod>.yaml."""
    yolo = root / "yolo"
    n = defaultdict(int)
    for sp in (SPLITS if split == "all" else (split,)):
        for img in sorted((root / "LLVIP" / "infrared" / sp).glob("*.jpg")):
            stem = img.stem
            if scenes and stem[:2] not in scenes:
                continue
            xml = root / "LLVIP" / "Annotations" / f"{stem}.xml"
            vis = root / "LLVIP" / "visible" / sp / f"{stem}.jpg"
            if not (xml.exists() and vis.exists()):
                continue
            label = "\n".join(voc_to_yolo(xml))
            for mod, src in (("infrared", img), ("visible", vis)):
                link_or_copy(src, yolo / mod / sp / "images" / src.name)
                lab = yolo / mod / sp / "labels" / f"{stem}.txt"
                lab.parent.mkdir(parents=True, exist_ok=True)
                lab.write_text(label + ("\n" if label else ""), encoding="utf-8")
            n[sp] += 1
    for mod in ("visible", "infrared"):
        have = [sp for sp in SPLITS if (yolo / mod / sp / "images").is_dir()]
        if not have:
            continue
        train = "train" if "train" in have else have[0]
        val = "test" if "test" in have else have[0]
        (yolo / f"{mod}.yaml").write_text(
            f"# LLVIP {mod}, YOLO-мітки з VOC (tools/fetch_llvip.py --yolo); val — test-спліт авторів\n"
            f"path: {(yolo / mod).resolve().as_posix()}\ntrain: {train}/images\nval: {val}/images\n"
            f"names:\n  0: person\n", encoding="utf-8")
    print(f"YOLO: train {n['train']}, test {n['test']} пар -> {yolo}  (data: {yolo / 'visible.yaml'}, "
          f"{yolo / 'infrared.yaml'})")
    if len([sp for sp in SPLITS if n[sp]]) < 2 and split == "all":
        print("  увага: є лише один спліт — у yaml train і val указують на нього")


def print_list(infos, raw=False):
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # scene -> split -> [пар, байт]
    for i in infos:
        p = i.filename[len(RAW_PREFIX):].split("/") if raw else i.filename.split("/")
        if raw and len(p) == 4 and p[0] == "LLVIP_raw_images" and p[3].endswith(".jpg"):
            sp, mod = p[1], p[2]
        elif not raw and len(p) == 4 and p[1] in ("infrared", "visible") and p[3].endswith(".jpg"):
            mod, sp = p[1], p[2]
        else:
            continue
        st = stats[p[3][:2]][sp]
        st[0] += mod == "infrared"
        st[1] += i.file_size
    if not raw:
        ann = sum(i.file_size for i in infos if i.filename.startswith("LLVIP/Annotations/"))
    print(f"{'сцена':>5} {'день/ніч':>8} {'train':>6} {'test':>6} {'МБ':>8}")
    tot = defaultdict(int)
    for sc in sorted(stats):
        d = stats[sc]
        tr, te = d["train"][0], d["test"][0]
        mb = (d["train"][1] + d["test"][1]) / 2**20
        tot["train"] += tr
        tot["test"] += te
        tot["mb"] += mb
        print(f"{sc:>5} {'день' if sc in DAY_SCENES else 'ніч':>8} {tr or '':>6} {te or '':>6} {mb:8.1f}")
    print(f"{'разом':>5} {'':>8} {tot['train']:>6} {tot['test']:>6} {tot['mb']:8.1f}")
    if raw:
        vids = sorted((i for i in infos if i.filename.endswith(".mp4")), key=lambda i: i.filename)
        print(f"Відео: {len(vids) // 2} пар, {sum(i.file_size for i in vids) / 2**30:.2f} ГБ")
        for v in vids:
            print(f"  {v.filename[len(RAW_PREFIX):]:<40} {v.file_size / 2**20:7.1f} МБ")
        print("Розмір кадрів: visible 1920×1080, infrared 1280×720 (не вирівняні). Розмітки немає.")
    else:
        print(f"Розмітка (VOC xml): {ann / 2**20:.1f} МБ. Кадри 1280×1024, пари вирівняні.")
        for sp in SPLITS:
            n = tot[sp]
            print(f"Демо-підмножина {sp}: кожен {DEMO_STEP[sp]}-й кадр ≈ {-(-n // DEMO_STEP[sp])} пар")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="сцени, кількість пар, день/ніч, розміри; нічого не качає")
    ap.add_argument("--split", choices=["train", "test", "all"], default="all",
                    help="спліт авторів: train (сцени 01–18, 25), test (19–24, 26) або all (за замовчуванням)")
    ap.add_argument("--step", type=int, help="брати кожен N-й кадр (за замовчуванням 6 train / 4 test; "
                    f"--annotations-only: 1; --raw: {RAW_STEP}, 0 — без кадрів)")
    ap.add_argument("--scenes", nargs="+", help="лише ці сцени, напр. 19 26 (день — 25, 26); фільтр після проріджування --step")
    ap.add_argument("--full", action="store_true", help="усі 15 488 пар: увесь zip (4,0 ГБ) з докачуванням + розпаковка")
    ap.add_argument("--keep-zip", action="store_true", help="з --full: не видаляти LLVIP.zip після розпаковки")
    ap.add_argument("--annotations-only", action="store_true", help="лише VOC xml вибраних кадрів")
    ap.add_argument("--yolo", action="store_true", help="ще й YOLO-мітки (0 cx cy w h) і yaml для ultralytics в <out>/yolo")
    ap.add_argument("--raw", action="store_true", help="сирі невирівняні пари й відео (Google Drive, 10,3 ГБ архів)")
    ap.add_argument("--videos", nargs="+", default=[], help="з --raw: номери пар відео, напр. 01 07")
    ap.add_argument("--out", type=Path, help=f"куди класти (за замовчуванням {OUT} або, з --raw, {OUT_RAW})")
    ap.add_argument("--workers", type=int, default=16, help="паралельних запитів")
    args = ap.parse_args()

    scenes = {sc.zfill(2) for sc in args.scenes} if args.scenes else None
    if args.full and args.list:
        ap.error("--list нічого не качає, а --full качає весь архів; для огляду запустіть просто --list")
    if args.keep_zip and not args.full:
        ap.error("--keep-zip має сенс лише з --full")
    if args.videos and not args.raw:
        ap.error("--videos стосується лише сирих даних: додайте --raw")
    if args.workers < 1:
        ap.error("--workers має бути ≥ 1")
    if args.full and (args.split != "all" or args.step or scenes or args.annotations_only or args.raw):
        ap.error("--full — це весь архів; для частини використайте --split / --scenes / --step 1 без --full")
    if args.raw and (args.annotations_only or args.yolo or args.keep_zip):
        ap.error("--raw не має розмітки; --annotations-only, --yolo і --keep-zip до нього не застосовні")
    if args.annotations_only and args.yolo:
        ap.error("--yolo потребує зображень, тож не поєднується з --annotations-only")
    if args.step is not None and args.step < (0 if args.raw else 1):
        ap.error("--step має бути ≥ 1 (з --raw ≥ 0)")
    out = args.out or (OUT_RAW if args.raw else OUT)
    print(LICENCE)
    s = session(args.workers)

    if args.raw:
        print("[raw] читаю каталог LLVIP_raw_data.zip…")
        try:
            url, infos = open_index(s, [GDRIVE.format(RAW_ID)], RAW_SIZE)
        except SourceError:
            print(RAW_MANUAL)
            sys.exit(2)
        if args.list:
            print_list(infos, raw=True)
            return
        step = RAW_STEP if args.step is None else args.step
        members = select_raw(infos, args.split, step, scenes, [v.zfill(2) for v in args.videos])
        failed = download(s, url, members, out, strip=RAW_PREFIX, workers=args.workers)
    elif args.full:
        out.mkdir(parents=True, exist_ok=True)
        zpath = out / "LLVIP.zip"
        part = zpath.with_suffix(".zip.part")
        failed = []
        if not zpath.exists() and not part.exists() and not args.keep_zip:
            # Повторний запуск або вже є частина файлів: каталог архіву (≈ 3 МБ через Range) скаже, чого бракує.
            try:
                url, infos = open_index(s, LLVIP_URLS, LLVIP_SIZE)
            except SourceError:
                infos = None
            miss = missing_members(infos, out) if infos else None
            if miss is not None and sum(i.file_size for i in miss) < LLVIP_SIZE // 4:
                # Бракує небагато (або нічого) — докачуємо окремі файли замість 4 ГБ архіву; результат той самий.
                print(f"[full] уже є {len(infos) - len(miss)} із {len(infos)} файлів — решту докачую поштучно")
                failed = download(s, url, miss, out, workers=args.workers)
                zpath = None
        if zpath is not None:
            need = LLVIP_SIZE * 2
            free = shutil.disk_usage(out).free
            print(f"[full] архів {LLVIP_SIZE / 2**30:.2f} ГБ + стільки ж після розпаковки; "
                  f"вільно {free / 2**30:.1f} ГБ")
            if free + (part.stat().st_size if part.exists() else 0) < need * 1.05 and not zpath.exists():
                sys.exit(f"Замало місця: потрібно ≈ {need / 2**30:.1f} ГБ. Звільніть місце або вкажіть --out "
                         "на інший диск")
            try:
                stream_download(s, LLVIP_URLS, LLVIP_SIZE, zpath)
            except SourceError as e:
                sys.exit(f"Не вдалося завантажити LLVIP.zip: {e}")
            print(f"[full] розпаковую -> {out}")
            try:
                extract(zpath, out)
            except BadArchive as e:
                zpath.unlink()
                sys.exit(f"LLVIP.zip пошкоджений ({e}); файл видалено — перезапустіть скрипт, архів буде "
                         "завантажено заново (уже розпаковані файли лишаються)")
            if not args.keep_zip:
                zpath.unlink()
    else:
        print("[llvip] читаю каталог архіву…")
        try:
            url, infos = open_index(s, LLVIP_URLS, LLVIP_SIZE)
        except SourceError as e:
            sys.exit(f"Не вдалося відкрити LLVIP.zip: {e}. Спробуйте пізніше або завантажте вручну "
                     "(https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md) і розпакуйте в " + str(out))
        if args.list:
            print_list(infos)
            return
        step = args.step or (1 if args.annotations_only else None)
        members = select(infos, args.split, step, scenes, args.annotations_only)
        if not members:
            sys.exit("Під вибір не потрапив жоден кадр — перевірте --split / --scenes / --step (див. --list)")
        failed = download(s, url, members, out, workers=args.workers)

    if failed:
        for f, e in failed[:10]:
            print(f"[fail] {f}: {e}")
        sys.exit(f"Не завантажено {len(failed)} файлів — перезапустіть скрипт, готові файли буде пропущено")
    if args.yolo:
        make_yolo(out, "all" if args.full else args.split, scenes)
    print(f"Готово -> {out}")


if __name__ == "__main__":
    main()
