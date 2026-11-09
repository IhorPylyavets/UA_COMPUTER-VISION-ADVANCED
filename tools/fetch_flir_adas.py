#!/usr/bin/env python
"""
Teledyne FLIR ADAS Thermal Dataset v2 — в artifacts/rgbt/flir_adas_v2/ (і 16-біт кадри для Demo 1 заняття 12).

    python tools/fetch_flir_adas.py --list                     # теки, кількість кадрів, розміри (нічого не качає)
    python tools/fetch_flir_adas.py --raw16                    # 60 сирих 16-біт TIFF -> artifacts/rgbt/raw16/ (Demo 1)
    python tools/fetch_flir_adas.py --raw16 200 --splits video # 200 TIFF з відео-спліту (без двох 8-біт послідовностей)
    python tools/fetch_flir_adas.py                            # val + video, кожен 5-й кадр, обидві модальності, ≈ 0,7 ГБ
    python tools/fetch_flir_adas.py --pairs --step 10          # лише пари RGB↔T з відео (офіційна мапа), кожна 10-та
    python tools/fetch_flir_adas.py --splits val --modality thermal --step 1   # увесь тепловий val (JPEG + TIFF)
    python tools/fetch_flir_adas.py --annotations-only         # coco.json і таблиці val + video (≈ 5 МБ мережі, ≈ 87 МБ на диску)
    python tools/fetch_flir_adas.py --full                     # усе (≈ 12 ГБ на диску), файл за файлом через Range
    python tools/fetch_flir_adas.py --download-zip             # увесь FLIR_ADAS_v2.zip (11,9 ГБ) з докачуванням і md5
    python tools/fetch_flir_adas.py --zip D:/FLIR_ADAS_v2.zip --raw16      # з уже завантаженого архіву
    python tools/fetch_flir_adas.py --src D:/FLIR_ADAS_v2 --pairs          # з уже розпакованої теки
    python tools/fetch_flir_adas.py --url <посилання з листа FLIR> --list  # з іншого посилання на той самий zip

Що це: 26 442 розмічені кадри (README FLIR), 15 класів (person, bike, car, motor, bus, train, truck, light, hydrant,
sign, dog, skateboard, stroller, scooter, other vehicle), розмітка COCO (coco.json, рекомендована) і сирий формат
Conservator (index.json, великий — лише з --index). Тепловізор Tau 2 640×512, HFOV 45°; RGB — BlackFly S, HFOV 52,8°.
Для теплового каналу є і 8-біт JPEG після AGC (data/*.jpg), і сирий 16-біт TIFF без AGC (analyticsData/*.tiff,
T-linear, реальний діапазон ≈ 14 біт) з тим самим іменем — вони піксель-у-піксель збігаються (той самий сенсор).
Виняток: у video_thermal_test дві послідовності (video-ZAtDSNuZZjkZFvMAo, 1 033 кадри, і video-t3f7QC8hZr6zYXpEZ,
565 кадрів) мають в analyticsData 8-біт TIFF, а не 16-біт. --raw16 їх пропускає (і будь-яку іншу послідовність,
у якій трапиться не-uint16 кадр, — список у raw16/.flir_not_uint16.txt), решта режимів копіює як є і попереджає.
У train (133 послідовності) і val (17) перевірено по кадру з кожної — усі 16-біт.

Теки архіву (у zip немає кореневої теки) і як їх вибирати через --splits:
    train  -> images_rgb_train/,  images_thermal_train/     10 319 RGB / 10 742 T
    val    -> images_rgb_val/,    images_thermal_val/        1 085 RGB /  1 144 T
    video  -> video_rgb_test/,    video_thermal_test/        3 749 RGB /  3 749 T (послідовні кадри 30 к/с, track_id)
У кожній: data/*.jpg, coco.json, index.json, *annotation_counts.{txt,tsv}; у теплових ще analyticsData/*.tiff.

ВАЖЛИВО про пари RGB↔T. У v2 RGB і тепловізор — різні камери з різними полями зору й роздільністю, кадри
НЕ вирівняні піксельно. Офіційна відповідність кадрів є лише для відео: rgb_to_thermal_vid_map.json
(3 749 пар «ім'я RGB-кадру -> ім'я T-кадру», синхронні за часом; лежить окремо від zip, скрипт його качає).
Для train / val офіційних пар немає (різні video_id у двох камер), тож --pairs бере лише video.
Геометрична реєстрація пар (гомографія / кроп) — ваше завдання в ДЗ 12. Готові вирівняні пари є у FLIR-aligned
(5 142 пари на основі v1, лише 8-біт) — tools/fetch_rgbt.py.

Джерела, за пріоритетом:
  --zip ФАЙЛ   уже завантажений FLIR_ADAS_v2.zip (офіційний або дзеркало з префіксом FLIR_ADAS_v2/);
  --src ТЕКА   уже розпакована тека;
  --url URL    будь-яке пряме посилання на zip (наприклад, з листа після форми). Якщо сервер підтримує
               HTTP Range — читаємо з архіву лише потрібні файли; інакше zip качається цілком з докачуванням;
  за замовчуванням — офіційний публічний файл Teledyne FLIR (Amazon S3 / CloudFront, Range працює, форма не
               потрібна): https://adas-dataset-v2.flirconservator.com/dataset/full/FLIR_ADAS_v2.zip
               (11 865 443 849 Б, md5 3150533d3a712c598aa110d5513a1671). Каталог архіву — ≈ 8 МБ за 5 запитів,
               далі кожен файл — один Range-запит з перевіркою CRC (як tools/fetch_mdmt.py).
Якщо жодне джерело недоступне (або --offline без --zip/--src) — скрипт пояснює, як отримати архів, і виходить з кодом 2.

Ліцензія: «Teledyne FLIR Non-Commercial Image Dataset License Agreement» — лише некомерційне використання;
є обмеження на публікацію натренованих на цих даних моделей (ваг). Завантажуючи дані, ви приймаєте умови.
Посилання на умови з офіційного README.txt — https://www.flir.com/oem/adas/adas-dataset-agree/ — зараз не
відкривається (404); умови показує сторінка реєстрації https://www.oem.flir.com/solutions/automotive/adas-dataset-form/
і сторінка датасету https://adas-dataset-v2.flirconservator.com/ . Не викладайте дані в публічний доступ.

Структура на диску (як в архіві):
    artifacts/rgbt/flir_adas_v2/{images_rgb_train,images_rgb_val,video_rgb_test}/data/*.jpg, coco.json, ...
    artifacts/rgbt/flir_adas_v2/{images_thermal_*,video_thermal_test}/{data/*.jpg,analyticsData/*.tiff}, coco.json, ...
    artifacts/rgbt/flir_adas_v2/rgb_to_thermal_vid_map.json, README.txt
    artifacts/rgbt/raw16/<ім'я кадру>.tiff                                                   (--raw16)

Уже наявні файли правильного розміру пропускаються — скрипт можна перезапускати. Наявні файли в raw16/ не
змінюються й не видаляються (навіть пошкоджені — такі скрипт лише називає). Уже завантажений <out>/FLIR_ADAS_v2.zip
повного розміру використовується як --zip; md5 перевіряється один раз (позначка FLIR_ADAS_v2.zip.md5ok).
Запуск з кореня репозиторію.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mdmt import HttpFile, fetch_member  # noqa: E402  (Range-читання zip — спільний код)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
OUT = ARTIFACTS / "rgbt" / "flir_adas_v2"
RAW16 = ARTIFACTS / "rgbt" / "raw16"

BASE = "https://adas-dataset-v2.flirconservator.com/"
ZIP_URL = BASE + "dataset/full/FLIR_ADAS_v2.zip"
ZIP_SIZE = 11865443849
ZIP_MD5 = "3150533d3a712c598aa110d5513a1671"
MAP_NAME = "rgb_to_thermal_vid_map.json"
SIDECARS = {MAP_NAME: BASE + "dataset/" + MAP_NAME, "README.txt": BASE + "dataset/README.txt"}
FORM = "https://www.oem.flir.com/solutions/automotive/adas-dataset-form/"
TERMS = "https://www.flir.com/oem/adas/adas-dataset-agree/"   # з офіційного README.txt; зараз 404

SPLITS = {"train": ("images_rgb_train", "images_thermal_train"),
          "val": ("images_rgb_val", "images_thermal_val"),
          "video": ("video_rgb_test", "video_thermal_test")}
FOLDERS = [f for pair in SPLITS.values() for f in pair]
FOLDER_RE = re.compile(r"^(?:.*/)?(" + "|".join(FOLDERS) + r")/(.+)$")
DEFAULT_SPLITS = ["val", "video"]
DEFAULT_STEP = 5
RAW16_SPLITS = ["val"]           # 1 144 кадри з багатьох поїздок, день і ніч — різноманітніше, ніж одне відео
# Послідовності, у яких analyticsData/*.tiff — 8-біт (перевірено 4 кадри з кожної); решта відео, train і val — 16-біт.
KNOWN_8BIT = {"video-ZAtDSNuZZjkZFvMAo", "video-t3f7QC8hZr6zYXpEZ"}
SKIP_FILE = ".flir_not_uint16.txt"  # у теці raw16: послідовності, знайдені не-uint16 під час попередніх запусків
SEQ_RE = re.compile(r"^(video-[A-Za-z0-9]+)-frame-")

LICENCE = (f"FLIR ADAS v2: лише некомерційне використання; завантажуючи, ви приймаєте умови Teledyne FLIR\n"
           f"(їх показують {FORM} і {BASE}; посилання з README — {TERMS} — зараз 404).\n"
           f"Не викладайте дані й натреновані на них ваги в публічний доступ.")

NO_SOURCE = f"""
Немає доступного джерела FLIR ADAS v2. Як отримати архів:
  1) Зареєструйтесь на офіційній сторінці Teledyne FLIR (ім'я, email, організація) і прийміть умови
     (лише некомерційне використання):
        {FORM}
     (умови також на {BASE}; посилання з офіційного README {TERMS} зараз не відкривається)
  2) Завантажте ОДИН файл FLIR_ADAS_v2.zip (11 865 443 849 Б, md5 {ZIP_MD5}).
     Сторінка завантаження: {BASE}
     Прямий файл:           {ZIP_URL}
     Або 12 частин по 1 ГБ (dataset/parts/FLIR_ADAS_v2.zip.00 … .11) — це просто розрізаний zip, склейте:
        cat FLIR_ADAS_v2.zip.?? > FLIR_ADAS_v2.zip            (Git Bash / Linux / macOS)
     Перевірка: md5sum FLIR_ADAS_v2.zip   (Windows: certutil -hashfile FLIR_ADAS_v2.zip MD5)
     Окремо (не в zip) лежить мапа пар відео: {SIDECARS[MAP_NAME]}
  3) Далі, з кореня репозиторію:
        python tools/fetch_flir_adas.py --zip <шлях>/FLIR_ADAS_v2.zip --list
        python tools/fetch_flir_adas.py --zip <шлях>/FLIR_ADAS_v2.zip --raw16      # 60 кадрів 16-біт для Demo 1
        python tools/fetch_flir_adas.py --zip <шлях>/FLIR_ADAS_v2.zip --pairs      # пари RGB↔T з відео
     Уже розпакували — --src <тека>; маєте інше пряме посилання — --url <посилання>.
"""


class SourceError(Exception):
    pass


def session(workers=16):
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers, max_retries=3))
    return s


def probe(s, url):
    """(кінцевий url, розмір, чи працює Range) одним GET з Range: bytes=0-0."""
    with s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        if r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
            return r.url, int(r.headers["Content-Range"].rsplit("/", 1)[1]), True
        return r.url, int(r.headers.get("Content-Length", 0)), False


class RangeFile(HttpFile):
    """HttpFile з fetch_mdmt.py, але розмір — з probe() (деякі сервери віддають на HEAD не те)."""

    def __init__(self, url, size, session):  # noqa: D107  (батьківський __init__ свідомо не викликаємо — він робить HEAD)
        io.RawIOBase.__init__(self)
        self.s, self.url, self.size, self.pos = session, url, size, 0


def normalise(name):
    """Ім'я в архіві -> шлях як в офіційному zip (без префікса на кшталт FLIR_ADAS_v2/) або None, якщо не потрібне."""
    name = name.replace("\\", "/")
    if name.endswith("/"):
        return None
    if m := FOLDER_RE.match(name):
        return f"{m[1]}/{m[2]}"
    base = name.rsplit("/", 1)[-1]
    return base if base in SIDECARS else None


# ----------------------------------------------------------------------------------------------------------- джерела
class Source:
    """entries: {нормалізований шлях: (розмір, посилання на член архіву / файл)}"""
    parallel = True
    label = ""

    def fetch(self, rel, dest: Path):
        raise NotImplementedError


class HttpZip(Source):
    def __init__(self, s, url, size):
        self.s, self.url = s, url
        with zipfile.ZipFile(io.BufferedReader(RangeFile(url, size, s), 1 << 20)) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
        self.entries = {r: (i.file_size, i) for i in infos if (r := normalise(i.filename))}
        self.label = url

    def fetch(self, rel, dest):
        info = self.entries[rel][1]
        for k in range(4):
            try:
                return fetch_member(self.s, self.url, info, dest)
            except Exception:  # noqa: BLE001
                if k == 3:
                    raise
                time.sleep(2 ** k)


class LocalZip(Source):
    parallel = False

    def __init__(self, path: Path):
        self.z = zipfile.ZipFile(path)
        self.entries = {r: (i.file_size, i) for i in self.z.infolist()
                        if not i.is_dir() and (r := normalise(i.filename))}
        self.label, self.near = str(path), path.parent

    def fetch(self, rel, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self.z.open(self.entries[rel][1]) as src, open(tmp, "wb") as dst:   # zipfile сам перевіряє CRC
            shutil.copyfileobj(src, dst, 1 << 20)
        tmp.replace(dest)


class LocalDir(Source):
    parallel = False

    def __init__(self, path: Path):
        self.entries = {}
        for p in path.rglob("*"):
            if p.is_file() and (r := normalise(p.relative_to(path).as_posix())):
                self.entries.setdefault(r, (p.stat().st_size, p))
        self.label, self.near = str(path), path

    def fetch(self, rel, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(self.entries[rel][1], tmp)
        tmp.replace(dest)


def stream_download(s, url, dest: Path, size=None, tries=10):
    """Увесь файл одним потоком у dest.part з докачуванням (Range: bytes=N-), якщо сервер його підтримує."""
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"  {dest.name} уже є ({dest.stat().st_size / 2**30:.2f} ГБ)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if size and have == size:
            break
        try:
            with s.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                if r.status_code == 416 and have:          # уже все є
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    print("  сервер не підтримує докачування — починаю спочатку")
                    have = 0
                full = int(r.headers["Content-Range"].rsplit("/", 1)[1]) if r.status_code == 206 \
                    else int(r.headers.get("Content-Length", 0)) or None
                if size and full and full != size:
                    raise SourceError(f"розмір {full} замість {size} — це інший файл")
                size = size or full
                t0, got, shown = time.time(), 0, 0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if got - shown >= 64 << 20 or (size and have + got == size):
                            shown = got
                            mbps = got / 2**20 / max(time.time() - t0, 1e-3)
                            print(f"\r  {(have + got) / 2**30:.2f}/{(size or 0) / 2**30:.2f} ГБ, {mbps:.1f} МБ/с",
                                  end="", flush=True)
                print()
            if size is None or part.stat().st_size == size:
                break
        except SourceError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"\n  обрив ({str(e)[:100]}), повтор {k + 1}/{tries}…")
            time.sleep(min(2 ** k, 60))
    if not part.exists() or (size and part.stat().st_size != size):
        raise SourceError("файл не докачано; перезапустіть — завантаження продовжиться з місця зупинки")
    part.replace(dest)
    return dest


def md5_ok(path: Path, expected):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for i, chunk in enumerate(iter(lambda: f.read(1 << 24), b"")):
            h.update(chunk)
            if i % 64 == 0:
                print(f"\r  md5: {i * 16 / 1024:.1f} ГБ", end="", flush=True)
    print()
    return h.hexdigest() == expected


def open_local_zip(path: Path):
    """LocalZip з українським поясненням замість traceback, якщо архів пошкоджений чи недокачаний."""
    try:
        return LocalZip(path)
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        sys.exit(f"{path} не читається як zip ({str(e)[:100]}). Архів пошкоджений або недокачаний: перевірте md5 "
                 f"({ZIP_MD5} для повного FLIR_ADAS_v2.zip, {ZIP_SIZE} Б) або заново склейте всі 12 частин "
                 f"(cat FLIR_ADAS_v2.zip.?? > FLIR_ADAS_v2.zip).")


def check_md5(zpath: Path):
    """md5 повного zip — один раз; після успіху поруч лишається позначка *.md5ok (md5, розмір, mtime)."""
    mark = zpath.with_name(zpath.name + ".md5ok")
    st = zpath.stat()
    stamp = f"{ZIP_MD5} {st.st_size} {int(st.st_mtime)}"
    if mark.is_file() and mark.read_text(encoding="utf-8").strip() == stamp:
        print("[zip] md5 уже перевірено раніше")
        return
    print("[zip] перевіряю md5…")
    if not md5_ok(zpath, ZIP_MD5):
        sys.exit(f"md5 не збігається з {ZIP_MD5}: видаліть {zpath} і завантажте знову")
    mark.write_text(stamp + "\n", encoding="utf-8")


def open_source(args, s):
    """Джерело за пріоритетом --zip, --src, --url, офіційний файл. Немає жодного — інструкція й exit 2."""
    if args.zip:
        if not args.zip.is_file():
            sys.exit(f"Немає файлу {args.zip}")
        return open_local_zip(args.zip)
    if args.src:
        if not args.src.is_dir():
            sys.exit(f"Немає теки {args.src}")
        src = LocalDir(args.src)
        if not any("/" in r for r in src.entries):
            sys.exit(f"У {args.src} немає тек images_*/video_* FLIR ADAS v2 (очікується структура архіву)")
        return src
    zpath = args.out / "FLIR_ADAS_v2.zip"
    if not args.url and zpath.is_file() and zpath.stat().st_size == ZIP_SIZE:
        print(f"[zip] беру вже завантажений {zpath}")
        if args.download_zip:
            check_md5(zpath)
        return open_local_zip(zpath)
    if args.offline:
        print(NO_SOURCE)
        sys.exit(2)
    url = args.url or ZIP_URL
    official = url == ZIP_URL
    try:
        final, size, ranges = probe(s, url)
        if official and size != ZIP_SIZE:
            raise SourceError(f"розмір {size} замість {ZIP_SIZE} — файл на сервері змінився")
        if args.download_zip or not ranges:
            if not ranges:
                print("  сервер не підтримує HTTP Range — качаю zip цілком (докачування неможливе)")
            free = shutil.disk_usage(next(p for p in [args.out, *args.out.parents] if p.exists())).free
            print(f"[zip] {size / 2**30:.2f} ГБ -> {zpath} (вільно {free / 2**30:.1f} ГБ)")
            if size and free < size * 1.05 and not zpath.exists() and not zpath.with_suffix(".zip.part").exists():
                sys.exit("Замало місця на диску. Вкажіть --out на інший диск")
            stream_download(s, final, zpath, size or None)
            if size == ZIP_SIZE:
                check_md5(zpath)
            return open_local_zip(zpath)
        print(f"[index] читаю каталог архіву ({size / 2**30:.2f} ГБ) через Range…")
        return HttpZip(s, final, size)
    except (SourceError, requests.RequestException, zipfile.BadZipFile, OSError) as e:
        print(f"Не вдалося відкрити {url[:90]}: {str(e)[:160]}")
        print(NO_SOURCE)
        sys.exit(2)


# ----------------------------------------------------------------------------------------------------------- мапа пар
def get_small(s, url, dest: Path):
    r = s.get(url, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def ensure_sidecars(src, out: Path, s, offline, need_map=False):
    """README.txt і мапа пар: з джерела, з теки поруч із --zip / в --src, або з офіційного сайту (≈ 0,5 МБ)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SIDECARS.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if name in src.entries:
            src.fetch(name, dest)
        elif (near := getattr(src, "near", None)) and (near / name).is_file():
            shutil.copyfile(near / name, dest)
        elif not offline:
            try:
                get_small(s, url, dest)
            except requests.RequestException as e:
                print(f"  не вдалося завантажити {name}: {str(e)[:100]}")
    path = out / MAP_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if need_map:
        sys.exit(f"Немає {MAP_NAME}: покладіть його поруч із zip або в {out} (джерело: {SIDECARS[MAP_NAME]})")
    return None


def peek_map(src):
    """Мапа для --list без запису на диск: лише якщо вона вже є в джерелі / поруч (з HTTP не качаємо)."""
    near = getattr(src, "near", None)
    if MAP_NAME in src.entries and not isinstance(src, HttpZip):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / MAP_NAME
            src.fetch(MAP_NAME, tmp)
            return json.loads(tmp.read_text(encoding="utf-8"))
    if near and (near / MAP_NAME).is_file():
        return json.loads((near / MAP_NAME).read_text(encoding="utf-8"))
    return None


# ----------------------------------------------------------------------------------------------------------- вибір
def folders_for(splits, modality):
    mods = ("rgb", "thermal") if modality == "both" else (modality,)
    return [f for sp in splits for f in SPLITS[sp] if f.split("_")[1] in mods]


def frames(entries, folder):
    return sorted(r for r in entries if r.startswith(f"{folder}/data/") and r.lower().endswith((".jpg", ".jpeg")))


def tiff_of(rel):
    folder, _, name = rel.split("/", 2)
    return f"{folder}/analyticsData/{name.rsplit('.', 1)[0]}.tiff"


def ann_files(entries, folder, index):
    out = []
    for r in entries:
        if r.startswith(folder + "/") and r.count("/") == 1:
            if r.endswith("index.json") and not index:
                continue
            out.append(r)
    return sorted(out)


def select(entries, splits, modality, step, ann_only, tiff, index, pairs, mapping):
    """Список нормалізованих шляхів для копіювання. pairs — лише відео, за офіційною мапою RGB -> T."""
    out = []
    if pairs:
        for f in SPLITS["video"]:
            out += ann_files(entries, f, index)
        if ann_only:
            return out
        miss = 0
        for k in sorted(mapping)[::step]:
            rgb, th = f"video_rgb_test/data/{k}", f"video_thermal_test/data/{mapping[k]}"
            if rgb not in entries or th not in entries:
                miss += 1
                continue
            out += [rgb, th] + ([tiff_of(th)] if tiff and tiff_of(th) in entries else [])
        if miss:
            print(f"  {miss} пар з мапи відсутні в джерелі — пропускаю")
        return out
    for f in folders_for(splits, modality):
        out += ann_files(entries, f, index)
        if ann_only:
            continue
        for r in frames(entries, f)[::step]:
            out.append(r)
            if tiff and "thermal" in f and tiff_of(r) in entries:
                out.append(tiff_of(r))
    return out


# ----------------------------------------------------------------------------------------------------------- копіювання
def copy_all(src, rels, root: Path, workers=16, dest_of=None):
    """Копіює/качає вибрані файли; готові файли правильного розміру пропускає. -> [(шлях, помилка)]"""
    dest_of = dest_of or (lambda r: root / r)
    todo = [r for r in rels if not (dest_of(r).exists() and dest_of(r).stat().st_size == src.entries[r][0])]
    total = sum(src.entries[r][0] for r in todo)
    print(f"  {len(rels)} файлів, з них потрібно {len(todo)} ({total / 2**30:.2f} ГБ) -> {root}")
    done, failed, t0 = 0, [], time.time()

    def report(k):
        if k % 100 == 0 or k == len(todo):
            mbps = done / 2**20 / max(time.time() - t0, 1e-3)
            print(f"\r  {k}/{len(todo)} файлів, {done / 2**30:.2f}/{total / 2**30:.2f} ГБ, {mbps:.1f} МБ/с",
                  end="", flush=True)

    if src.parallel:
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(src.fetch, r, dest_of(r)): r for r in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    failed.append((futs[fut], str(e)[:160]))
                done += src.entries[futs[fut]][0]
                report(k)
    else:
        for k, r in enumerate(todo, 1):
            try:
                src.fetch(r, dest_of(r))
            except Exception as e:  # noqa: BLE001
                failed.append((r, str(e)[:160]))
            done += src.entries[r][0]
            report(k)
    if todo:
        print()
    return failed


def seq_of(rel):
    """Ідентифікатор послідовності (video-<id>) з імені кадру; якщо формат інший — саме ім'я."""
    name = rel.rsplit("/", 1)[-1]
    return m[1] if (m := SEQ_RE.match(name)) else name


def tiff_bits(path: Path):
    """BitsPerSample з заголовка TIFF (тег 258) без читання пікселів; None, якщо не вдалося."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            bo = {b"II": "<", b"MM": ">"}[head[:2]]
            f.seek(int.from_bytes(head[4:8], "little" if bo == "<" else "big"))
            n = int(np.frombuffer(f.read(2), bo + "u2")[0])
            for _ in range(n):
                e = f.read(12)
                tag, typ = np.frombuffer(e[:4], bo + "u2")
                if tag == 258:
                    if int(np.frombuffer(e[4:8], bo + "u4")[0]) == 1 and typ == 3:
                        return int(np.frombuffer(e[8:10], bo + "u2")[0])
                    f.seek(int(np.frombuffer(e[8:12], bo + "u4")[0]))
                    return int(np.frombuffer(f.read(2), bo + "u2")[0])
            return 1  # тега немає — за специфікацією TIFF це 1 біт
    except Exception:  # noqa: BLE001
        return None


def warn_not16(rels, root: Path):
    """Попередження, якщо серед скопійованих analyticsData/*.tiff є не 16-біт (див. KNOWN_8BIT)."""
    tifs = [r for r in rels if "/analyticsData/" in r and (root / r).is_file()]
    bad = defaultdict(int)
    for r in tifs:
        if tiff_bits(root / r) != 16:
            bad[seq_of(r)] += 1
    if bad:
        print(f"[увага] {sum(bad.values())} з {len(tifs)} TIFF в analyticsData — не 16-біт (послідовності: "
              + ", ".join(f"{k} ×{v}" for k, v in sorted(bad.items())) + "). Для сирого сигналу беріть інші.")


def raw16(src, n, splits, dst: Path, workers):
    """n рівномірно розташованих (за іменем) 16-біт TIFF з analyticsData вибраних сплітів -> dst/<ім'я>.tiff.

    Послідовності з KNOWN_8BIT і з dst/.flir_not_uint16.txt пропускаються. Якщо щойно завантажений кадр не uint16,
    його послідовність дописується в цей список, кадр видаляється, і вибір повторюється без неї — щоб вийшло N кадрів,
    а перезапуск нічого не качав. Наявні файли в dst не перезаписуються й не видаляються."""
    import cv2
    tifs = sorted(r for f in folders_for(splits, "thermal") for r in src.entries
                  if r.startswith(f"{f}/analyticsData/") and r.lower().endswith((".tif", ".tiff")))
    if not tifs:
        sys.exit(f"У джерелі немає analyticsData/*.tiff для сплітів {splits}")
    dst.mkdir(parents=True, exist_ok=True)
    skip_path = dst / SKIP_FILE
    try:
        learned = set(skip_path.read_text(encoding="utf-8").split()) if skip_path.is_file() else set()
    except OSError:
        learned = set()
    bad_seq = KNOWN_8BIT | learned
    dest_of = lambda r: dst / r.rsplit("/", 1)[1]  # noqa: E731
    written, failed, foreign_bad, pick = set(), [], [], []
    for _ in range(10):
        cands = [r for r in tifs if seq_of(r) not in bad_seq]
        if not cands:
            pick = []
            break
        idx = np.unique(np.linspace(0, len(cands) - 1, min(n, len(cands))).round().astype(int))
        pick = [cands[i] for i in idx]
        print(f"[raw16] {len(pick)} з {len(cands)} TIFF ({', '.join(splits)}"
              + (f"; без {len(tifs) - len(cands)} кадрів 8-біт послідовностей" if len(cands) < len(tifs) else "")
              + ")")
        fresh = [r for r in pick if not dest_of(r).exists()]      # наявні файли не перезаписуємо
        failed = copy_all(src, fresh, dst, workers, dest_of)
        written |= {r for r in fresh if dest_of(r).exists()}
        new_bad, foreign_bad = set(), []
        for r in pick:
            p = dest_of(r)
            if not p.exists():
                continue
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is not None and im.dtype == np.uint16 and im.ndim == 2:
                continue
            if r in written:            # видаляємо лише записане цим запуском, чуже не чіпаємо
                p.unlink()
                written.discard(r)
                new_bad.add(seq_of(r))
            else:
                foreign_bad.append(p.name)
        if not new_bad:
            break
        print(f"[raw16] не uint16: {', '.join(sorted(new_bad))} — пропускаю ці послідовності й добираю кадри")
        bad_seq |= new_bad
        try:
            skip_path.write_text("\n".join(sorted(bad_seq - KNOWN_8BIT)) + "\n", encoding="utf-8")
        except OSError:
            pass
    for r in written - set(pick):   # кадри попередніх раундів, що випали з остаточного вибору (лише наші)
        dest_of(r).unlink(missing_ok=True)
    ok = sum(1 for r in pick if dest_of(r).exists()) - len(foreign_bad)
    print(f"[raw16] {ok} кадрів uint16 у {dst}" + (f" (просили {n})" if ok < n else "")
          + (f"; наявні файли не читаються як uint16 (не чіпаю, замініть вручну): {len(foreign_bad)} "
             f"({foreign_bad[:3]})" if foreign_bad else ""))
    if not ok:
        sys.exit("Жоден TIFF не читається як uint16 — перевірте джерело")
    return failed


def print_list(src, mapping):
    st = defaultdict(lambda: [0, 0, 0, 0, 0])     # folder -> [jpg, jpg Б, tiff, tiff Б, json Б]
    for r, (size, _) in src.entries.items():
        if "/" not in r:
            continue
        folder, rest = r.split("/", 1)
        s = st[folder]
        if rest.startswith("data/"):
            s[0] += 1
            s[1] += size
        elif rest.startswith("analyticsData/"):
            s[2] += 1
            s[3] += size
        else:
            s[4] += size
    print(f"Джерело: {src.label}")
    print(f"{'split':>6}  {'тека':<22} {'JPEG':>6} {'ГБ':>6} {'TIFF16':>7} {'ГБ':>6} {'json МБ':>8}")
    tot = [0, 0, 0, 0, 0]
    for sp, pair in SPLITS.items():
        for f in pair:
            s = st.get(f, [0] * 5)
            tot = [a + b for a, b in zip(tot, s)]
            print(f"{sp:>6}  {f:<22} {s[0]:>6} {s[1] / 2**30:6.2f} {s[2]:>7} {s[3] / 2**30:6.2f} {s[4] / 2**20:8.1f}")
    print(f"{'разом':>6}  {'':<22} {tot[0]:>6} {tot[1] / 2**30:6.2f} {tot[2]:>7} {tot[3] / 2**30:6.2f} "
          f"{tot[4] / 2**20:8.1f}")
    if mapping is not None:
        have = sum(1 for k, v in mapping.items() if f"video_rgb_test/data/{k}" in src.entries
                   and f"video_thermal_test/data/{v}" in src.entries)
        print(f"Пари RGB↔T у відео ({MAP_NAME}): {len(mapping)} у мапі, {have} наявні в джерелі")
    else:
        print(f"Пари RGB↔T у відео: мапа {MAP_NAME} окремо від zip ({SIDECARS[MAP_NAME]}), 3 749 пар")
    print("Train / val: офіційних пар RGB↔T немає. Кадри RGB і T не вирівняні піксельно (різні камери).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_argument_group("джерело (за пріоритетом)")
    g.add_argument("--zip", type=Path, help="уже завантажений FLIR_ADAS_v2.zip")
    g.add_argument("--src", type=Path, help="уже розпакована тека")
    g.add_argument("--url", help="пряме посилання на zip (за замовчуванням — офіційний файл Teledyne FLIR)")
    g.add_argument("--offline", action="store_true", help="не звертатися до мережі (лише --zip / --src)")
    g.add_argument("--download-zip", action="store_true",
                   help="качати zip цілком (11,9 ГБ, докачування + md5) у <out>/FLIR_ADAS_v2.zip, потім розпакувати вибір")
    ap.add_argument("--list", action="store_true", help="теки, кількість кадрів і розміри; нічого не копіює")
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS),
                    help=f"train / val / video (за замовчуванням {' '.join(DEFAULT_SPLITS)}; для --raw16 — val)")
    ap.add_argument("--modality", choices=["rgb", "thermal", "both"], default="both")
    ap.add_argument("--step", type=int, help=f"кожен N-й кадр за іменем (за замовчуванням {DEFAULT_STEP}; --full: 1)")
    ap.add_argument("--pairs", action="store_true", help="лише синхронні пари RGB↔T відео за офіційною мапою")
    ap.add_argument("--annotations-only", action="store_true", help="лише coco.json і таблиці кількостей (+ мапа)")
    ap.add_argument("--index", action="store_true", help="ще й index.json (сирий формат Conservator, до 154 МБ на теку)")
    ap.add_argument("--no-tiff", action="store_true", help="без 16-біт TIFF (analyticsData) для теплових кадрів")
    ap.add_argument("--full", action="store_true", help="усе: усі спліти, обидві модальності, кожен кадр, index.json")
    ap.add_argument("--raw16", type=int, nargs="?", const=60, metavar="N",
                    help="лише N (за замовчуванням 60) рівномірно взятих 16-біт TIFF -> artifacts/rgbt/raw16/ (Demo 1)")
    ap.add_argument("--raw16-dir", type=Path, default=RAW16, help=f"куди класти --raw16 (за замовчуванням {RAW16})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"куди класти датасет (за замовчуванням {OUT})")
    ap.add_argument("--workers", type=int, default=16, help="паралельних запитів (для HTTP)")
    args = ap.parse_args()

    if args.step is not None and args.step < 1:
        ap.error("--step має бути ≥ 1")
    if args.raw16 is not None and args.raw16 < 1:
        ap.error("--raw16 N: N ≥ 1")
    if args.pairs and args.modality != "both":
        ap.error("--pairs — це пари RGB↔T, --modality тут завжди both")
    if args.full and (args.splits or args.step or args.pairs or args.annotations_only or args.no_tiff):
        ap.error("--full — це все; для частини використайте --splits / --step / --modality без --full")
    if sum(map(bool, (args.zip, args.src, args.url))) > 1:
        ap.error("вкажіть одне джерело: --zip, --src або --url")
    if args.offline and args.url:
        ap.error("--offline і --url суперечать одне одному: --url — це мережа")

    print(LICENCE)
    s = session(args.workers)
    src = open_source(args, s)
    print(f"[index] {len(src.entries)} потрібних файлів у джерелі ({src.label[:90]})")

    if args.list:
        print_list(src, peek_map(src))
        return
    if args.raw16 is not None:
        failed = raw16(src, args.raw16, args.splits or RAW16_SPLITS, args.raw16_dir, args.workers)
    else:
        splits = list(SPLITS) if args.full else (args.splits or DEFAULT_SPLITS)
        if args.pairs and set(splits) - {"video"}:
            if args.splits:
                print("  --pairs: офіційні пари є лише у video; train / val пропускаю")
            splits = ["video"]
        step = args.step or (1 if args.full or args.annotations_only else DEFAULT_STEP)
        mapping = ensure_sidecars(src, args.out, s, args.offline, need_map=args.pairs)
        rels = select(src.entries, splits, args.modality, step, args.annotations_only, not args.no_tiff,
                      args.index or args.full, args.pairs, mapping)
        if not rels:
            sys.exit("Під вибір не потрапив жоден файл — перевірте --splits / --modality / --step (див. --list)")
        print(f"[select] {', '.join(splits)}, {'пари' if args.pairs else args.modality}, кожен {step}-й кадр"
              + (", лише розмітка" if args.annotations_only else ""))
        failed = copy_all(src, rels, args.out, args.workers)
        if not args.no_tiff and not args.annotations_only:
            warn_not16(rels, args.out)
    if failed:
        for f, e in failed[:10]:
            print(f"[fail] {f}: {e}")
        sys.exit(f"Не завантажено {len(failed)} файлів — перезапустіть скрипт, готові файли буде пропущено")
    print(f"Готово -> {args.raw16_dir if args.raw16 is not None else args.out}")


if __name__ == "__main__":
    main()
