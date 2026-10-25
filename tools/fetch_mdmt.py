#!/usr/bin/env python
"""
MDMT (Multi-Drone Multi-Target tracking) для заняття 9 — в artifacts/mdmt/.

Два дрони одночасно знімають ту саму сцену (1920×1080); об'єкт має той самий ID в обох потоках.
Звідси: MOT з рухомою камерою, SOT, ReID-пари «той самий / інший», крос-камерна асоціація.

    python tools/fetch_mdmt.py --list                 # які послідовності є і скільки важать (нічого не качає)
    python tools/fetch_mdmt.py                        # пари 46 і 48: ДЕМО 02 §7 і ДЗ 9, ≈ 0.9 ГБ
    python tools/fetch_mdmt.py --seqs 22 36 49        # конкретні пари (обидва дрони + розмітка)
    python tools/fetch_mdmt.py --all-pairs            # усі 17 повних пар val + test, ≈ 11 ГБ

Джерело — дзеркало на Hugging Face (huggingface.co/datasets/mrhwl123/MDMT): один zip на 16,7 ГБ.
Скрипт читає з нього лише потрібні файли через HTTP Range, тож повний архів качати не треба.
Офіційне джерело — Baidu Netdisk з github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking (25,8 ГБ).
У дзеркалі немає кадрів другого дрона для train, тому повні пари є лише у val і test.

Ліцензію автори не вказали: використання — академічне, з посиланням на статтю
Liu et al., «Robust Multi-Drone Multi-Target Tracking to Resolve Target Occlusion: A Benchmark», IEEE TMM 2023.

Структура на диску (як в архіві):
    artifacts/mdmt/<split>/<drone>/<seq>-<drone>/00000001.jpg ...
    artifacts/mdmt/new_xml/<drone>/<seq>-<drone>.xml

Уже завантажені файли пропускаються, тож скрипт можна перезапускати.
Запуск з кореня репозиторію.
"""
import argparse
import io
import re
import struct
import sys
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
URL = "https://huggingface.co/datasets/mrhwl123/MDMT/resolve/main/Multi-Drone-Multi-Object-Detection-and-Tracking.zip"
ROOT = "Multi-Drone-Multi-Object-Detection-and-Tracking/"
DEFAULT_SEQS = ["46", "48"]
FRAME_RE = re.compile(r"^(train|val|test)/([12])/(\d+)-\2/[^/]+\.jpg$")
XML_RE = re.compile(r"^new_xml/([12])/(\d+)-\1\.xml$")


class HttpFile(io.RawIOBase):
    """Файл на HTTP із довільним доступом через Range — щоб zipfile прочитав каталог архіву."""

    def __init__(self, url: str, session: requests.Session):
        self.s = session
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        r.raise_for_status()
        b[: len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def read_index(session):
    """Каталог архіву (кілька запитів, без завантаження даних): url, розмір, список ZipInfo."""
    f = HttpFile(URL, session)
    with zipfile.ZipFile(io.BufferedReader(f, 1 << 20)) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.startswith(ROOT)]
    return f.url, f.size, infos


def group_sequences(infos):
    """seq -> {'split', 'frames': {drone: [ZipInfo]}, 'xml': {drone: ZipInfo}}"""
    seqs = defaultdict(lambda: {"split": None, "frames": defaultdict(list), "xml": {}})
    for i in infos:
        rel = i.filename[len(ROOT):]
        if m := FRAME_RE.match(rel):
            split, drone, seq = m.groups()
            seqs[seq]["split"] = split
            seqs[seq]["frames"][drone].append(i)
        elif m := XML_RE.match(rel):
            drone, seq = m.groups()
            seqs[seq]["xml"][drone] = i
    return seqs


def is_pair(s) -> bool:
    return all(s["frames"].get(d) for d in "12") and all(d in s["xml"] for d in "12")


def fetch_member(session, url, info: zipfile.ZipInfo, dest: Path):
    """Один Range-запит: локальний заголовок + дані; розпаковуємо, перевіряємо CRC, пишемо атомарно."""
    if dest.exists() and dest.stat().st_size == info.file_size:
        return 0
    start = info.header_offset
    end = start + 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size  # 1024 — запас на extra-поле
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    buf = r.content
    sig, *_, n_len, x_len = struct.unpack("<IHHHHHIIIHH", buf[:30])
    if sig != 0x04034B50:
        raise IOError(f"{info.filename}: неочікуваний локальний заголовок")
    raw = buf[30 + n_len + x_len: 30 + n_len + x_len + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise IOError(f"{info.filename}: метод стиснення {info.compress_type} не підтримано")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise IOError(f"{info.filename}: CRC або розмір не збігаються")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def download(session, url, members, out_dir: Path, workers: int):
    total = sum(i.file_size for i in members)
    done_bytes, n_done, failed = 0, 0, []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_member, session, url, i, out_dir / i.filename[len(ROOT):]): i for i in members}
        for fut in as_completed(futs):
            info = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((info.filename, str(e)[:200]))
            done_bytes += info.file_size
            n_done += 1
            if n_done % 50 == 0 or n_done == len(members):
                print(f"\r  {n_done}/{len(members)} файлів, {done_bytes / 2**30:.2f}/{total / 2**30:.2f} ГБ",
                      end="", flush=True)
    print()
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="показати послідовності й розміри, нічого не качати")
    ap.add_argument("--seqs", nargs="+", default=DEFAULT_SEQS, help="номери пар, напр. 46 48")
    ap.add_argument("--all-pairs", action="store_true", help="усі пари, де є обидва дрони й розмітка")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "mdmt")
    ap.add_argument("--workers", type=int, default=16, help="паралельних запитів")
    args = ap.parse_args()

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.workers, max_retries=3))
    print("[index] читаю каталог архіву...")
    url, size, infos = read_index(session)
    seqs = group_sequences(infos)
    print(f"[index] {len(infos)} файлів, архів {size / 2**30:.1f} ГБ")

    if args.list:
        print(f"{'seq':>4} {'split':>5} {'кадри д1':>9} {'кадри д2':>9} {'ГБ':>6}  пара")
        for k in sorted(seqs, key=int):
            s = seqs[k]
            gb = sum(i.file_size for d in s["frames"].values() for i in d) / 2**30
            n1, n2 = len(s["frames"].get("1", [])), len(s["frames"].get("2", []))
            print(f"{k:>4} {s['split'] or '-':>5} {n1:>9} {n2:>9} {gb:6.2f}  {'так' if is_pair(s) else 'ні'}")
        return

    wanted = sorted((k for k, s in seqs.items() if is_pair(s)), key=int) if args.all_pairs else args.seqs
    members = []
    for k in wanted:
        s = seqs.get(k)
        if not s or not is_pair(s):
            sys.exit(f"Пара {k} неповна або відсутня в дзеркалі. Див. --list")
        members += [i for d in "12" for i in s["frames"][d]] + [s["xml"][d] for d in "12"]
        n = {d: len(s["frames"][d]) for d in "12"}
        print(f"[seq {k}] {s['split']}: дрон 1 — {n['1']} кадрів, дрон 2 — {n['2']} кадрів")

    print(f"[download] {len(members)} файлів, {sum(i.file_size for i in members) / 2**30:.2f} ГБ -> {args.out}")
    failed = download(session, url, members, args.out, args.workers)
    if failed:
        for name, err in failed[:10]:
            print(f"[fail] {name}: {err}")
        sys.exit(f"Не завантажено {len(failed)} файлів — перезапустіть скрипт, готові файли буде пропущено")
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
