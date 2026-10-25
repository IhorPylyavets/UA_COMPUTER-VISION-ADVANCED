"""mdmt.py — MDMT (CVAT XML) → кадри для ноутбуків і формат MOTChallenge для TrackEval.

Дані: python tools/fetch_mdmt.py --seqs 46 48  (лягають в artifacts/mdmt/; 48 — демо 1 §7, 46 і 48 — ДЗ 9).

У ноутбуці:
    from mdmt import load_seq, run_trackeval, export_trackeval, write_tracker
    frames = load_seq("46", drone=1)            # [[(id, cx, cy, w, h), ...] для кожного кадру] — як make_gt у демо 1
    run_trackeval({"46-1": frames}, {"BYTE": {"46-1": hyp}})   # HOTA / DetA / AssA / MOTA / IDF1 одразу зі списків
    export_trackeval(["46", "48"])              # або GT у структурі TrackEval на диску — для запуску TrackEval окремо
    write_tracker(hyp, "46", 1, "BYTE")         # hyp = [(t, id, cx, cy, w, h), ...] — вихід run_tracker

З командного рядка (з кореня репозиторію):
    python lessons/09-mot-reid/mdmt.py --seqs 46 48                 # усі класи як один клас
    python lessons/09-mot-reid/mdmt.py --seqs 46 --classes car      # лише авто

Структура TrackEval (MotChallenge2DBox, BENCHMARK=MDMT, SPLIT_TO_EVAL=all):
    artifacts/mdmt/trackeval/gt/mot_challenge/MDMT-all/46-1/{gt/gt.txt, seqinfo.ini}
    artifacts/mdmt/trackeval/gt/mot_challenge/seqmaps/MDMT-all.txt
    artifacts/mdmt/trackeval/trackers/mot_challenge/MDMT-all/<трекер>/data/46-1.txt

TrackEval оцінює лише клас pedestrian (id 1), тому всі вибрані класи записуємо з class = 1:
для окремої оцінки авто чи людей експортуйте з --classes. Прапорець occluded з MDMT
пишемо в колонку visibility (1 — видно, 0 — перекрито); TrackEval її не використовує.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "mdmt"
CLASSES = ("car", "person", "bicycle")
FPS = 30  # у статті не вказано; потрібен лише для seqinfo.ini
W, H = 1920, 1080


def seq_name(seq: str, drone: int) -> str:
    return f"{seq}-{drone}"


def frames_dir(seq: str, drone: int, root: Path = ROOT) -> Path:
    """Тека з кадрами (спліт — train / val / test — визначаємо за наявністю)."""
    for split in ("val", "test", "train"):
        d = root / split / str(drone) / seq_name(seq, drone)
        if d.is_dir():
            return d
    raise FileNotFoundError(f"Немає кадрів {seq_name(seq, drone)} у {root}. Запустіть tools/fetch_mdmt.py --seqs {seq}")


_FRAMES: Dict[Path, List[Path]] = {}


def frame_list(seq: str, drone: int, root: Path = ROOT) -> List[Path]:
    """Кадри послідовності за порядком. Нумерація файлів не завжди з 1 (у 22-1 — з 00000201.jpg),
    а розмітка завжди з 0, тож кадр t — це t-й файл у відсортованому списку."""
    d = frames_dir(seq, drone, root)
    if d not in _FRAMES:
        _FRAMES[d] = sorted(d.glob("*.jpg"))
    return _FRAMES[d]


def frame_path(seq: str, drone: int, t: int, root: Path = ROOT) -> Path:
    """Кадр t (з нуля, як у розмітці)."""
    return frame_list(seq, drone, root)[t]


def load_boxes(seq: str, drone: int, classes: Optional[Iterable[str]] = None, root: Path = ROOT):
    """Усі бокси послідовності: список (t, id, x1, y1, w, h, label, occluded), t — з нуля."""
    xml = root / "new_xml" / str(drone) / f"{seq_name(seq, drone)}.xml"
    keep = set(classes) if classes else None
    out = []
    for tr in ET.parse(xml).getroot().iter("track"):
        label = tr.get("label")
        if keep is not None and label not in keep:
            continue
        tid = int(tr.get("id"))
        for b in tr.iter("box"):
            if b.get("outside") == "1":
                continue
            x1, y1, x2, y2 = (float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            out.append((int(b.get("frame")), tid, x1, y1, x2 - x1, y2 - y1, label, b.get("occluded") == "1"))
    return out


def load_seq(seq: str, drone: int, classes: Optional[Iterable[str]] = None,
             root: Path = ROOT) -> List[List[Tuple[int, float, float, float, float]]]:
    """GT у форматі make_gt з демо 1: для кожного кадру список (id, cx, cy, w, h)."""
    boxes = load_boxes(seq, drone, classes, root)
    n = len(list(frames_dir(seq, drone, root).glob("*.jpg")))
    frames: List[list] = [[] for _ in range(n)]
    for t, tid, x, y, w, h, *_ in boxes:
        if t < n:
            frames[t].append((tid, x + w / 2, y + h / 2, w, h))
    return frames


def _benchmark(classes: Optional[Sequence[str]]) -> str:
    return "MDMT" if not classes else "MDMT_" + "_".join(sorted(classes))


def export_trackeval(seqs: Sequence[str], drones: Sequence[int] = (1, 2), classes: Optional[Sequence[str]] = None,
                     root: Path = ROOT, split: str = "all") -> Path:
    """GT кожної пари seq × drone у структурі MotChallenge2DBox. Повертає теку gt/mot_challenge."""
    bench = _benchmark(classes)
    gt_root = root / "trackeval" / "gt" / "mot_challenge"
    names = []
    for seq in seqs:
        for d in drones:
            name = seq_name(seq, d)
            n = len(list(frames_dir(seq, d, root).glob("*.jpg")))
            sdir = gt_root / f"{bench}-{split}" / name
            (sdir / "gt").mkdir(parents=True, exist_ok=True)
            rows = sorted(load_boxes(seq, d, classes, root), key=lambda r: (r[0], r[1]))
            with open(sdir / "gt" / "gt.txt", "w") as f:
                for t, tid, x, y, w, h, _, occ in rows:
                    # TrackEval вимагає id ≥ 1, а в MDMT вони з нуля
                    f.write(f"{t + 1},{tid + 1},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,1,{0 if occ else 1}\n")
            (sdir / "seqinfo.ini").write_text(
                f"[Sequence]\nname={name}\nimDir=img1\nframeRate={FPS}\nseqLength={n}\n"
                f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n")
            names.append(name)
    (gt_root / "seqmaps").mkdir(parents=True, exist_ok=True)
    (gt_root / "seqmaps" / f"{bench}-{split}.txt").write_text("name\n" + "\n".join(names) + "\n")
    return gt_root


def write_tracker(hyp: Iterable[Sequence[float]], seq: str, drone: int, tracker: str,
                  classes: Optional[Sequence[str]] = None, root: Path = ROOT, split: str = "all") -> Path:
    """Вихід трекера [(t, id, cx, cy, w, h), ...] з t від нуля -> data/<seq>-<drone>.txt."""
    d = root / "trackeval" / "trackers" / "mot_challenge" / f"{_benchmark(classes)}-{split}" / tracker / "data"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{seq_name(seq, drone)}.txt"
    with open(path, "w") as f:
        for t, tid, cx, cy, w, h, *rest in sorted(hyp, key=lambda r: (r[0], r[1])):
            conf = rest[0] if rest else 1
            f.write(f"{int(t) + 1},{int(tid)},{cx - w / 2:.1f},{cy - h / 2:.1f},{w:.1f},{h:.1f},{conf:.3f},-1,-1,-1\n")
    return path


def trackeval_args(classes: Optional[Sequence[str]] = None, root: Path = ROOT, split: str = "all") -> Dict:
    """Конфіг для trackeval.datasets.MotChallenge2DBox під цю структуру."""
    return dict(GT_FOLDER=str(root / "trackeval" / "gt" / "mot_challenge"),
                TRACKERS_FOLDER=str(root / "trackeval" / "trackers" / "mot_challenge"),
                BENCHMARK=_benchmark(classes), SPLIT_TO_EVAL=split, DO_PREPROC=False)


def _import_trackeval():
    """import trackeval без шуму: TrackEval друкує помилки необов'язкових залежностей (BURST, pycocotools)
    і досі звертається до np.float / np.int, прибраних у numpy 1.24."""
    import contextlib
    import io

    import numpy as np
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval
    return trackeval


def has_trackeval() -> bool:
    try:
        _import_trackeval()
        return True
    except ImportError:
        return False


def run_trackeval(gt: Dict[str, list], hyps: Dict[str, Dict[str, list]], run: str = "demo",
                  fps: Optional[Dict[str, float]] = None, root: Path = ROOT) -> Dict[str, Dict[str, float]]:
    """HOTA / DetA / AssA / MOTA / IDF1 / IDSW через офіційний TrackEval, усі послідовності разом.

    gt   = {назва: кадри у форматі load_seq}            — назва без пробілів, напр. "48-1_fps10"
    hyps = {трекер: {назва: [(t, id, cx, cy, w, h), ...]}} — вихід run_tracker
    Файли пишуться в artifacts/mdmt/trackeval/runs/<run>/ і перезаписуються при кожному виклику."""
    import contextlib
    import io
    import shutil

    trackeval = _import_trackeval()
    base = root / "trackeval" / "runs" / run
    shutil.rmtree(base, ignore_errors=True)
    gt_dir, tr_dir = base / "gt" / "L9-all", base / "trackers" / "L9-all"
    for name, frames in gt.items():
        (gt_dir / name / "gt").mkdir(parents=True, exist_ok=True)
        with open(gt_dir / name / "gt" / "gt.txt", "w") as f:
            for t, objs in enumerate(frames):
                for tid, cx, cy, w, h in objs:
                    f.write(f"{t + 1},{tid + 1},{cx - w / 2:.1f},{cy - h / 2:.1f},{w:.1f},{h:.1f},1,1,1\n")
        (gt_dir / name / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nframeRate={(fps or {}).get(name, FPS)}\nseqLength={len(frames)}\n"
            f"imWidth={W}\nimHeight={H}\n")
    (base / "gt" / "seqmaps").mkdir(parents=True, exist_ok=True)
    (base / "gt" / "seqmaps" / "L9-all.txt").write_text("name\n" + "\n".join(gt) + "\n")
    for tracker, per_seq in hyps.items():
        (tr_dir / tracker / "data").mkdir(parents=True, exist_ok=True)
        for name, hyp in per_seq.items():
            with open(tr_dir / tracker / "data" / f"{name}.txt", "w") as f:
                for t, tid, cx, cy, w, h, *_ in sorted(hyp, key=lambda r: (r[0], r[1])):
                    f.write(f"{int(t) + 1},{int(tid)},{cx - w / 2:.1f},{cy - h / 2:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")

    with contextlib.redirect_stdout(io.StringIO()):  # TrackEval друкує конфіги й прогрес кожної послідовності
        ev = trackeval.Evaluator(dict(trackeval.Evaluator.get_default_eval_config(), PRINT_CONFIG=False,
                                      PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
                                      PLOT_CURVES=False, USE_PARALLEL=False, TIME_PROGRESS=False))
        ds = trackeval.datasets.MotChallenge2DBox(dict(
            trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(), PRINT_CONFIG=False,
            GT_FOLDER=str(base / "gt"), TRACKERS_FOLDER=str(base / "trackers"), BENCHMARK="L9",
            SPLIT_TO_EVAL="all", TRACKERS_TO_EVAL=list(hyps), DO_PREPROC=False))
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        res, _ = ev.evaluate([ds], metrics)
    out = {}
    for tracker in hyps:
        per_seq = res["MotChallenge2DBox"][tracker]
        out[tracker] = {}
        for name in list(gt) + ["COMBINED_SEQ"]:
            c = per_seq[name]["pedestrian"]
            out[tracker][name] = dict(HOTA=float(c["HOTA"]["HOTA"].mean()), DetA=float(c["HOTA"]["DetA"].mean()),
                                      AssA=float(c["HOTA"]["AssA"].mean()), MOTA=float(c["CLEAR"]["MOTA"]),
                                      IDF1=float(c["Identity"]["IDF1"]), IDSW=int(c["CLEAR"]["IDSW"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MDMT XML -> MOTChallenge / TrackEval")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--drones", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--classes", nargs="+", choices=CLASSES, help="за замовчуванням — усі як один клас")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    out = export_trackeval(a.seqs, a.drones, a.classes, a.root)
    print(f"GT: {out}\nTrackEval: BENCHMARK={_benchmark(a.classes)} SPLIT_TO_EVAL=all GT_FOLDER={out}")
