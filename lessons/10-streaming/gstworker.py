"""gstworker.py — GStreamer-частина streamlab, яка працює в окремому процесі.

Чому окремий процес: pip-колеса numpy / matplotlib / pyzmq (ядро Jupyter) і системні бібліотеки GStreamer
(libsrt, freetype тощо) на частині збірок конфліктують у одному процесі — падіння SIGSEGV/SIGABRT.
Тут GStreamer завантажується першим, ноутбук не імпортує gi взагалі, а результати повертаються через pickle.
"""
from __future__ import annotations

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo  # noqa: E402
Gst.init(None)
for _name in ('srtsrc', 'srtsink'):
    Gst.ElementFactory.make(_name, None)

import heapq  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

from streamcore import Scene, Impair, make_scene  # noqa: E402,F401


class _Pipe:
    """Одна дорога: черга з ємністю, затримкою, джиттером і втратами."""

    def __init__(self, imp: Impair, out_sock: socket.socket, t0: float, seed: int):
        self.imp, self.sock, self.t0 = imp, out_sock, t0
        self.rnd = random.Random(seed)
        self.heap: list = []
        self.cv = threading.Condition()
        self.last_dep = 0.0
        self.last_arr = 0.0
        self.bad = False
        self.dest = None
        self.stats = dict(rx=0, lost=0, tail=0, sent=0)
        self.on_drop = None
        self.run = True
        threading.Thread(target=self._tx, daemon=True).start()

    def push(self, data: bytes, dest) -> None:
        imp, now = self.imp, time.perf_counter()
        self.dest = dest
        self.stats['rx'] += 1
        ts = now - self.t0
        drop = any(a <= ts < b for a, b in imp.blackout)
        if imp.ge:
            p_gb, p_bg, lb = imp.ge
            self.bad = (self.rnd.random() < p_gb) if not self.bad else (self.rnd.random() >= p_bg)
            drop = drop or (self.bad and self.rnd.random() < lb)
        drop = drop or self.rnd.random() < imp.loss
        if drop:
            self.stats['lost'] += 1
            if self.on_drop:
                self.on_drop(ts)
            return
        dep = now
        if imp.rate_mbps > 0:
            start = max(now, self.last_dep)
            if (start - now) * 1000 > imp.queue_limit_ms:
                self.stats['tail'] += 1
                return
            dep = start + len(data) * 8 / (imp.rate_mbps * 1e6)
            self.last_dep = dep
        arr = dep + imp.delay_ms / 1000 + max(0.0, self.rnd.gauss(0, imp.jitter_ms)) / 1000
        if imp.keep_order:
            arr = max(arr, self.last_arr)
        self.last_arr = max(self.last_arr, arr)
        with self.cv:
            heapq.heappush(self.heap, (arr, self.stats['rx'], data, dest))
            self.cv.notify()

    def _tx(self) -> None:
        while self.run:
            with self.cv:
                while self.run and not self.heap:
                    self.cv.wait(0.05)
                if not self.run:
                    return
                arr, _, data, dest = self.heap[0]
                wait = arr - time.perf_counter()
                if wait > 0:
                    self.cv.wait(min(wait, 0.002))
                    continue
                heapq.heappop(self.heap)
            try:
                self.sock.sendto(data, dest)
                self.stats['sent'] += 1
            except OSError:
                pass


class LinkEmulator:
    """UDP-relay: клієнт → listen_port → target_port; відповіді йдуть назад клієнту.

    forward — погіршення в напрямку клієнт → ціль (відео), reverse — у зворотному (ACK/NAK SRT, RTCP).
    """

    def __init__(self, listen_port: int, target_port: int, forward: Impair, reverse: Optional[Impair] = None,
                 host: str = '127.0.0.1', seed: int = 1):
        self.t0 = time.perf_counter()
        self.a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.a.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.a.bind((host, listen_port))
        self.b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.b.bind((host, 0))
        self.target = (host, target_port)
        self.client = None
        self.fwd = _Pipe(forward, self.b, self.t0, seed)
        self.rev = _Pipe(reverse or Impair(), self.a, self.t0, seed + 1)
        self.run = True
        threading.Thread(target=self._rx_a, daemon=True).start()
        threading.Thread(target=self._rx_b, daemon=True).start()

    def _rx_a(self):
        self.a.settimeout(0.1)
        while self.run:
            try:
                data, addr = self.a.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            self.client = addr
            self.fwd.push(data, self.target)

    def _rx_b(self):
        self.b.settimeout(0.1)
        while self.run:
            try:
                data, _ = self.b.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if self.client:
                self.rev.push(data, self.client)

    def close(self):
        self.run = False
        self.fwd.run = self.rev.run = False
        time.sleep(0.15)
        self.a.close(); self.b.close()

    @property
    def stats(self):
        return dict(forward=dict(self.fwd.stats), reverse=dict(self.rev.stats))


# ============================================================== відправник
class Sender:
    """appsrc → videoconvert → x264enc → (RTP/UDP | MPEG-TS/SRT). Штовхає кадри Scene у реальному часі."""

    def __init__(self, scene: Scene, port: int, enc: str = 'tune=zerolatency speed-preset=superfast bitrate=1000',
                 transport: str = 'rtp', srt_latency_ms: int = 120, mtu: int = 1200):
        self.scene, self.T = scene, 1.0 / scene.fps
        caps = f'video/x-raw,format=RGB,width={scene.w},height={scene.h},framerate={scene.fps}/1'
        if transport == 'rtp':
            sink = f'rtph264pay config-interval=-1 pt=96 mtu={mtu} ! udpsink host=127.0.0.1 port={port} sync=false async=false'
        else:
            sink = (f'h264parse config-interval=-1 ! mpegtsmux alignment=7 ! '
                    f'srtsink uri="srt://127.0.0.1:{port}?mode=caller" latency={srt_latency_ms} sync=false')
        desc = (f'appsrc name=src is-live=true format=time do-timestamp=false caps="{caps}" ! videoconvert ! '
                f'x264enc name=enc {enc} ! {sink}')
        self.p = Gst.parse_launch(desc)
        self.src = self.p.get_by_name('src')
        self.enc = self.p.get_by_name('enc')
        self.t_send: Dict[int, float] = {}
        self.sizes: Dict[int, tuple] = {}            # idx → (байти, ключовий)
        self.pts0 = None
        self.enc.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, self._probe)
        self.run = False
        self.idx = 0

    def _probe(self, pad, info):
        b = info.get_buffer()
        if b.pts == Gst.CLOCK_TIME_NONE:
            return Gst.PadProbeReturn.OK
        if self.pts0 is None:          # x264enc зсуває PTS на 1000 год — рахуємо від першого кадру (без B-кадрів)
            self.pts0 = b.pts
        idx = int(round((b.pts - self.pts0) / Gst.SECOND / self.T))
        self.sizes[idx] = (b.get_size(), not b.has_flags(Gst.BufferFlags.DELTA_UNIT))
        return Gst.PadProbeReturn.OK

    def force_key_unit(self):
        """IDR на вимогу — те, що робить PLI/FIR від приймача."""
        ev = GstVideo.video_event_new_upstream_force_key_unit(Gst.CLOCK_TIME_NONE, True, 0)
        self.enc.get_static_pad('src').send_event(ev)

    def start(self, seconds: float):
        self.p.set_state(Gst.State.PLAYING)
        self.run = True
        self.th = threading.Thread(target=self._loop, args=(seconds,), daemon=True)
        self.th.start()

    def _loop(self, seconds):
        t0 = time.perf_counter()
        n = int(seconds * self.scene.fps)
        for i in range(n):
            if not self.run:
                break
            due = t0 + i * self.T
            while time.perf_counter() < due:
                time.sleep(min(0.002, max(0.0, due - time.perf_counter())))
            img = self.scene.frame(i)
            buf = Gst.Buffer.new_wrapped(img.tobytes())
            buf.pts = int(i * self.T * Gst.SECOND)
            buf.duration = int(self.T * Gst.SECOND)
            self.t_send[i] = time.perf_counter()
            self.src.emit('push-buffer', buf)
            self.idx = i
        self.src.emit('end-of-stream')

    def wait(self):
        self.th.join()

    def stop(self):
        self.run = False
        self.p.set_state(Gst.State.NULL)


# ============================================================== приймач
class Receiver:
    """(udpsrc → rtpjitterbuffer → depay | srtsrc → tsdemux) → avdec_h264 → RGB → appsink."""

    def __init__(self, scene: Scene, port: int, transport: str = 'rtp', jb_ms: int = 20,
                 max_buffers: int = 1, drop: bool = True, srt_latency_ms: int = 120):
        self.scene = scene
        if transport == 'rtp':
            src = (f'udpsrc port={port} buffer-size=4194304 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! '
                   f'rtpjitterbuffer latency={jb_ms} drop-on-latency=true ! rtph264depay ! h264parse')
        else:
            src = (f'srtsrc uri="srt://127.0.0.1:{port}?mode=listener" latency={srt_latency_ms} ! queue ! '
                   f'tsdemux ! h264parse')
        sinkopts = f'max-buffers={max_buffers} drop={"true" if drop else "false"}'
        desc = (f'{src} ! avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! '
                f'appsink name=sink sync=false {sinkopts}')
        self.p = Gst.parse_launch(desc)
        self.sink = self.p.get_by_name('sink')
        self.records: List[dict] = []
        self.run = False

    def start(self):
        self.p.set_state(Gst.State.PLAYING)

    def pull(self, t_send: Dict[int, float], seconds: float, work_s: float = 0.0, with_psnr: bool = False, keep_every: int = 0):
        """Забирає кадри; work_s — імітація «детектора», що тримає потік."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            s = self.sink.emit('try-pull-sample', int(0.05 * Gst.SECOND))
            if s is None:
                continue
            t_rx = time.perf_counter()
            st = s.get_caps().get_structure(0)
            w, h = st.get_value('width'), st.get_value('height')
            buf = s.get_buffer()
            ok, m = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            stride = len(m.data) // h                     # рядки RGB вирівняні до 4 байтів
            img = np.ndarray((h, w, 3), np.uint8, m.data, strides=(stride, 3, 1))
            idx = self.scene.read(img)
            rec = dict(idx=idx, t_rx=t_rx, age=(t_rx - t_send[idx]) * 1000 if idx in t_send else np.nan)
            if with_psnr and idx not in t_send:
                rec['psnr'], rec['bad'] = np.nan, 1.0      # номер не читається — кадр зіпсований
            if keep_every and idx in t_send and idx % keep_every == 0:
                rec['img'] = img.copy()                   # декодований кадр для офлайн-аналізу
            if with_psnr and idx in t_send:
                rec['psnr'] = self.scene.psnr(img, idx)
                rec['bad'] = self.scene.corrupt(img, idx)
            buf.unmap(m)
            self.records.append(rec)
            if work_s:
                time.sleep(work_s)
        return self.records

    def stop(self):
        self.p.set_state(Gst.State.NULL)


# ============================================================== допоміжне
def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_stream(scene: Scene, seconds: float, enc: str, transport: str = 'rtp', impair: Optional[Impair] = None,
               reverse: Optional[Impair] = None, jb_ms: int = 20, srt_latency_ms: int = 120, max_buffers: int = 1,
               drop: bool = True, work_s: float = 0.0, with_psnr: bool = False, on_drop=None, seed: int = 1,
               settle_s: float = 0.6, on_start=None, keep_every: int = 0):
    """Повний прогін: відправник → (емулятор) → приймач. Повертає (records, sender, link)."""
    rx_port = free_port()
    link = None
    tx_port = rx_port
    if impair is not None:
        tx_port = free_port()
        link = LinkEmulator(tx_port, rx_port, impair, reverse, seed=seed)
        if on_drop:
            link.fwd.on_drop = on_drop
    rx = Receiver(scene, rx_port, transport, jb_ms, max_buffers, drop, srt_latency_ms)
    rx.start()
    time.sleep(0.3)
    tx = Sender(scene, tx_port, enc, transport, srt_latency_ms)
    tx.start(seconds)
    if link:                       # час емулятора рахуємо від старту відправника — так задаються серії втрат
        link.t0 = link.fwd.t0 = link.rev.t0 = time.perf_counter()
    if on_start:
        on_start(tx)
    rec = rx.pull(tx.t_send, seconds + settle_s, work_s, with_psnr, keep_every)
    tx.stop(); rx.stop()
    if link:
        link.close()
    return rec, tx, link


class PLI:
    """Імітація PLI: після останньої втрати і паузи quiet_s приймач просить IDR, запит іде RTT/2, а відповідь — I-кадр.

    Таймер перезапускається кожною втратою, тож IDR не загубиться всередині тієї самої серії.
    """

    def __init__(self, rtt_s: float = 0.08, quiet_s: float = 0.03):
        self.rtt, self.quiet, self.tx, self.timer, self.sent = rtt_s, quiet_s, None, None, []

    def on_start(self, tx):
        self.tx = tx

    def on_drop(self, ts):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.quiet + self.rtt / 2, self._fire)
        self.timer.start()

    def _fire(self):
        if self.tx:
            self.sent.append(time.perf_counter())
            self.tx.force_key_unit()


def info() -> dict:
    f = Gst.ElementFactory.make('appsink', None)
    return dict(version=Gst.version_string(), leaky_type=f.find_property('leaky-type') is not None,
                srt=all(Gst.ElementFactory.find(n) is not None for n in ('srtsrc', 'srtsink')))


def main() -> None:
    job = pickle.load(sys.stdin.buffer)
    if job.get('cmd') == 'info':
        out = info()
    else:
        sc = make_scene(job.pop('scene'))
        pli_rtt = job.pop('pli_rtt', None)
        pli = PLI(rtt_s=pli_rtt) if pli_rtt else None
        rec, tx, link = run_stream(sc, on_drop=pli.on_drop if pli else None, on_start=pli.on_start if pli else None, **job)
        out = dict(rec=rec, t_send=tx.t_send, sizes=tx.sizes, stats=link.stats if link else None,
                   pli_sent=len(pli.sent) if pli else 0)
    sys.stdout.buffer.write(pickle.dumps(out))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
