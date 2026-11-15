"""Copperline backend: launching it headless and driving it over its control
protocol (CCP, docs/debugger/control.md in the Copperline repository).

Selected with ``run --copperline BIN``. What differs from the Amiberry path:

- **Config.** A matrix entry's machine (model, chipset, CPU, FPU, memory,
  video standard) is written as a Copperline TOML file. Amiberry's engine
  options (fast / compatible / cycle-exact core, threaded CPU, softfloat) have
  no counterpart: Copperline has one cycle-driven core. ``cachesize`` > 0
  maps to its optional JIT (``[cpu] jit = true``).
- **Execution.** ``--control`` starts a headless, unpaced machine that only
  runs while a resume verb is pending. The backend advances it a few frames
  at a time (``step_frame``) and reads both transports between the steps, so
  a launch is driven by emulated time alone: the same disk and config give
  the same records on every host and under any host load. Timeouts and the
  hang limit are therefore emulated seconds here (a wall-clock limit of
  ``WALL_FACTOR`` times ``--timeout`` only guards against a stuck emulator
  process).
- **Serial.** ``[serial] mode = "stdout"``: Paula's serial output is the
  process's stdout, redirected to ``raw/<entry>/<disk>/serial.bin``.
- **Mailbox and acks.** ``mem.read`` / ``mem.write`` (CONTRACT sections 7, 13).
- **Host requests.** ``key`` -> ``input.key``, ``mouse`` -> ``input.mouse``,
  ``joy`` and ``cd32pad`` -> ``input.joy`` (CCP has joystick and pad
  injection), ``screenshot`` -> ``capture.screenshot``, ``pcmcia`` ->
  ``pcmcia.eject`` / ``pcmcia.insert``, ``pacing`` -> acked as "not warped"
  (the emulated machine does not see host pacing), ``audio_mark`` -> refused
  (no capture yet), ``wait`` -> acked.
- **Not done here.** hostspeed (a host real-time check: module bit 26 is
  masked off), narrowing and repro packs (their neighbour configs and
  ``config.uae`` are Amiberry's), ``--dump-config`` validation, CD boot.
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import signal
import socket
import subprocess
import time

from . import adf as adfmod
from . import emulator, envinfo, hostreq, mailbox, matrix, records
from .emulator import RunnerError

HOSTSPEED_BIT = 1 << 26
WALL_FACTOR = 6.0        # wall-clock guard: this many host seconds per emulated second of --timeout
STEP_FRAMES = 2          # frames per step while nothing is pending
DONE_GRACE = 2.0         # emulated seconds to wait for the slower transport after a done record
SCAN_INTERVAL = 0.5      # emulated seconds between chip RAM scans for the mailbox
BOOT_LIMIT = 300.0       # emulated seconds a disk may take to send its first record (ROM boot + payload load)
RTC_SEED = "2026-03-14 15:09:26"   # battery clock at power-on (emulated time from there)

MODELS = {"A500": "A500", "A500+": "A500Plus", "A600": "A600", "A1200": "A1200", "A3000": "A3000",
          "A4000": "A4000", "A4000T": "A4000", "A1000": "A1000", "A2000": "A500", "AROS-AGA": "A1200",
          "AROS-ECS": "A500", "CD32": "CD32", "CDTV": "CDTV"}
CPUS = ("68000", "68010", "68EC020", "68020", "68EC030", "68030", "68EC040", "68040", "68EC060", "68060")


class Unsupported(RunnerError):
    """The entry describes a machine Copperline cannot be configured as."""


# ---------------------------------------------------------------------------
# matrix
def usable(entry):
    """(ok, why not). Amiberry's threaded-CPU twins, its hostspeed-only
    entries and its "hardware instruction policy" twins have no counterpart:
    Copperline has one core, always with the hardware's trap behaviour."""
    if entry.get("twin_kind") == "threaded" or entry.get("jit_variant") == "threaded":
        return False, "threaded CPU twin (an Amiberry engine option)"
    if entry["id"].endswith("-strict"):
        return False, "instruction-policy twin: Copperline always runs the hardware policy"
    if entry.get("twin_kind") in ("softfloat",):
        return False, "softfloat twin (an Amiberry engine option)"
    try:
        mask = int(str((entry.get("params") or {}).get("modules", "-1")), 0)
    except ValueError:
        mask = -1
    if mask == HOSTSPEED_BIT:
        return False, "hostspeed-only entry (host real-time check; Copperline's headless machine is unpaced)"
    if entry.get("boot") == "cd":
        return False, "CD boot is not wired up in the Copperline backend yet"
    if entry.get("board_roms") or entry.get("media_files"):
        return False, "expansion board entries are not mapped to Copperline boards yet"
    return True, None


def filter_entries(entries):
    keep, dropped = [], []
    for e in entries:
        ok, why = usable(e)
        if ok:
            keep.append(e)
        else:
            dropped.append({"id": e["id"], "why": why})
    return keep, dropped


def _mb(value, unit):
    """Amiberry size option -> bytes."""
    try:
        return int(str(value or 0), 0) * unit
    except ValueError:
        return 0


def _size(n):
    if n == 0:
        return "0"
    if n % (1 << 20) == 0:
        return f"{n >> 20}M"
    return f"{n >> 10}K"


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_config(entry, rom_path, floppy, floppy_speed=100, extra=None, media=None):
    """The TOML text for one launch and a dict describing the mapping."""
    o = dict(entry["options"])
    o.update(extra or {})
    model = MODELS.get(entry.get("model"))
    if model is None:
        raise Unsupported(f"model {entry.get('model')!r} has no Copperline machine profile")
    cpu = entry.get("cpu_variant") or entry.get("cpu")
    cpu = {"68040nommu": "68040", "68060nommu": "68060", "68LC040": "68040", "68LC060": "68060"}.get(cpu, cpu)
    if cpu not in CPUS:
        raise Unsupported(f"CPU variant {entry.get('cpu_variant')!r} has no Copperline CPU model")
    fpu = str(o.get("fpu_model", "0")) not in ("0", "")
    if cpu in ("68000", "68010", "68EC040", "68EC060"):
        fpu = False
    chipset = {"ocs": "OCS", "ecs": "ECS", "ecs_agnus": "ECS", "ecs_denise": "ECS", "aga": "AGA"}.get(
        str(o.get("chipset", "")).lower())
    if chipset is None:
        raise Unsupported(f"chipset {o.get('chipset')!r}")
    ntsc = str(o.get("ntsc", "false")).lower() == "true"
    jit = _mb(o.get("cachesize"), 1) > 0
    chip = _mb(o.get("chipmem_size"), 512 << 10)
    slow = _mb(o.get("bogomem_size"), 256 << 10)
    fast = _mb(o.get("fastmem_size"), 1 << 20)
    z3 = _mb(o.get("z3mem_size"), 1 << 20)
    mb = _mb(o.get("mbresmem_size"), 1 << 20)
    notes = []
    if slow > (512 << 10):
        notes.append(f"slow RAM {_size(slow)} -> 512K (Copperline's maximum)")
        slow = 512 << 10
    if _mb(o.get("a3000mem_size"), 1 << 20):
        notes.append("a3000mem_size (accelerator-slot RAM) not mapped")
    L = []
    if rom_path:
        L.append(f"rom = {_q(rom_path)}")
    L += ["", "[machine]", f"profile = {_q(model)}"]
    if model in ("A3000", "A4000", "A500Plus", "CDTV"):
        # the battery clock of these machines, seeded and ticking in emulated
        # time: a headless machine is unpaced, and a clock that follows the
        # host would not keep the colour clock's rate (misc_hw rtc_rate)
        L.append(f"rtc_time = {_q(RTC_SEED)}")
    L += ["", "[chipset]", f"revision = {_q(chipset)}", f"video = {_q('NTSC' if ntsc else 'PAL')}",
          "", "[cpu]", f"model = {_q(cpu)}", f"fpu = {'true' if fpu else 'false'}"]
    if jit:
        L.append("jit = true")
    L += ["", "[memory]", f"chip = {_q(_size(chip))}", f"slow = {_q(_size(slow))}", f"fast = {_q(_size(fast))}"]
    if z3:
        L.append(f"z3 = {_q(_size(z3))}")
    if model in ("A3000", "A4000"):
        L.append(f"motherboard = {_q(_size(mb))}")
    elif mb:
        notes.append("mbresmem_size without a Ramsey machine: not mapped")
    L += ["", "[serial]", 'mode = "stdout"',
          "", "[audio]", "output_enabled = false",
          "", "[input]", 'port1 = "mouse"', 'port2 = "joystick"',
          "", "[floppy]", f"speed = {int(floppy_speed)}"]
    media = media or {}
    if media.get("df1"):
        L.append("drives = 2")
    if floppy:
        L += ["", "[floppy.df0]", f"path = {_q(floppy)}", "write_protected = true"]
    if media.get("df1"):
        L += ["", "[floppy.df1]", f"path = {_q(media['df1'])}", "write_protected = false"]
    if media.get("ide_master"):
        L += ["", "[ide]", f"master = {_q(media['ide_master'])}"]
    info = {"model": model, "cpu": cpu, "fpu": fpu, "chipset": chipset, "video": "NTSC" if ntsc else "PAL",
            "jit": jit, "chip": _size(chip), "slow": _size(slow), "fast": _size(fast), "z3": _size(z3),
            "motherboard": _size(mb) if model in ("A3000", "A4000") else None, "notes": notes,
            "rom": rom_path or "bundled AROS", "floppy_speed": int(floppy_speed)}
    return "\n".join(L) + "\n", info


# ---------------------------------------------------------------------------
# control protocol client
class CCPError(OSError):
    pass


class CCP:
    """JSON-RPC 2.0 client for one Copperline control session. It also is the
    memory reader mailbox.py wants (read_longs)."""

    def __init__(self, info_path, proc, timeout=60.0):
        self.requests = 0
        self.sock = None
        self.buf = b""
        self.next_id = 1
        self.pos = {"seconds": 0.0, "frame": 0}
        deadline = time.monotonic() + timeout
        info = None
        while time.monotonic() < deadline:
            emulator.check_interrupt()
            if proc.poll() is not None:
                raise CCPError(f"Copperline exited (status {proc.returncode}) before its control server came up")
            try:
                with open(info_path, encoding="utf-8") as fh:
                    info = json.load(fh)
                break
            except (OSError, ValueError):
                time.sleep(0.05)
        if info is None:
            raise CCPError(f"no control info file {info_path} after {timeout:.0f} s")
        host, _, port = info["listen"].rpartition(":")
        self.endpoint = info["listen"]
        self.sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=30)
        self.hello = self.call("hello", {"token": info["token"]})

    def call(self, method, params=None, timeout=120.0):
        rid = self.next_id
        self.next_id += 1
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self.sock.settimeout(timeout)
        try:
            self.sock.sendall((json.dumps(req) + "\n").encode())
            self.requests += 1
            while True:
                nl = self.buf.find(b"\n")
                if nl < 0:
                    chunk = self.sock.recv(1 << 16)
                    if not chunk:
                        raise CCPError("control connection closed")
                    self.buf += chunk
                    continue
                line, self.buf = self.buf[:nl], self.buf[nl + 1:]
                msg = json.loads(line)
                if msg.get("id") != rid:
                    continue            # an event notification
                if "error" in msg:
                    e = msg["error"]
                    raise CCPError(f"{method}: {e.get('message')} (code {e.get('code')})")
                return msg.get("result")
        except socket.timeout:
            raise CCPError(f"{method}: no reply within {timeout:.0f} s")
        except ValueError as e:
            raise CCPError(f"{method}: malformed reply: {e}")

    # -- execution ---------------------------------------------------------
    def step_frames(self, n):
        stop = self.call("step_frame", {"n": max(1, int(n))}, timeout=600.0)
        if isinstance(stop, dict):
            self.pos = stop
        return stop

    def seconds(self):
        return float(self.pos.get("seconds") or 0.0)

    def sleep(self, seconds, frame_hz=50):
        """Emulated sleep (hostreq.Requests' sleep)."""
        if seconds > 0:
            self.step_frames(round(seconds * frame_hz) or 1)

    # -- memory ------------------------------------------------------------
    def read_bytes(self, addr, n):
        out = bytearray()
        while n > 0:
            take = min(n, 1 << 19)
            r = self.call("mem.read", {"addr": addr, "len": take, "encoding": "base64"})
            data = base64.b64decode(r["data"])
            if len(data) != take:
                raise CCPError(f"mem.read {addr:#x}+{take}: got {len(data)} bytes")
            out += data
            addr += take
            n -= take
        return bytes(out)

    def read_longs(self, addrs):
        if not addrs:
            return []
        lo, hi = min(addrs), max(addrs) + 4
        try:
            if hi - lo <= (16 << 20):
                blob = self.read_bytes(lo, hi - lo)
                return [int.from_bytes(blob[a - lo:a - lo + 4], "big") for a in addrs]
            return [int.from_bytes(self.read_bytes(a, 4), "big") for a in addrs]
        except CCPError:
            return [None] * len(addrs)

    def write_long(self, addr, value):
        self.call("mem.write", {"addr": addr, "data": f"{value & 0xffffffff:08x}"})
        return "OK"

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# host requests
class Requests(hostreq.Requests):
    KNOWN = hostreq.KNOWN + ("pcmcia",)

    def __init__(self, ccp, rawdir, rel, entry, **kw):
        super().__init__(ccp, rawdir, rel, options=entry["options"], **kw)
        self.ccp = ccp
        self.entry = entry
        self.port_device = {}

    def _ack(self, mbox_addr, rid):
        try:
            return self.ccp.write_long(mbox_addr + hostreq.ACK_OFFSET, int(rid))
        except (OSError, TypeError, ValueError) as e:
            return f"ERROR\t{type(e).__name__}: {e}"

    def _do_pacing(self, obj, args, rid, tag):
        module = tag if tag and tag != "boot" else None
        result = {"warp": 0, "was": "0", "switched": False,
                  "note": "Copperline's emulated machine does not see host pacing: nothing to switch"}
        if module:
            self.module_pacing[module] = 0
            result["module"] = module
        self.pacing_done = True
        return True, result

    def _do_screenshot(self, obj, args, rid, tag):
        frames = max(0, min(hostreq._int(args.get("frames")), 50 * 60))
        if frames:
            self.ccp.step_frames(frames)
        name = (f"shot-{rid}-{hostreq._safe_tag(tag)}.png" if rid is not None
                else f"shot-c{obj.get('seq')}-{hostreq._safe_tag(tag)}.png")
        path = os.path.join(self.rawdir, name)
        r = self.ccp.call("capture.screenshot", {"path": path})
        if not os.path.isfile(path):
            return False, {"error": f"capture.screenshot replied {r!r} but {path} does not exist"}
        res = {"path": self.rel + name, "sha256": hostreq._sha256(path), "size": os.path.getsize(path),
               "width": r.get("width"), "height": r.get("height")}
        if frames:
            res["waited_frames"] = frames
        return True, res

    def _do_audio_mark(self, obj, args, rid, tag):
        return False, {"error": "no audio capture in the Copperline backend yet (--audio-wav is not wired up)"}

    def _do_key(self, obj, args, rid, tag):
        code = hostreq._int(args.get("code"), -1)
        if not 0 <= code <= 0x7f:
            return False, {"error": f"key code {args.get('code')!r} is not an Amiga raw key code 0..0x7f"}
        up = 1 if hostreq._int(args.get("up")) else 0
        self.ccp.call("input.key", {"rawkey": code, "action": "release" if up else "press"})
        return True, {"injected": {"key": code, "up": up}}

    def _set_port(self, port, device):
        if self.port_device.get(port) != device:
            self.ccp.call("input.set_port", {"port": port + 1, "device": device})
            self.port_device[port] = device

    def _do_mouse(self, obj, args, rid, tag):
        port = hostreq._int(args.get("port"), 0)
        if port not in (0, 1):
            return False, {"error": f"mouse on port {port}: Copperline has controller ports 0 and 1"}
        self._set_port(port, "mouse")
        dx, dy, b = (hostreq._int(args.get(k)) for k in ("dx", "dy", "buttons"))
        self.ccp.call("input.mouse", {"dx": dx, "dy": dy, "left": bool(b & 1), "right": bool(b & 2),
                                      "middle": bool(b & 4), "port": port + 1})
        return True, {"injected": {"dx": dx, "dy": dy, "buttons": b, "port": port}}

    def _do_joy(self, obj, args, rid, tag):
        port = hostreq._int(args.get("port"), 1)
        if port not in (0, 1):
            return False, {"error": f"joystick on port {port}: Copperline has controller ports 0 and 1"}
        self._set_port(port, "joystick")
        dirs, fire = str(args.get("dirs") or ""), hostreq._int(args.get("fire"))
        self.ccp.call("input.joy", {"up": "u" in dirs, "down": "d" in dirs, "left": "l" in dirs,
                                    "right": "r" in dirs, "red": bool(fire & 1), "blue": bool(fire & 2),
                                    "port": port + 1})
        return True, {"injected": {"dirs": dirs, "fire": fire, "port": port}}

    def _do_cd32pad(self, obj, args, rid, tag):
        port = hostreq._int(args.get("port"), 1)
        if port not in (0, 1):
            return False, {"error": f"CD32 pad on port {port}: Copperline has controller ports 0 and 1"}
        self._set_port(port, "cd32")
        m = hostreq._int(args.get("buttons"))
        self.ccp.call("input.joy", {"blue": bool(m & 1), "red": bool(m & 2), "yellow": bool(m & 4),
                                    "green": bool(m & 8), "ffw": bool(m & 16), "rwd": bool(m & 32),
                                    "play": bool(m & 64), "port": port + 1})
        return True, {"injected": {"buttons": m, "port": port}}

    def _do_pcmcia(self, obj, args, rid, tag):
        op = args.get("op")
        if op == "eject":
            self.pcmcia_was = self.ccp.call("pcmcia.query")
            if not self.pcmcia_was.get("inserted"):
                return False, {"error": "no PCMCIA card in the slot"}
            self.ccp.call("pcmcia.eject")
            return True, {"op": "eject"}
        if op == "insert":
            was = getattr(self, "pcmcia_was", None) or {}
            p = {"card": was.get("card") or "sram"}
            if was.get("path"):
                p["path"] = was["path"]
            else:
                p["size"] = "4M"
            self.ccp.call("pcmcia.insert", p)
            return True, {"op": "insert", "card": p}
        return False, {"error": f"pcmcia op {op!r}: must be eject or insert"}


# ---------------------------------------------------------------------------
# one launch
class _Proc:
    """What report/analysis read of emulator.Process, over a Popen whose
    stdout is the serial capture and whose stderr is the log file."""

    def __init__(self, argv, env, cwd, serial_path, log_path):
        self.argv = argv
        self.log_path = log_path
        self.serial_fh = open(serial_path, "wb")
        self.log_fh = open(log_path, "wb")
        emulator.check_interrupt()
        self.proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=self.serial_fh, stderr=self.log_fh,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        with emulator._live_lock:
            emulator._live.add(self.proc)
        self.stop_method = None

    def alive(self):
        return self.proc.poll() is None

    def exit_info(self):
        rc = self.proc.poll()
        if rc is None:
            return {"running": True}
        d = {"running": False, "returncode": rc}
        if rc < 0:
            try:
                d["signal"] = signal.Signals(-rc).name
            except ValueError:
                d["signal"] = f"signal {-rc}"
        return d

    def tail(self, n=60):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return [x.rstrip("\n") for x in fh.readlines()[-n:]]
        except OSError:
            return []

    def stop(self, ccp=None):
        if not self.alive():
            self.stop_method = "already exited"
        else:
            if ccp is not None:
                try:
                    ccp.call("shutdown", timeout=5)
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=5)
                    self.stop_method = "control shutdown"
                except subprocess.TimeoutExpired:
                    pass
            for sig, name, wait in ((signal.SIGTERM, "SIGTERM", 3.0), (signal.SIGKILL, "SIGKILL", 5.0)):
                if self.stop_method:
                    break
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=wait)
                    self.stop_method = name
                except subprocess.TimeoutExpired:
                    pass
            self.stop_method = self.stop_method or "could not stop"
        with emulator._live_lock:
            emulator._live.discard(self.proc)
        self.serial_fh.close()
        self.log_fh.close()
        return self.stop_method


class _SerialTail:
    """Feeds what the emulator appended to serial.bin to the parser."""

    def __init__(self, path, on_data):
        self.fh = open(path, "rb")
        self.on_data = on_data
        self.bytes = 0

    def pump(self):
        data = self.fh.read()
        if data:
            self.bytes += len(data)
            self.on_data(data)
        return len(data)

    def close(self):
        self.fh.close()


def _observe(opts, entry, run, proc, ccp, tail, res):
    """Steps the machine and watches both transports. Returns why it ended:
    done | hang | timeout | crash | error. All limits are emulated seconds."""
    wall_deadline = time.monotonic() + max(600.0, opts.timeout * WALL_FACTOR)
    last_scan = -1e9
    done_at = None
    hb, hb_at = None, 0.0
    busy_until = -1.0
    while True:
        emulator.check_interrupt()
        if not proc.alive():
            return "crash"
        now = ccp.seconds()
        stop = ccp.step_frames(1 if now < busy_until else STEP_FRAMES)
        now = ccp.seconds()
        if isinstance(stop, dict) and stop.get("reason") == "double_fault":
            tail.pump()
            res["hang"] = {"unchanged_s": 0, "via": "the CPU halted (double fault)", "stop": stop}
            return "hang"
        tail.pump()
        run.harvest()
        if run.follower is None:
            a = run.mbox_from_hello(ccp)
            if a is None and now - last_scan >= SCAN_INTERVAL and run.any_records():
                last_scan = now
                a = run.scan(ccp)
            if a is not None:
                run.follower = mailbox.RingFollower(ccp, a)
                run.mb["found_at"] = f"{a:08x}"
                run.mb.setdefault("found_after_s", round(now, 3))
        h = run.poll_mailbox(ccp) if run.follower is not None else None
        run.harvest()
        if run.requests.waiting():
            busy_until = now + 1.0
            run.requests.service(run.follower.addr if run.follower is not None else None)
        s_done, m_done = run.done_seen(0)
        finished_state = h is not None and h.state in (2, 3, 4)
        if (s_done or m_done or finished_state) and done_at is None:
            done_at = now
        if done_at is not None:
            if (s_done and (m_done or run.follower is None)) or now - done_at > DONE_GRACE:
                return "done"
        if h is not None and h.state == 1:
            if h.heartbeat != hb:
                hb, hb_at = h.heartbeat, now
            elif now - hb_at > opts.hang:
                res["hang"] = {"heartbeat": h.heartbeat, "unchanged_s": opts.hang, "mod": h.mod, "case": h.case,
                               "via": "mailbox heartbeat (emulated seconds)", "at_emulated_s": round(now, 2)}
                return "hang"
        elif run.follower is None and now > BOOT_LIMIT and not run.any_records():
            res["hang"] = {"unchanged_s": BOOT_LIMIT, "via": "no record on serial after the boot limit "
                           "(emulated seconds)"}
            return "hang"
        if now > opts.timeout:
            return "timeout"
        if time.monotonic() > wall_deadline:
            res["reason"] = (f"wall-clock guard: {max(600.0, opts.timeout * WALL_FACTOR):.0f} host seconds passed "
                             f"at {now:.1f} emulated seconds")
            return "timeout"


def run_disk(opts, ctx, entry, binary=None, extra_options=None, disk=None):
    """runner._run_disk for Copperline: same result shape (docs/schema.md)."""
    from . import runner as R
    binary = binary or opts.binary
    rid = entry["id"]
    safe = R.safe_id(rid)
    dname = disk[0] if disk else None
    dsuf = f".{dname}" if dname else ""
    res = R.base_result(entry)
    res["binary"] = binary
    res["disk_name"] = dname
    res["emulator"] = "copperline"
    ok, why = usable(entry)
    if not ok:
        res["status"], res["reason"] = "skip", f"not run on Copperline: {why}"
        return res
    if "builtin" in entry["rom"]:
        rom_path = None
        res["rom"] = {"builtin": "aros", "name": "AROS ROM bundled with Copperline"}
    else:
        rom = ctx["romset"].find(entry["rom"]["ids"])
        if rom is None:
            res["status"] = "skip"
            res["reason"] = f"no ROM: needs {entry['rom']['what']}; none found in {opts.kickstarts}"
            return res
        rom_path = rom["path"]          # read in place: Copperline never writes next to a ROM
        res["rom"] = R._rom_info(rom)
    with ctx["lock"]:
        ctx["seq"][0] += 1
        etmp = os.path.join(ctx["tmp"], f"e{ctx['seq'][0]:03d}")
    R._safe_rmtree(etmp, ctx["tmp"])
    media_dir = os.path.join(etmp, "media")
    os.makedirs(media_dir)
    with open(os.path.join(etmp, "entry.txt"), "w", encoding="utf-8") as fh:
        fh.write(rid + "\n")
    # Run parameters. Instruction policy: the hardware's (no intcompat /
    # fpucompat), whatever the entry's Amiberry options say. hostspeed is a
    # host real-time check and is masked off.
    params = R.merge_params(opts.params, entry.get("params", {}))
    for name in entry.get("flags") or ():
        R._add_flag(params, name)
    R._set_flag(params, "intcompat", False)
    R._set_flag(params, "fpucompat", False)
    try:
        mask = R._mask(params.get("modules", "all"))
    except ValueError:
        mask = 0xFFFFFFFFFFFFFFFF
    params["modules"] = f"{mask & ~HOSTSPEED_BIT:#x}"
    fs = 100
    params.setdefault("floppy", str(fs))
    floppy = ""
    if opts.no_disk:
        res["disk"] = None
    else:
        floppy = os.path.join(media_dir, f"{safe}{dsuf}.adf")
        res["disk"] = adfmod.bake(disk[1] if disk else ctx["adf_src"], floppy, params,
                                  cache_dir=os.path.join(ctx["tmp"], "baked"), lock=ctx.get("bake_lock"))
    res["floppy_speed"] = fs
    # Test-owned media for devio (CONTRACT section 11, media.py): a signed
    # DF1 disk, and a signed RDB hardfile on the machines with an IDE port.
    media_cfg = {}
    if opts.media and not opts.no_disk:
        from . import media as M
        u = M._new_uuid(None)
        made = [M.create_adf(os.path.join(media_dir, "amt-df1.adf"), u)]
        media_cfg["df1"] = made[0]["path"]
        if MODELS.get(entry.get("model")) in ("A600", "A1200", "A4000"):
            # 16 MB = 32 cylinders of 16 x 63 = 63 cylinders of 16 x 32: Copperline's
            # IDE drives report 16 heads x 32 sectors, and a size that is not a whole
            # number of those cylinders leaves the last blocks unreachable by CHS
            made.append(M.create_rdb_hdf(os.path.join(media_dir, "amt-ide-rdb.hdf"), 16, u))
            media_cfg["ide_master"] = made[1]["path"]
        res["media"] = {"available": True, "function": "copperline.run_disk", "media": made, "uuid": u.hex(),
                        "not_attached": ["PCMCIA SRAM card (Copperline writes its own CIS, without CISTPL_FORMAT)"]}
    try:
        text, cfg_info = build_config(entry, rom_path, floppy, floppy_speed=fs, extra=extra_options,
                                      media=media_cfg)
    except Unsupported as e:
        res["status"], res["reason"] = "skip", f"not run on Copperline: {e}"
        if not opts.keep:
            R._safe_rmtree(etmp, ctx["tmp"])
        return res
    conf = os.path.join(etmp, f"{safe}{dsuf}.toml")
    with open(conf, "w", encoding="utf-8") as fh:
        fh.write(text)
    cfgdir = os.path.join(ctx["results"], "configs")
    with open(os.path.join(cfgdir, f"{safe}{dsuf}.toml"), "w", encoding="utf-8") as fh:
        fh.write(text)
    res["config"] = {"file": f"configs/{safe}{dsuf}.toml", "copperline": cfg_info, "dump_ok": True,
                     "differences": [], "unknown_options": [], "fixup_messages": []}
    res["resolved_options"] = {}
    res["warp"] = {"requested": "off", "effective": False, "boot": False, "boot_reason": "not needed",
                   "group": R.build_group(res.get("disk") or {}), "verified": True, "modules": [],
                   "reason": "Copperline headless: the host is unpaced, emulated time is unaffected"}
    rel = f"raw/{safe}/{dname}/" if dname else f"raw/{safe}/"
    rawdir = os.path.join(ctx["results"], rel)
    os.makedirs(rawdir, exist_ok=True)
    res["raw"] = rel
    home = os.path.join(etmp, "home")
    env = emulator.isolated_env(home, tmpdir=os.path.join(etmp, "tmp"))
    # Copperline's diagnostics are start-up environment knobs (COPPERLINE_DIAG_*,
    # COPPERLINE_DBG_*); they do not change the emulated timeline, and their
    # output lands in the entry's copperline.log.
    env.update({k: v for k, v in os.environ.items() if k.startswith("COPPERLINE_")})
    for k in ("SDL_VIDEO_DRIVER", "SDL_VIDEODRIVER", "SDL_AUDIO_DRIVER", "SDL_AUDIODRIVER", "AMIBERRY_HOME_DIR"):
        env.pop(k, None)
    # no reverse-execution ring: the run only goes forward, and mem.write
    # (the acks) would make a replay diverge anyway
    env.setdefault("COPPERLINE_DBG_RR_BUDGET_MB", "0")
    info_path = os.path.join(etmp, "ccp.json")
    argv = [binary, "--config", conf, "--noaudio", "--control", ":0", "--control-info", info_path]
    res["command"] = {"argv": argv, "env": {k: env[k] for k in ("HOME", "TMPDIR") if k in env}}
    slot = emulator.emulator_slot()
    t_wait = time.time()
    res["emulator_slot"] = slot.__enter__()
    res["slot_wait_s"] = round(time.time() - t_wait, 2)
    t_start = time.time()
    load_start = R._loadavg()
    run = R.EntryRun(opts, entry, rawdir, rel)
    R.log(opts, f"{rid}{(' [' + dname + ']') if dname else ''}: launching Copperline ({entry['title']})")
    proc, ccp, tail, ended = None, None, None, None
    try:
        proc = _Proc(argv, env, home, os.path.join(rawdir, "serial.bin"), os.path.join(rawdir, "copperline.log"))
        run.pid = proc.proc.pid
        tail = _SerialTail(os.path.join(rawdir, "serial.bin"), run.on_serial)
        try:
            ccp = CCP(info_path, proc.proc, timeout=opts.boot_timeout)
            res["ipc"] = {"endpoint": ccp.endpoint, "version": ccp.hello}
            frame_hz = 60 if cfg_info["video"] == "NTSC" else 50
            run.requests = Requests(ccp, rawdir, rel, entry, frame_hz=frame_hz, t0=run.t0,
                                    log=(lambda m: R.log(opts, f"{rid}: {m}")) if opts.verbose else None)
            ended = _observe(opts, entry, run, proc, ccp, tail, res)
        except (OSError, RunnerError) as e:
            emulator.check_interrupt()
            time.sleep(0.5)
            ended = "crash" if not proc.alive() else "error"
            res["reason"] = f"{type(e).__name__}: {e}"
        res["emulated_s"] = round(ccp.seconds(), 2) if ccp is not None else None
    finally:
        exit_before_stop = proc.exit_info() if proc is not None else {}
        stop = proc.stop(ccp) if proc is not None else None
        if tail is not None:
            tail.pump()
            tail.close()
        if ccp is not None:
            ccp.close()
        run.close()
        slot.__exit__(None, None, None)
    res["launch_attempts"] = []
    res["duration_s"] = round(time.time() - t_start, 2)
    res["started"] = datetime.datetime.fromtimestamp(t_start, datetime.timezone.utc).isoformat()
    res["process"] = {"pid": proc.proc.pid, "exit_before_stop": exit_before_stop, "stop_method": stop,
                      "exit": proc.exit_info(), "log": f"{rel}copperline.log", "log_tail": proc.tail(80)}
    res["log_counts"] = {}
    res["engine"] = None
    res["serial"] = {"port": None, "connected_after_s": 0.0, "accepted_in_log": True, "bytes": tail.bytes,
                     "closed_by_peer": False, "error": None, "raw": f"{rel}serial.bin", "via": "stdout"}
    res["mailbox"] = dict(run.mb, raw=f"{rel}mailbox.jsonl")
    res["host_context"] = dict(envinfo.host_brief(), jobs=opts.jobs, loadavg_start=load_start,
                               loadavg_end=R._loadavg(), slot_wait_s=res.get("slot_wait_s"))
    if ccp is not None:
        res["ipc"]["requests"] = ccp.requests
    merged = records.merge({"serial": run.serial_parser.records, "mailbox": run.mbox_parser.records})
    host_recs = run.requests.host if run.requests else []
    if run.requests is not None:
        res["host_requests"] = dict(run.requests.summary(), raw=f"{rel}host.jsonl")
    if host_recs:
        with open(os.path.join(rawdir, "host.jsonl"), "w", encoding="utf-8") as fh:
            for h in host_recs:
                fh.write(json.dumps(h) + "\n")
    hostreq.attach(merged, host_recs)
    res["records"] = R.summarize_records(merged, run)
    with open(os.path.join(rawdir, "records.jsonl"), "w", encoding="utf-8") as fh:
        for mrec in merged.records:
            fh.write(json.dumps(mrec) + "\n")
    res["ended"] = ended
    res["status"], why = R.classify(res, merged, ended)
    if res.get("reason") is None:
        res["reason"] = why
    if res["status"] == "crash":
        res["reason"] = res["reason"].replace("Amiberry", "Copperline")
    res["_merged"] = merged
    res["modules"] = {m["rec"].get("mod"): m["rec"].get("id") for m in merged.records
                      if m["rec"].get("t") == "mod" and m["rec"].get("ev") == "begin"}
    if not opts.keep:
        R._safe_rmtree(etmp, ctx["tmp"])
    R.log(opts, f"{rid}{(' [' + dname + ']') if dname else ''}: {res['status']} ({res['reason']}); "
          f"{len(merged.records)} records, {res.get('emulated_s')} emulated s in {res['duration_s']} host s")
    return res
