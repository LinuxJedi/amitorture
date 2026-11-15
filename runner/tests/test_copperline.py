"""Copperline backend (runner/amitorture/copperline.py): the parts that need
no emulator."""
import base64
import json
import os
import socket
import tempfile
import threading
import unittest

from amitorture import copperline, hostreq, matrix


def entry(eid):
    return matrix.all_entries()[eid]


class Config(unittest.TestCase):
    def test_a500_maps_model_cpu_and_memory(self):
        text, info = copperline.build_config(entry("a500-68000-ce"), "/roms/kick13.rom", "/tmp/x.adf")
        self.assertEqual((info["model"], info["cpu"], info["chipset"], info["video"]),
                         ("A500", "68000", "OCS", "PAL"))
        self.assertEqual((info["chip"], info["slow"], info["fast"]), ("512K", "512K", "0"))
        self.assertFalse(info["fpu"])
        self.assertIn('rom = "/roms/kick13.rom"', text)
        self.assertIn('[serial]\nmode = "stdout"', text)
        self.assertIn('[floppy.df0]\npath = "/tmp/x.adf"', text)

    def test_a3000_gets_motherboard_ram_and_fpu(self):
        text, info = copperline.build_config(entry("a3000-68030-882-compat"), "/r/k.rom", "/tmp/x.adf")
        self.assertEqual((info["model"], info["cpu"], info["motherboard"]), ("A3000", "68030", "8M"))
        self.assertTrue(info["fpu"])
        self.assertIn('motherboard = "8M"', text)

    def test_builtin_rom_and_jit(self):
        text, info = copperline.build_config(entry("aros-aga-68020-fast"), None, "/tmp/x.adf")
        self.assertNotIn("rom =", text)
        self.assertEqual((info["z3"], info["jit"]), ("16M", False))
        _, info = copperline.build_config(entry("a1200-68ec020-fast-jit"), "/r/k.rom", "/tmp/x.adf")
        self.assertTrue(info["jit"])
        self.assertEqual(info["cpu"], "68EC020")

    def test_media(self):
        text, _ = copperline.build_config(entry("a1200-68ec020-fast"), "/r/k.rom", "/tmp/x.adf",
                                          media={"df1": "/m/df1.adf", "ide_master": "/m/ide.hdf"})
        self.assertIn("drives = 2", text)
        self.assertIn('[floppy.df1]\npath = "/m/df1.adf"\nwrite_protected = false', text)
        self.assertIn('[ide]\nmaster = "/m/ide.hdf"', text)

    def test_ntsc(self):
        _, info = copperline.build_config(entry("a600-68000-compat-ntsc"), "/r/k.rom", "/tmp/x.adf")
        self.assertEqual((info["video"], info["chipset"], info["chip"]), ("NTSC", "ECS", "1M"))


class Filter(unittest.TestCase):
    def test_engine_twins_and_hostspeed_entries_are_dropped(self):
        keep, dropped = copperline.filter_entries(matrix.expand("smoke"))
        ids = [e["id"] for e in keep]
        self.assertIn("a500-68000-ce", ids)
        self.assertIn("a1200-68ec020-fast-jit", ids)
        self.assertFalse([i for i in ids if i.endswith("-thr") or i.endswith("-strict") or "-real" in i])
        self.assertTrue(all(d["why"] for d in dropped))
        self.assertEqual(len(keep) + len(dropped), len(matrix.expand("smoke")))


class FakeServer:
    """A one-connection JSON-RPC server that records calls."""

    def __init__(self):
        self.calls = []
        self.mem = bytearray(1 << 16)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def close(self):
        self.thread.join(timeout=5)
        self.sock.close()

    def serve(self):
        conn, _ = self.sock.accept()
        with conn, conn.makefile("rwb") as fh:
            self.handle(fh)

    def handle(self, fh):
        for line in fh:
            req = json.loads(line)
            m, p = req["method"], req.get("params") or {}
            self.calls.append((m, p))
            if m == "mem.read":
                res = {"data": base64.b64encode(bytes(self.mem[p["addr"]:p["addr"] + p["len"]])).decode()}
            elif m == "mem.write":
                data = bytes.fromhex(p["data"])
                self.mem[p["addr"]:p["addr"] + len(data)] = data
                res = {"written": len(data)}
            elif m == "step_frame":
                res = {"reason": "step", "seconds": 1.0, "frame": 50}
            else:
                res = {}
            fh.write((json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": res}) + "\n").encode())
            fh.flush()


class Proc:
    returncode = None

    def poll(self):
        return None


class Control(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.tmp = tempfile.TemporaryDirectory()
        self.info = os.path.join(self.tmp.name, "ccp.json")
        with open(self.info, "w") as fh:
            json.dump({"listen": f"127.0.0.1:{self.srv.port}", "token": "t", "proto": 1}, fh)
        self.ccp = copperline.CCP(self.info, Proc(), timeout=5)

    def tearDown(self):
        self.ccp.close()
        self.srv.close()
        self.tmp.cleanup()

    def test_memory_and_ack(self):
        self.srv.mem[0x100:0x108] = bytes.fromhex("0102030411223344")
        self.assertEqual(self.ccp.read_longs([0x104, 0x100]), [0x11223344, 0x01020304])
        req = copperline.Requests(self.ccp, "/nonexistent", "", entry("a500-68000-ce"))
        req.offer({"t": "request", "seq": 3, "id": 7, "what": "wait", "tag": "x"}, "serial")
        made = req.service(0x200)
        self.assertEqual([h["acked"] for h in made], [True])
        self.assertEqual(self.srv.mem[0x200 + hostreq.ACK_OFFSET:0x200 + hostreq.ACK_OFFSET + 4],
                         bytes.fromhex("00000007"))

    def test_requests_map_to_control_methods(self):
        req = copperline.Requests(self.ccp, "/nonexistent", "", entry("a500-68000-ce"))
        for i, (what, args) in enumerate([
                ("key", {"code": 0x45, "up": 0}),
                ("mouse", {"dx": 3, "dy": -2, "buttons": 1, "port": 0}),
                ("joy", {"port": 1, "dirs": "ur", "fire": 1}),
                ("cd32pad", {"port": 1, "buttons": 0x41}),
                ("pacing", {}),
                ("audio_mark", {"edge": "start"})]):
            req.offer({"t": "request", "seq": i, "id": i + 1, "what": what, "tag": "t", "args": args}, "serial")
        made = req.service(0x200)
        self.assertEqual([h["acked"] for h in made], [True, True, True, True, True, False])
        calls = {m: p for m, p in self.srv.calls}
        self.assertEqual(calls["input.key"], {"rawkey": 0x45, "action": "press"})
        self.assertEqual(calls["input.mouse"]["port"], 1)
        self.assertEqual(calls["input.mouse"]["left"], True)
        joy = [p for m, p in self.srv.calls if m == "input.joy"]
        self.assertEqual((joy[0]["up"], joy[0]["right"], joy[0]["red"], joy[0]["port"]), (True, True, True, 2))
        self.assertEqual((joy[1]["blue"], joy[1]["play"], joy[1]["red"]), (True, True, False))
        self.assertIn(("input.set_port", {"port": 2, "device": "cd32"}), self.srv.calls)
        # the refused request got id | $80000000, so the guest stops waiting
        self.assertEqual(self.srv.mem[0x200 + hostreq.ACK_OFFSET:0x200 + hostreq.ACK_OFFSET + 4],
                         bytes.fromhex("80000006"))


if __name__ == "__main__":
    unittest.main()
