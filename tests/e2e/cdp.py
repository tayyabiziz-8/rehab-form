"""Tiny Chrome DevTools Protocol client (stdlib only) for driving headless Chrome in tests."""
import socket, os, json, base64, struct, subprocess, time, urllib.request, hashlib, tempfile, shutil

class WS:
    def __init__(self, url):
        host, rest = url[5:].split("/", 1); h, p = host.split(":"); self.s = socket.create_connection((h, int(p)), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf: buf += self.s.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
    def _read(self, n):
        while len(self.buf) < n:
            d = self.s.recv(1 << 20)
            if not d: raise EOFError("closed")
            self.buf += d
        out, self.buf = self.buf[:n], self.buf[n:]; return out
    def send(self, text):
        data = text.encode(); hdr = bytearray([0x81]); n = len(data)
        if n < 126: hdr.append(0x80 | n)
        elif n < 65536: hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else: hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4); self.s.sendall(bytes(hdr) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
    def recv(self):
        msg = b""
        while True:
            b0, b1 = self._read(2); fin, op = b0 & 0x80, b0 & 0x0F; n = b1 & 0x7F
            if n == 126: n = struct.unpack(">H", self._read(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            if op == 9: continue
            msg += payload
            if fin: return msg.decode()

class Chrome:
    def __init__(self, video=None, port=9333, fake_cam=True, deny=False, extra=()):
        self.dir = tempfile.mkdtemp(prefix="cdp-"); self.port = port
        args = ["google-chrome", "--headless=new", "--no-sandbox", f"--remote-debugging-port={port}", f"--user-data-dir={self.dir}", "--hide-scrollbars",
                "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required", "--mute-audio"]
        if fake_cam: args += ["--use-fake-device-for-media-stream"] + (["--use-fake-ui-for-media-stream"] if not deny else ["--deny-permission-prompts"])
        if video: args += [f"--use-file-for-fake-video-capture={video}"]
        args += list(extra) + ["about:blank"]
        self.p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try: tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
            except Exception: time.sleep(0.25)
        page = [t for t in tabs if t["type"] == "page"][0]; self.ws = WS(page["webSocketDebuggerUrl"]); self.id = 0; self.errors = []; self.logs = []
        for d in ("Page", "Runtime", "Log", "Network"): self.cmd(f"{d}.enable")
    def cmd(self, method, **params):
        self.id += 1; mid = self.id; self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == mid:
                if "error" in m: raise RuntimeError(f"{method}: {m['error']}")
                return m["result"]
            self._event(m)
    def _event(self, m):
        k = m.get("method")
        if k == "Runtime.exceptionThrown": self.errors.append(m["params"]["exceptionDetails"].get("exception", {}).get("description", str(m["params"]["exceptionDetails"]))[:300])
        elif k == "Runtime.consoleAPICalled" and m["params"]["type"] in ("error", "warning"): self.logs.append((m["params"]["type"], " ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:200]))
    def pump(self, secs=0.2):
        self.ws.s.settimeout(secs)
        try:
            while True: self._event(json.loads(self.ws.recv()))
        except Exception: pass
        self.ws.s.settimeout(60)
    def goto(self, url, wait=1.0):
        self.cmd("Page.navigate", url=url); time.sleep(wait); self.pump(0.1)
    def ev(self, js, wait=True):
        r = self.cmd("Runtime.evaluate", expression=js, awaitPromise=wait, returnByValue=True)
        if "exceptionDetails" in r: raise RuntimeError("JS: " + r["exceptionDetails"].get("exception", {}).get("description", str(r["exceptionDetails"]))[:400])
        return r["result"].get("value")
    def view(self, w, h, mobile=False, scale=2, reduced=False, dark=False):
        self.cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=scale, mobile=mobile)
        feats = [{"name": "prefers-reduced-motion", "value": "reduce" if reduced else "no-preference"}, {"name": "prefers-color-scheme", "value": "dark" if dark else "light"}]
        self.cmd("Emulation.setEmulatedMedia", features=feats)
    def shot(self, path, full=False):
        r = self.cmd("Page.captureScreenshot", format="png", captureBeyondViewport=full); open(path, "wb").write(base64.b64decode(r["data"]))
    def grant(self, origin, perms=("videoCapture",)):
        self.cmd("Browser.grantPermissions", origin=origin, permissions=list(perms))
    def close(self):
        try: self.p.terminate(); self.p.wait(5)
        except Exception: self.p.kill()
        shutil.rmtree(self.dir, ignore_errors=True)
