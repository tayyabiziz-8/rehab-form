"""The phone path: the app over HTTPS on the LAN address (needs tests/phone/tls_proxy.py running), with ?diag=1."""
from lib import *
import socket
def lan():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]; s.close(); return ip
BASE = f"https://{lan()}:8443/"
c = boot(video=f"{S}/cam/ELK.mjpeg", url=BASE + "?diag=1", extra=("--ignore-certificate-errors",))
check("LAN HTTPS: the page is a secure context, so the camera is allowed", c.ev("isSecureContext && !!navigator.mediaDevices?.getUserMedia"), BASE)
check("LAN HTTPS: the pose model loads from the app (not Google)", c.ev("__rehab.state.model==='ready' && __rehab.state.modelSrc==='local'"), c.ev("__rehab.state.model+'/'+__rehab.state.modelSrc"))
check("LAN HTTPS: HEAD for the model returns 200 through the proxy", c.ev("fetch('/models/pose_landmarker_full.task',{method:'HEAD'}).then(r=>r.status)")==200)
pick(c, "ELK"); check("diag: opens the camera over HTTPS", open_camera(c)); wait(c, "__rehab.state.found", 40); time.sleep(3)
line = c.ev("document.getElementById('diag').textContent"); print("   diag line:", line)
check("diag: shows delegate, size, frame rate and pose time on the live screen", c.ev("!document.getElementById('diag').hidden") is not None and all(k in line for k in ("fps", "ms/frame")) and ("GPU" in line or "CPU" in line), line)
click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(10); click(c, "#btnRec"); wait(c, "__rehab.state.screen==='result'", 30); time.sleep(1)
rep = json.loads(c.ev("document.getElementById('diagOut').textContent")); print("   report:", {k: rep[k] for k in ("poseModel", "liveFps", "poseMsPerFrame", "recordedFps", "recordedFrames", "serverStatus", "secureContext")})
check("diag: the result screen has a copyable report with the numbers", not c.ev("document.getElementById('diagBox').hidden") and rep["secureContext"] and rep["poseMsPerFrame"] and rep["recordedFrames"] > 10 and rep["poseModel"]["delegate"] in ("GPU", "CPU"))
done(c, "LAN HTTPS")
c = boot(video=f"{S}/cam/ELK.mjpeg", url=BASE, extra=("--ignore-certificate-errors",))
pick(c, "ELK"); open_camera(c); wait(c, "__rehab.state.found", 40); time.sleep(1)
check("no ?diag: the numbers panel stays hidden for normal use", c.ev("document.getElementById('diag').hidden") is True and c.ev("document.getElementById('diagBox').hidden") is True)
done(c, "LAN HTTPS, normal")
