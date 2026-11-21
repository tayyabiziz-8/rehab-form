from lib import *
c = boot(video=f"{S}/cam/ELK.mjpeg"); pick(c, "ELK")
def reject(name, msg="boom"):
    c.ev(f"navigator.mediaDevices.getUserMedia=()=>Promise.reject(new DOMException({json.dumps(msg)},{json.dumps(name)}))")
    click(c, "#btnCam"); time.sleep(0.8)
    return c.ev("JSON.stringify({err:document.getElementById('camErr').textContent, shown:!document.getElementById('camErr').hidden, btn:document.getElementById('btnCam').textContent, dis:document.getElementById('btnCam').disabled, gate:!document.getElementById('gate').hidden})")
for name, must in (("NotAllowedError", "blocked"), ("NotFoundError", "No camera"), ("NotReadableError", "Another app"), ("OverconstrainedError", "settings"), ("AbortError", "couldn't start")):
    r = json.loads(reject(name)); check(f"camera {name}: message shown, step 2 kept, button usable", must in r["err"] and r["shown"] and r["gate"] and not r["dis"] and r["btn"] == "Open camera", r["err"][:80])
c.ev("Object.defineProperty(navigator,'mediaDevices',{value:undefined,configurable:true})"); click(c, "#btnCam"); time.sleep(0.5)
check("insecure page (no mediaDevices): explains secure page, offers a file", "secure" in c.ev("document.getElementById('camErr').textContent") and c.ev("!!document.querySelector('label[for=file]')"))
c.shot(f"{S}/shots/g3-camerror.png"); done(c, "camera errors")
# permission already granted
c = Chrome(video=f"{S}/cam/ELK.mjpeg"); c.view(360, 640, mobile=True); c.grant("http://localhost:8000"); c.goto(URL, wait=2); wait(c, "__rehab.state.model!=='loading'", 45)
pick(c, "RTK"); time.sleep(3.5)
check("already granted: step 2 stays on screen", c.ev("!document.getElementById('gate').hidden && !document.body.classList.contains('is-live')"))
check("already granted: says the camera is allowed", c.ev("!document.getElementById('permNote').hidden"))
check("already granted: camera stays off until pressed", c.ev("!__rehab.state.stream && !__rehab.state.running"))
check("already granted: Open camera works", open_camera(c)); done(c, "granted")
# pose model cannot load
c = Chrome(video=f"{S}/cam/ELK.mjpeg"); c.view(360, 640, mobile=True); c.cmd("Network.setBlockedURLs", urls=["*cdn.jsdelivr.net*", "*storage.googleapis.com*"]); c.goto(URL, wait=3); wait(c, "__rehab.state.model!=='loading'", 45)
check("model blocked: choose screen reports it", c.ev("__rehab.state.model==='failed' && document.getElementById('modelLine').textContent.includes(\"didn't load\")"))
pick(c, "ELK"); open_camera(c); time.sleep(2)
check("model blocked: Record disabled, hint explains", c.ev("document.getElementById('btnRec').disabled && document.getElementById('hint').textContent.includes(\"didn't load\")"))
c.shot(f"{S}/shots/g3-modelfail.png"); done(c, "model failed")
