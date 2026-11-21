from lib import *
def counting(c): c.ev("window.__calls=0; window.__rf=window.fetch; window.fetch=(...a)=>{ if(String(a[0]).includes('/api/session')) window.__calls++; return window.__rf(...a)}")
c = boot(video=f"{S}/cam/ELK.mjpeg"); counting(c)
# ---------- 90 s auto-stop
pick(c, "ELK"); open_camera(c); wait(c, "__rehab.state.found", 40); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(9)
c.ev("__rehab.state.recStart -= 91000"); ok = wait(c, "['analysing','result'].includes(__rehab.state.screen)", 20); wait(c, "__rehab.state.screen==='result' || !document.getElementById('anaErr').hidden", 20)
check("90 s limit: the set stops by itself and is analysed", ok and c.ev("__calls")==1, c.ev("__calls"))
# ---------- double Stop sends one request
click(c, "#btnAgain") if c.ev("__rehab.state.screen==='result'") else click(c, "#btnRerecord"); time.sleep(0.6); open_camera(c); wait(c, "__rehab.state.found", 40)
c.ev("__calls=0"); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(10)
c.ev("document.getElementById('btnRec').click(); document.getElementById('btnRec').click()"); time.sleep(4)
check("double tap on Stop sends only one request", c.ev("__calls")==1, c.ev("__calls"))
# ---------- leaving in the middle of a recording
click(c, "#btnAgain") if c.ev("__rehab.state.screen==='result'") else click(c, "#btnRerecord"); time.sleep(0.6); open_camera(c); wait(c, "__rehab.state.found", 40)
c.ev("__calls=0"); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(3); click(c, "#liveBack"); time.sleep(1)
check("back during a recording: returns to choose, camera off, nothing sent", c.ev("__rehab.state.screen==='choose' && !__rehab.state.stream && !__rehab.state.recording && __calls===0 && !document.body.classList.contains('is-live')"))
# ---------- rotate the phone while live
pick(c, "ELK"); open_camera(c); wait(c, "__rehab.state.found", 40)
c.view(640, 360, mobile=True); time.sleep(1.2)
check("landscape phone: Record stays on screen and enabled", c.ev("document.getElementById('btnRec').getBoundingClientRect().bottom<=innerHeight+1 && !document.getElementById('btnRec').disabled"), c.ev("Math.round(document.getElementById('btnRec').getBoundingClientRect().bottom)+'/'+innerHeight"))
c.shot(f"{S}/shots/g3-landscape.png"); c.view(360, 640, mobile=True); time.sleep(0.8)
check("rotating back: still live and tracking", c.ev("document.body.classList.contains('is-live') && __rehab.state.running"))
click(c, "#liveBack"); time.sleep(0.5)
# ---------- video file instead of a camera
pick(c, "ELK"); c.cmd("DOM.enable"); doc = c.cmd("DOM.getDocument")["root"]["nodeId"]; nid = c.cmd("DOM.querySelector", nodeId=doc, selector="#file")["nodeId"]
c.cmd("DOM.setFileInputFiles", files=[f"{S}/cam/clip.webm"], nodeId=nid); ok = wait(c, "document.body.classList.contains('is-live')", 15)
check("video file: opens the live view from a file", ok and c.ev("__rehab.state.source==='file'"))
check("video file: hint explains, no mirroring, Record enabled once the model is ready", c.ev("document.getElementById('hint').textContent.includes('Record') && !document.getElementById('stage').classList.contains('mirror') && !document.getElementById('btnRec').disabled"))
c.ev("__calls=0"); click(c, "#btnRec"); time.sleep(1); check("video file: Record starts at once (no countdown) and plays", c.ev("__rehab.state.phase==='recording' && !document.getElementById('video').paused"))
ok = wait(c, "__rehab.state.screen==='result'", 90) ; check("video file: reaches the end by itself and is analysed", ok and c.ev("__calls")==1, c.ev("__rehab.state.screen")); print("   result:", S_(c)["status"], S_(c)["text"][:80])
done(c, "G3c-1")
# ---------- real RTK (no validated checks) and CTK
for ex in ("RTK", "CTK"):
    c = boot(video=f"{S}/cam/{ex}.mjpeg"); pick(c, ex); open_camera(c); wait(c, "__rehab.state.found", 40)
    click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(11); click(c, "#btnRec"); ok = wait(c, "__rehab.state.screen==='result'", 30); time.sleep(2)
    r = S_(c); print(f"   {ex}:", r["status"], r["findings"], "|", (r["text"] or "")[:90])
    check(f"{ex}: real recording gets a result and the patient text matches", ok and c.ev("document.getElementById('patientText').textContent===__rehab.state.result.feedback.text"))
    if ex == "RTK": check("RTK: no automatic checks, so no clinician findings to open", r["status"] in ("no_checks", "not_seen", "not_this_exercise") or not r["findings"], r["status"])
    check(f"{ex}: clinician button only when there are findings", c.ev("document.getElementById('btnClin').hidden")==(not r["findings"]))
    if r["findings"]:
        click(c, "#btnClin"); time.sleep(1.5); check(f"{ex}: clinician view opens with a replay", c.ev("__rehab.state.screen==='clinician' && !!__rehab.state.rp"))
        c.ev("window.scrollTo(0,0)"); c.shot(f"{S}/shots/g3-{ex}-clin.png")
    done(c, ex)
