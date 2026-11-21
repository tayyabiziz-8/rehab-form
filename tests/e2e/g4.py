from lib import *
OV = "(()=>{const bad=[]; for(const e of document.querySelectorAll('body *')){ if(e.closest('[hidden]')||e.classList.contains('skip')) continue; const r=e.getBoundingClientRect(); if(r.width>0 && r.right>innerWidth+1) bad.push(e.tagName+'#'+e.id+'.'+(e.className.baseVal??e.className)+':'+Math.round(r.right)); } return JSON.stringify({iw:innerWidth, sw:document.documentElement.scrollWidth, bad:bad.slice(0,4)})})()"
def screens(c, w, label):
    out = {}
    out["choose"] = json.loads(c.ev(OV))
    pick(c, "ELK"); time.sleep(0.5); out["step2"] = json.loads(c.ev(OV))
    open_camera(c); wait(c, "__rehab.state.found", 40); time.sleep(0.6); out["live"] = json.loads(c.ev(OV))
    live_ok = c.ev("document.getElementById('btnRec').getBoundingClientRect().bottom<=innerHeight+1")
    click(c, "#liveBack"); time.sleep(0.4)
    c.ev("__submit([10,7,10])"); wait(c, "__rehab.state.screen==='result'", 20); time.sleep(1.2); out["result"] = json.loads(c.ev(OV))
    click(c, "#btnClin"); time.sleep(1.5); out["clinician"] = json.loads(c.ev(OV))
    bad = {k: v for k, v in out.items() if v["sw"] > v["iw"] or v["bad"] or v["iw"] != w}
    check(f"{label}: no horizontal overflow on any screen", not bad, json.dumps(bad)[:160]); check(f"{label}: Record reachable on the live screen", live_ok)
    return out
c = boot(video=f"{S}/cam/ELK.mjpeg"); inject_synth(c)
screens(c, 360, "360x640")
for (w, h, m) in ((320, 568, True), (390, 844, True), (768, 1024, True), (1024, 768, False), (1440, 900, False)):
    c.view(w, h, mobile=m); c.goto(URL, wait=2); wait(c, "__rehab.state.model!=='loading'", 45); inject_synth(c)
    screens(c, w, f"{w}x{h}")
    if w == 1440:
        check("1440x900: clinician workspace fits one screen with no overlap", c.ev("(()=>{const q=id=>document.getElementById(id).getBoundingClientRect(); return document.documentElement.scrollHeight<=innerHeight+2 && q('det').bottom<=q('cwave').top+2})()"), c.ev("document.documentElement.scrollHeight+'/'+innerHeight"))
        c.shot(f"{S}/shots/g4-desktop-clin.png")
done(c, "responsive")
# ---------- reduced motion
c = boot(video=f"{S}/cam/ELK.mjpeg"); c.view(360, 640, mobile=True, reduced=True); c.goto(URL, wait=2); wait(c, "__rehab.state.model!=='loading'", 45); inject_synth(c)
check("reduced motion is reported to the page", c.ev("matchMedia('(prefers-reduced-motion: reduce)').matches"))
pick(c, "ELK"); a = sig(c, "#demo"); time.sleep(1.2); check("reduced motion: set-up demo is a still figure", a == sig(c, "#demo"))
check("reduced motion: screen transitions are off", c.ev("getComputedStyle(document.getElementById('s-session').querySelector('.gate')).animationName==='none'"))
open_camera(c); wait(c, "__rehab.state.found", 40); click(c, "#btnRec"); time.sleep(0.8)
check("reduced motion: countdown numeral has no animation", c.ev("getComputedStyle(document.querySelector('#count .n')).animationName==='none'"))
c.ev("document.getElementById('btnRec').click()"); time.sleep(0.5); click(c, "#liveBack"); time.sleep(0.4)
c.ev("__submit([10,7,10])"); wait(c, "__rehab.state.screen==='result'", 20); time.sleep(1.0)
a = sig(c, "#pplate"); time.sleep(1.0); check("reduced motion: result plate is complete and still", a == sig(c, "#pplate") and pixels(c, "#pplate") > 100)
check("reduced motion: no shutter flash class used", c.ev("!document.getElementById('stage').classList.contains('flash')"))
click(c, "#btnClin"); time.sleep(1.2); check("reduced motion: clinician replay does not autoplay", c.ev("__rehab.state.rp.playing===false"))
click(c, ".sent"); time.sleep(0.5); check("reduced motion: tapping a sentence selects it without autoplaying", c.ev("document.querySelectorAll('.sent.on').length===1 && __rehab.state.rp.playing===false"))
c.shot(f"{S}/shots/g4-reduced.png"); done(c, "reduced motion")
# ---------- the page is Night by default and ignores the system scheme
c = boot(video=None, fake_cam=False); c.view(360, 640, mobile=True, dark=False); c.goto(URL, wait=2)
check("default theme is Night even when the system prefers light", c.ev("document.documentElement.dataset.theme")=="night" and c.ev("getComputedStyle(document.body).backgroundColor")!="rgb(255, 247, 236)"); c.close()
