from lib import *
c = boot(video=f"{S}/cam/ELK.mjpeg"); inject_synth(c)
c.ev("window.__calls=0; window.__rf=window.fetch; window.__mode='ok'; window.fetch=(...a)=>{ if(String(a[0]).includes('/api/session')){ window.__calls++; if(window.__mode==='down') return Promise.reject(new TypeError('x')); if(window.__mode==='500') return Promise.resolve(new Response(JSON.stringify({ok:false,error:'ValueError: boom'}),{status:500})); if(window.__mode==='html') return Promise.resolve(new Response('<html>oops</html>',{status:502})); } return window.__rf(...a)}")
errtext = lambda: c.ev("document.getElementById('anaErrMsg').textContent")
# ---------- server failures after a real recording
pick(c, "ELK"); open_camera(c); wait(c, "__rehab.state.found", 40)
c.ev("window.__mode='down'"); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(11); click(c, "#btnRec"); wait(c, "!document.getElementById('anaErr').hidden", 20)
n = c.ev("__rehab.state.pending.frames.length")
check("server down: error screen with retry, recording kept", "reach the app server" in errtext() and c.ev("!document.getElementById('btnRetry').hidden") and n > 10, f"{n} frames")
check("camera is switched off once the set is recorded (privacy)", c.ev("!__rehab.state.stream && document.getElementById('video').srcObject===null"))
c.ev("window.__mode='500'"); click(c, "#btnRetry"); time.sleep(1.2); check("server 500: shows the server's message", "boom" in errtext(), errtext())
c.ev("window.__mode='html'"); click(c, "#btnRetry"); time.sleep(1.2); check("unreadable reply: explains it", "unreadable" in errtext(), errtext())
c.ev("window.__mode='ok'"); click(c, "#btnRetry"); ok = wait(c, "__rehab.state.screen==='result'", 25)
check("retry after failures: the same recording is analysed", ok); r = S_(c); print("   result:", r["status"], r["findings"], r["text"][:70])
check("every retry reused the same frames (no re-record needed)", c.ev("__calls")==4, c.ev("__calls"))
# real server 400 through the same path
c.ev("window.__mode='ok'"); c.ev("(async()=>{__rehab.state.exercise='ELK'; await __rehab.submit(__frames([10,7,10]).slice(0,20),{width:720,height:1280},'standard',true)})()"); wait(c, "!document.getElementById('anaErr').hidden", 20)
check("server 400 (too short for the server): its message is shown", len(errtext()) > 5 and c.ev("!document.getElementById('anaErr').hidden"), errtext())
click(c, "#btnRerecord"); time.sleep(0.8); check("'Record again' from the error screen goes to step 2", c.ev("!document.getElementById('gate').hidden && !document.body.classList.contains('is-live')"))
# ---------- too short via the real buttons
open_camera(c); wait(c, "__rehab.state.found", 40); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10); time.sleep(0.4); click(c, "#btnRec"); time.sleep(1)
check("stop straight after start: 'too short', no retry button, camera off", "too short" in errtext() and c.ev("document.getElementById('btnRetry').hidden && !__rehab.state.stream"), errtext())
click(c, "#btnRerecord"); time.sleep(0.6)
# ---------- tracking failure during a live set
open_camera(c); wait(c, "__rehab.state.found", 40); c.ev("__rehab.state.landmarker.detectForVideo=()=>{throw new Error('gl lost')}")
ok = wait(c, "!document.getElementById('gate').hidden && !document.body.classList.contains('is-live')", 25)
check("tracking fails: back to step 2 with an explanation", ok and "Pose tracking stopped" in c.ev("document.getElementById('camErr').textContent"))
done(c, "G3b-1")
