from lib import *
c = boot(video=None, fake_cam=False)
pick(c, "ELK"); c.cmd("DOM.enable"); doc = c.cmd("DOM.getDocument")["root"]["nodeId"]; nid = c.cmd("DOM.querySelector", nodeId=doc, selector="#file")["nodeId"]
c.cmd("DOM.setFileInputFiles", files=[f"{S}/cam/clip.webm"], nodeId=nid); check("video file: live view opens", wait(c, "document.body.classList.contains('is-live')", 15))
click(c, "#btnRec"); ok = wait(c, "__rehab.state.screen==='result' || (__rehab.state.screen==='analysing' && !document.getElementById('anaErr').hidden)", 90)
r = S_(c); print("   ", r["screen"], r["status"], r["err"] or r["text"][:80])
check("video file: ends by itself and returns feedback (not an error)", r["screen"] == "result", r["err"])
check("video file: recorded timestamps strictly increase", c.ev("(()=>{const t=__rehab.state.result.frames.map(f=>f.t_ms); return t.every((x,i)=>i===0||x>t[i-1])})()"))
done(c, "file")
