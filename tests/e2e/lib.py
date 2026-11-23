"""Helpers for the browser scenarios. S is this folder: fake-webcam clips live in S/cam, screenshots go to S/shots."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import Chrome
S = os.path.dirname(os.path.abspath(__file__))
os.makedirs(S + "/shots", exist_ok=True)
URL = "http://localhost:8000/"
RESULTS = []
def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)
def wait(c, js, t=30, step=0.35):
    end = time.time() + t
    while time.time() < end:
        try:
            if c.ev(js): return True
        except Exception: pass
        time.sleep(step)
    return False
def boot(video=None, w=360, h=640, mobile=True, url=None, **kw):
    c = Chrome(video=video, **kw); c.view(w, h, mobile=mobile); c.goto(url or URL, wait=2); wait(c, "__rehab.state.model!=='loading'", 45); return c
def pick(c, ex): c.ev(f"document.querySelector('[data-ex={ex}]').click()"); time.sleep(0.5)
def click(c, sel):
    r = c.ev(f"(()=>{{const b=document.querySelector({json.dumps(sel)}).getBoundingClientRect(); return [b.left+b.width/2,b.top+b.height/2]}})()")
    for t in ("mousePressed", "mouseReleased"): c.cmd("Input.dispatchMouseEvent", type=t, x=r[0], y=r[1], button="left", clickCount=1)
    time.sleep(0.15)
def drag(c, x0, y0, x1, y1, steps=8):
    c.cmd("Input.dispatchMouseEvent", type="mousePressed", x=x0, y=y0, button="left", clickCount=1)
    for i in range(1, steps + 1): c.cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x0 + (x1 - x0) * i / steps, y=y0 + (y1 - y0) * i / steps, button="left", buttons=1)
    c.cmd("Input.dispatchMouseEvent", type="mouseReleased", x=x1, y=y1, button="left", clickCount=1)
def open_camera(c):
    click(c, "#btnCam"); ok = wait(c, "document.body.classList.contains('is-live')", 15); return ok
def record_set(c, secs=9):
    ok = wait(c, "__rehab.state.found", 30); 
    if not ok: return False
    click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 8); time.sleep(secs); click(c, "#btnRec"); return wait(c, "['result','analysing'].includes(__rehab.state.screen)&&__rehab.state.screen!=='analysing' || !document.getElementById('anaErr').hidden", 30)
def S_(c): return json.loads(c.ev("JSON.stringify({screen:__rehab.state.screen,phase:__rehab.state.phase,status:__rehab.state.result?.status,text:__rehab.state.result?.feedback.text,findings:(__rehab.state.result?.findings||[]).map(f=>[f.id,f.status,f.value,f.start_s,f.end_s]),err:document.getElementById('anaErrMsg').textContent,camErr:document.getElementById('camErr').textContent})"))
def done(c, label=""):
    c.pump(0.3); bad = [e for e in c.errors]; check(f"{label} no uncaught JS errors", not bad, "; ".join(bad)[:200]); c.close()
def inject_synth(c):
    c.ev("""new Promise((res,rej)=>{const s=document.createElement('script'); s.src='http://127.0.0.1:8010/shared/synth.js'; s.onload=res; s.onerror=rej; document.head.appendChild(s);})""")
    c.ev("""(()=>{const Y=window.SYNTH, idx={head:0,LS:11,RS:12,LE:13,RE:14,LW:15,RW:16,LH:23,RH:24,LK:25,RK:26,LA:27,RA:28};
      window.__frames=(peaks,w=720,h=1280,n=240)=>{ Y.REPS.forEach((r,i)=>r.peak=peaks[i]); const s=h*0.8/3.6, cx=w*0.5, hy=h*0.1+2.05*s; return Array.from({length:n},(_,i)=>{ const P=Y.pose(i*0.05); const lm=Array.from({length:33},()=>[.5,.2,0]); for(const [k,j] of Object.entries(idx)){const p=P[k]; lm[j]=[+((cx+p[0]*s)/w).toFixed(5),+((hy+p[1]*s)/h).toFixed(5),0];} return {t_ms:i*50,landmarks:lm}; }); };
      window.__submit=async(peaks,ex='ELK',st='standard')=>{ __rehab.state.exercise=ex; await __rehab.submit(__frames(peaks),{width:720,height:1280},st,true); }; return 1})()""")
def pixels(c, sel, thr=90):
    return c.ev(f"""(()=>{{const cv=document.querySelector({json.dumps(sel)}); const d=cv.getContext('2d').getImageData(0,0,cv.width,cv.height).data; const bg=[d[0],d[1],d[2]]; let n=0; for(let i=0;i<d.length;i+=16) if(Math.abs(d[i]-bg[0])+Math.abs(d[i+1]-bg[1])+Math.abs(d[i+2]-bg[2])>{thr}) n++; return n}})()""")
def sig(c, sel): return c.ev(f"document.querySelector({json.dumps(sel)}).toDataURL().length + ':' + document.querySelector({json.dumps(sel)}).toDataURL().slice(-40)")
