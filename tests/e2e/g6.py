from lib import *
def lum(hexc):
    h = hexc.strip().lstrip("#"); r, g, b = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]; f = lambda x: x / 12.92 if x <= .03928 else ((x + .055) / 1.055) ** 2.4
    return .2126 * f(r) + .7152 * f(g) + .0722 * f(b)
def cr(a, b): la, lb = lum(a), lum(b); return (max(la, lb) + .05) / (min(la, lb) + .05)
TOK = "(n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim())"
c = boot(video=None, fake_cam=False); inject_synth(c)
tok = lambda n: c.ev(f"{TOK}('{n}')")
# ---------- contrast of the colour pairs people actually read, in both themes
for theme in ("night", "day"):
    if c.ev("document.documentElement.dataset.theme") != theme: click(c, "#themeBtn"); time.sleep(0.5)
    pairs = [("--ink", "--paper", 7), ("--ink-2", "--paper", 4.5), ("--on-accent", "--accent", 4.5), ("--on-rec", "--rec", 4.5), ("--accent-text", "--paper", 4.5), ("--err", "--paper", 4.5)]
    bad = [(a, b, round(cr(tok(a), tok(b)), 1)) for a, b, m in pairs if cr(tok(a), tok(b)) < m]
    check(f"{theme}: text and button colour pairs meet contrast targets", not bad, str(bad))
night_ramp = None
click(c, "#themeBtn") if c.ev("document.documentElement.dataset.theme")!="night" else None; time.sleep(0.4)
ramp = [lum(tok(f"--s{i}")) for i in range(6)]
check("Night time-ramp gets steadily brighter from early to late (readable without telling hues apart)", all(b > a for a, b in zip(ramp, ramp[1:])), str([round(x, 2) for x in ramp]))
# ---------- theme toggle: persists, recolours canvases, updates the label
c.ev("window.scrollTo(0,0)"); a = sig(c, ".exrow canvas")
click(c, "#themeBtn"); time.sleep(0.8)
check("toggle: switches to Day and says how to switch back", c.ev("document.documentElement.dataset.theme")=="day" and "night" in c.ev("document.getElementById('themeBtn').getAttribute('aria-label')"))
check("toggle: canvases recolour too (not just the page)", a != sig(c, ".exrow canvas") and c.ev("getComputedStyle(document.body).color")=="rgb(34, 16, 63)")
check("toggle: the choice is remembered", c.ev("localStorage.getItem('rf-theme')")=="day"); c.goto(URL, wait=2)
check("toggle: still Day after a reload", c.ev("document.documentElement.dataset.theme")=="day")
click(c, "#themeBtn"); time.sleep(0.4); check("toggle: back to Night", c.ev("document.documentElement.dataset.theme")=="night")
# ---------- header never wraps on a phone, on any screen
inject_synth(c); wraps = {}
def header_ok(tag): wraps[tag] = c.ev("(()=>{const b=document.querySelector('.brand').getBoundingClientRect(), s=document.getElementById('stepText').getBoundingClientRect(); return [Math.round(b.height), Math.round(s.height)]})()")
header_ok("choose"); pick(c, "ELK"); header_ok("step2"); c.ev("__submit([10,7,10])"); wait(c, "__rehab.state.screen==='result'", 20); time.sleep(1); header_ok("result"); click(c, "#btnClin"); time.sleep(1); header_ok("clinician")
check("header: brand and step counter stay on one line on every screen at 360px", all(v[0] <= 30 and v[1] <= 22 for v in wraps.values()), str(wraps))
check("progress ribbon follows the step (4 of 4 on the clinician screen)", c.ev("getComputedStyle(document.getElementById('ribbon')).getPropertyValue('--p').trim()")=="1")
# ---------- result plate: confetti on a good set only
c.ev("__rehab.go('choose')"); c.ev("__submit([30,30,30])"); wait(c, "__rehab.state.screen==='result'", 20); time.sleep(1.8)
check("good set: the result plate has its burst of light", c.ev("!!__rehab.PP.d.praise && Array.isArray(__rehab.PP.d.parts) && __rehab.PP.d.parts.length>20"))
c.ev("__rehab.go('choose')"); c.ev("__submit([10,7,10])"); wait(c, "__rehab.state.screen==='result'", 20); time.sleep(1.2)
check("correction: no burst, but the ruler is drawn", c.ev("!__rehab.PP.d.praise && !!__rehab.PP.d.gauge"))
# ---------- live: light-painting trails, rep pulse, Stop layout
c.ev("__rehab.go('choose')"); pick(c, "ELK")
c.ev("""(()=>{const Y=window.SYNTH, idx={head:0,LS:11,RS:12,LE:13,RE:14,LW:15,RW:16,LH:23,RH:24,LK:25,RK:26,LA:27,RA:28}; Y.REPS.forEach((r,i)=>r.peak=[22,22,22][i]);
 window.__mkSim=(w,h)=>{const s=h*0.8/3.6,cx=w*0.5,hy=h*0.1+2.05*s; return ()=>{const st=__rehab.state; const t=st.recording?((performance.now()-st.recStart)/1000)%12:0; const P=Y.pose(t); const lm=Array.from({length:33},()=>({x:.5,y:.2,z:0})); for(const [k,i] of Object.entries(idx)){const p=P[k]; lm[i]={x:(cx+p[0]*s)/w,y:(hy+p[1]*s)/h,z:0}} return lm}};
 const st=__rehab.state; st.source='camera'; st.mirrorOn=true; st.sim={fn:__mkSim(720,1280),size:[720,1280],t0:performance.now()}; __rehab.enterLive()})()""")
wait(c, "__rehab.state.found", 10); click(c, "#btnRec"); wait(c, "__rehab.state.phase==='recording'", 10)
seen = {"pulse": False, "trail": 0, "isrec": False}
for _ in range(60):
    r = json.loads(c.ev("JSON.stringify({p:!!__rehab.state.pulse, w:__rehab.state.wr.length, cls:document.getElementById('btnRec').className})")); seen["pulse"] |= r["p"]; seen["trail"] = max(seen["trail"], r["w"]); seen["isrec"] |= "isrec" in r["cls"]; time.sleep(0.25)
check("live: wrists leave a light-painting trail", seen["trail"] > 5, seen["trail"])
check("live: a ring of light marks each counted repetition", seen["pulse"])
check("live: Record becomes the glowing Stop", seen["isrec"])
lay = json.loads(c.ev("""JSON.stringify((()=>{const b=document.getElementById('btnRec').getBoundingClientRect(), i=document.getElementById('recIco').getBoundingClientRect(), l=document.getElementById('recLabel').getBoundingClientRect(); return {btnH:Math.round(b.height), iconLeftOfLabel:i.right<=l.left+1, centred:Math.abs((i.top+i.height/2)-(b.top+b.height/2))<6}})())"""))
check("live: Stop shows its icon beside the label, button stays 64px tall", lay["iconLeftOfLabel"] and lay["centred"] and lay["btnH"] <= 70, str(lay))
c.shot(f"{S}/shots/g6-live-night.png"); done(c, "G6")
