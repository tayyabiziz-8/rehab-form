# Phone test

Goal: find out what a real phone does that headless Chrome cannot show: camera, frame rate, touch, glare, heat. Two phones at least (one Android, one iPhone).

## Set up (about two minutes)

```bash
python app/server.py                    # terminal 1: the app, on 127.0.0.1:8000
python tests/phone/tls_proxy.py         # terminal 2: private HTTPS on your Wi-Fi, prints the address
```

On the phone: join the **same Wi-Fi as the laptop**, open the printed `https://192.168.x.x:8443/?diag=1`.
The first time the browser warns about the certificate (it is self-signed, made on your laptop, valid 30 days). Accept it:
Chrome Android "Advanced → Proceed"; Safari iPhone "Show Details → visit this website".
`?diag=1` adds a small line on the camera view (GPU or CPU, frame rate, milliseconds per frame) and a copyable report on the result screen. Leave it off for ordinary use.

If the phone cannot reach the page: the laptop firewall may block port 8443 (`sudo ufw allow 8443/tcp`, remove it afterwards), the Wi-Fi may isolate devices (university Wi-Fi often does: use a phone hotspot), or the phone is on mobile data.
Nothing leaves your network. Only the proxy listens on it; the app stays on localhost. Stop both with Ctrl-C.

## What to check (tick each, note what you saw)

Camera and framing
- [ ] Step 2 stays until you press Open camera; the permission prompt appears and works.
- [ ] Front camera opens; the picture is mirrored like a mirror; the skeleton sits on you, not beside you.
- [ ] Phone at chest height, 2–3 m back: "We can see you" appears; hips in view hint works.
- [ ] Portrait and landscape both usable; rotating mid-view does not break it.

Recording
- [ ] Record, countdown readable from 2–3 m, then a 3–5 repetition set; Stop is easy to hit one-handed.
- [ ] Live angles and rep count look right (side bend rising on the stretch, one tally mark per rep).
- [ ] Wrist light trails and the ring on each rep are smooth, not stuttering.
- [ ] Phone does not get hot or drop frames within 30 s (the live line shows fps).

Result
- [ ] Do a good set and a deliberately weak one (bend too little / arms too low): the weak one is corrected, the good one is not.
- [ ] The sentence is fully visible; Record again and the clinician view work; scrubbing the waveform with a finger works.
- [ ] Day and Night both readable outdoors/indoors.

Break it on purpose
- [ ] Deny the camera permission once: the message explains how to allow it.
- [ ] Cover the camera mid-set; walk out of frame; lock and unlock the screen mid-view.
- [ ] Switch Wi-Fi off before pressing Stop: the error keeps the set and Try again works once Wi-Fi is back.

## What to send back
On each phone, after a recording, press **Copy report** on the result screen and paste it into a note. Add the phone model, browser and version, and what felt wrong. The numbers that matter: `poseModel.delegate` (GPU is expected), `liveFps` (aim for 15 or more), `poseMsPerFrame`, `recordedFps`, `serverStatus`.
