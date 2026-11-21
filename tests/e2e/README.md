# Browser scenarios (end to end)

Drives headless Chrome over the DevTools protocol (standard library only, see `cdp.py`) with a **fake webcam that plays recorded Keraal clips**, so the real MediaPipe model runs on real people. 152 checks across: choose, set-up, camera errors, countdown, recording, retry, server failures, result, clinician replay, scrubbing, strictness, video files, orientation, widths from 320 to 1440, reduced motion.

What this is, and is not: it exercises the real app, the real server and the real pose model on **recorded** video. It is not a physical webcam on a phone; headless software rendering also runs the model at about 3 frames per second, so frame-rate and touch feel still need a real device.

Run:

```bash
python app/server.py                                   # terminal 1
(cd design && python3 -m http.server 8010 --bind 127.0.0.1)   # terminal 2: serves the synthetic pose helper
tests/e2e/run_all.sh                                   # terminal 3
```
Needs Google Chrome, ffmpeg and `data/keraal/videos`. `g2` and others use `design/shared/synth.js` (synthetic stick-figure poses) for error and borderline sets, because the real clips are all correct executions.
