#!/bin/bash
# Runs every browser scenario against http://localhost:8000 (start `python app/server.py` first).
# Also needs the design prototype server for the synthetic-pose helper:  cd design && python3 -m http.server 8010 --bind 127.0.0.1
cd "$(dirname "$0")"
pkill -f "[u]ser-data-dir=/tmp/cdp-" 2>/dev/null; true   # clear any browser left behind by an earlier crashed run
[ -f cam/ELK.mjpeg ] || ./make_cams.sh
for g in g1 g2 g3a g3b g3c g3d g4 g5 g6; do echo "=== $g"; python3 $g.py 2>&1 | grep -E "^(PASS|FAIL)|Traceback|Error"; done
