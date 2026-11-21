#!/bin/bash
# Builds the fake-webcam clips from the Keraal Group 3 videos (local use only; do not publish frames of the participants).
# Needs ffmpeg and data/keraal/videos. Chrome plays the .mjpeg files as its camera; the file-upload test uses clip.webm.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; V="$HERE/../../data/keraal/videos"; mkdir -p "$HERE/cam"
for ex in ELK CTK RTK; do ffmpeg -v error -y -i "$V/G3-Anon-$ex-P1T1-Unknown-C-0.avi" -c:v copy -f mjpeg "$HERE/cam/$ex.mjpeg"; done
ffmpeg -v error -y -f mjpeg -framerate 10 -i "$HERE/cam/ELK.mjpeg" -t 12 -c:v libvpx -b:v 700k -an "$HERE/cam/clip.webm"
echo "clips ready in $HERE/cam"
