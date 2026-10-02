#!/usr/bin/env bash
# PPM frames -> MP4 (and a GIF).
set -euo pipefail
dir=${1:-frames}; out=${2:-life_wars.mp4}; fps=${3:-30}
command -v ffmpeg >/dev/null || { echo "install ffmpeg (apt install ffmpeg)"; exit 1; }
# pad to even dimensions: yuv420p needs them, and a balanced band split can give an odd height
ffmpeg -y -loglevel error -framerate "$fps" -i "$dir/frame_%05d.ppm" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 20 "$out"
ffmpeg -y -loglevel error -i "$out" -vf "fps=15,scale=512:-1" "${out%.mp4}.gif"
echo "wrote $out and ${out%.mp4}.gif"