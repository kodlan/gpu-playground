#!/usr/bin/env bash
# PPM frames -> MP4 (and a GIF).
#
#   ./make_video.sh [--crop square|vertical] [--size PX] [dir] [out] [fps]
#
# --crop zooms in on the colour border, where the ships fly, by cutting a
# window out of every frame centred on the middle row:
#   square    a SIZE x SIZE window
#   vertical  a 9:16 window, SIZE wide and SIZE * 16 / 9 tall (phone-screen shape)
# SIZE defaults to a quarter of the frame height (512 px on full-size frames,
# 128 on the default shrunk ones). The frame size is read from the first PPM.
set -euo pipefail

crop=""; size=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --crop) crop=$2; shift 2 ;;
        --size) size=$2; shift 2 ;;
        --crop=*) crop=${1#--crop=}; shift ;;
        --size=*) size=${1#--size=}; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        --*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
dir=${1:-frames}; out=${2:-life_wars.mp4}; fps=${3:-30}

command -v ffmpeg >/dev/null || { echo "install ffmpeg (apt install ffmpeg)"; exit 1; }
first=$dir/frame_00000.ppm
[[ -f $first ]] || { echo "no $first; run life_wars with --frame-every first" >&2; exit 1; }

# frame size from the PPM header: "P6", "W H", "255"
read -r fw fh < <(head -c 64 "$first" | sed -n '2p')

filter="pad=ceil(iw/2)*2:ceil(ih/2)*2"   # yuv420p needs even dimensions; a balanced band split can give an odd height
if [[ -n $crop ]]; then
    size=${size:-$((fh / 4))}
    case $crop in
        square)   cw=$size; ch=$size ;;
        vertical) cw=$size; ch=$((size * 16 / 9)) ;;
        *) echo "--crop must be square or vertical" >&2; exit 2 ;;
    esac
    cw=$((cw / 2 * 2)); ch=$((ch / 2 * 2))                       # even
    (( cw <= fw && ch <= fh )) || { echo "crop ${cw}x${ch} does not fit the ${fw}x${fh} frames; lower --size" >&2; exit 2; }
    cx=$(((fw - cw) / 2)); cy=$(((fh - ch) / 2))                 # centred on the border row
    filter="crop=$cw:$ch:$cx:$cy"
    echo "cropping ${cw}x${ch} at ($cx,$cy) out of ${fw}x${fh} frames"
fi

ffmpeg -y -loglevel error -framerate "$fps" -i "$dir/frame_%05d.ppm" -vf "$filter" -c:v libx264 -pix_fmt yuv420p -crf 20 "$out"
ffmpeg -y -loglevel error -i "$out" -vf "fps=15,scale=512:-1:flags=neighbor" "${out%.mp4}.gif"
echo "wrote $out and ${out%.mp4}.gif"
