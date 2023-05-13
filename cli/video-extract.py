#!/usr/bin/env python
"""
use ffmpeg for animation processing
"""
import os
import json
import subprocess
import pathlib
import argparse
import filetype
from util import log, Map


def probe(src: str):
    cmd = f"ffprobe -hide_banner -loglevel 0 -print_format json -show_format -show_streams \"{src}\""
    result = subprocess.run(cmd, shell = True, capture_output = True, text = True, check = True)
    data = json.loads(result.stdout)
    stream = [x for x in data['streams'] if x["codec_type"] == "video"][0]
    fmt = data['format'] if 'format' in data else {}
    res = {**stream, **fmt}
    video = Map({
        'codec': res.get('codec_name', 'unknown') + '/' + res.get('codec_tag_string', ''),
        'resolution': [int(res.get('width', 0)), int(res.get('height', 0))],
        'duration': float(res.get('duration', 0)),
        'frames': int(res.get('nb_frames', 0)),
        'bitrate': round(float(res.get('bit_rate', 0)) / 1024),
    })
    return video


def extract(src: str, dst: str, rate: float = 0.015, fps: float = 0, start = 0, end = 0):
    images = []
    if not os.path.isfile(src) or not filetype.is_video(src):
        log.error({ 'extract': 'input is not movie file' })
        return
    dst = dst if dst.endswith('/') else dst + '/'

    video = probe(src)
    log.info({ 'extract': { 'source': src, **video } })

    ssstart = f' -ss {start}' if start > 0 else ''
    ssend = f' -to {video.duration - end}' if start > 0 else ''
    filename = pathlib.Path(src).stem
    if rate > 0:
        cmd = f"ffmpeg -hide_banner -y -loglevel info {ssstart} {ssend} -i \"{src}\" -filter:v \"select='gt(scene,{rate})',metadata=print\" -vsync vfr -frame_pts 1 \"{dst}{filename}-%05d.jpg\""
    elif fps > 0:
        cmd = f"ffmpeg -hide_banner -y -loglevel info {ssstart} {ssend} -i \"{src}\" -r {fps} -vsync vfr -frame_pts 1 \"{dst}{filename}-%05d.jpg\""
    else:
        log.error({ 'extract': 'requires either rate or fps' })
        return 0
    log.debug({ 'extract': cmd })
    pathlib.Path(dst).mkdir(parents = True, exist_ok = True)
    result = subprocess.run(cmd, shell = True, capture_output = True, text = True, check = True)
    for line in result.stderr.split('\n'):
        if 'pts_time' in line:
            log.debug({ 'extract': { 'keyframe': line.strip().split(' ')[-1].split(':')[-1] } })
    images = next(os.walk(dst))[2]
    log.info({ 'extract': { 'destination': dst, 'keyframes': len(images), 'rate': rate, 'fps': fps } })
    return len(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ffmpeg pipeline")
    parser.add_argument("--input", type = str, required = True, help="input")
    parser.add_argument("--output", type = str, required = True, help="output")
    parser.add_argument("--rate", type = float, default = 0, required = False, help="extraction change rate threshold")
    parser.add_argument("--fps", type = float, default = 0, required = False, help="extraction frames per second")
    parser.add_argument("--skipstart", type = float, default = 1, required = False, help="skip time from start of video")
    parser.add_argument("--skipend", type = float, default = 1, required = False, help="skip time to end of video")
    params = parser.parse_args()
    extract(src = params.input, dst = params.output, rate = params.rate, fps = params.fps, start = params.skipstart, end = params.skipend)
