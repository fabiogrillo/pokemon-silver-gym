"""Compose the RL-vs-LLM side-by-side highlight montage (GIF + MP4).

Reads two footage dirs (agents/rl/record_run.py and agents/llm/run.py --record-dir),
a segments JSON (curated step ranges + real-thought captions), and renders:
left panel RL, right panel LLM with its reasoning captioned underneath.

Usage:
  .venv/bin/python -m agents.make_comparison_gif \
      --segments assets/comparison_segments.json \
      --out assets/comparison
"""

import argparse
import json
import os
import re
import subprocess
import textwrap

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Layout (all even — ffmpeg yuv420p needs even dims)
PANEL_W, PANEL_H = 320, 288          # 2x the 160x144 GB screen
MARGIN, HEADER_H, CAPTION_H = 16, 40, 72
CANVAS_W = MARGIN + PANEL_W + MARGIN + PANEL_W + MARGIN   # 688
CANVAS_H = HEADER_H + PANEL_H + CAPTION_H                 # 400

RL_LABEL = "RL - PPO (agent_087)"
LLM_LABEL = "LLM - qwen3-vl:8b (ReAct)"

_FONT_CANDIDATES = [
    "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.otf",
    "/usr/share/fonts/google-noto-vf/NotoSansMono[wght].ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]


def _font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def shorten_thought(text: str, max_len: int = 120) -> str:
    """Collapse whitespace and cut at a word boundary. Real text in, shorter real text out."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= max_len:
        return clean
    cut = clean[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def wrap_caption(text: str, width: int = 56, max_lines: int = 2) -> list:
    return textwrap.wrap(text, width=width)[:max_lines]


def upscale2x(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def compose_frame(rl_frame: np.ndarray, llm_frame: np.ndarray,
                  caption_lines: list) -> np.ndarray:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)
    label_font, caption_font = _font(18), _font(15)

    lx, rx = MARGIN, MARGIN + PANEL_W + MARGIN
    draw.text((lx, 10), RL_LABEL, fill=(120, 200, 255), font=label_font)
    draw.text((rx, 10), LLM_LABEL, fill=(255, 190, 120), font=label_font)

    canvas.paste(Image.fromarray(upscale2x(rl_frame)), (lx, HEADER_H))
    canvas.paste(Image.fromarray(upscale2x(llm_frame)), (rx, HEADER_H))

    y = HEADER_H + PANEL_H + 10
    for line in caption_lines:
        draw.text((rx, y), line, fill=(230, 230, 230), font=caption_font)
        y += 22
    return np.asarray(canvas, dtype=np.uint8)


def load_frames(dir_path: str, start: int, end: int) -> list:
    frames = []
    for i in range(start, end):
        path = os.path.join(dir_path, f"frame_{i:05d}.png")
        if os.path.exists(path):
            frames.append(iio.imread(path)[:, :, :3])
    return frames


def resample(frames: list, target_len: int) -> list:
    idx = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[i] for i in idx]


def build_montage(segments: list, rl_dir: str, llm_dir: str, fps: int) -> list:
    out = []
    for seg in segments:
        n = int(round(seg["seconds"] * fps))
        rl = resample(load_frames(rl_dir, *seg["rl"]), n)
        llm = resample(load_frames(llm_dir, *seg["llm"]), n)
        caption = wrap_caption(shorten_thought(seg.get("caption", "")))
        out.extend(compose_frame(r, l, caption) for r, l in zip(rl, llm))
    return out


def export(frames: list, out_base: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    gif_path, mp4_path = out_base + ".gif", out_base + ".mp4"
    iio.imwrite(gif_path, np.stack(frames), plugin="pillow", loop=0,
                duration=int(round(1000 / fps)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", gif_path,
         "-movflags", "+faststart", "-pix_fmt", "yuv420p", mp4_path],
        check=True,
    )
    for p in (gif_path, mp4_path):
        print(f"[montage] wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="segments JSON (fps, segments[])")
    ap.add_argument("--rl-dir", default="runs/comparison_footage/rl")
    ap.add_argument("--llm-dir", default="runs/comparison_footage/llm")
    ap.add_argument("--out", default="assets/comparison", help="output basename (no extension)")
    args = ap.parse_args()
    with open(args.segments) as f:
        spec = json.load(f)
    frames = build_montage(spec["segments"], args.rl_dir, args.llm_dir, spec.get("fps", 10))
    export(frames, args.out, spec.get("fps", 10))


if __name__ == "__main__":
    main()
