"""Compose the RL-vs-LLM side-by-side highlight montage (GIF + MP4).

Reads two footage dirs (agents/rl/record_run.py and agents/llm/run.py --record-dir),
a segments JSON (curated step ranges), and renders: left panel RL, right panel LLM
with its reasoning captioned underneath.

Each segment supports optional keys on top of the required {"seconds", "rl", "llm"}:
  - "rl_dir" / "llm_dir": per-segment footage dirs (else the CLI defaults), so one montage
    can mix chapters shot from different runs (corridor vs gym).
  - "llm_trace": path to the run's JSONL trace (agents/llm/run.py). The LLM footage has one
    frame per ReAct step and the trace one line per step, so frame index == trace step; the
    caption then follows the model's LIVE reasoning as the clip plays instead of one fixed
    quote. Captions hold for "caption_seconds" (default 2.0) so they stay readable — one
    thought per turn would flip faster than anyone can read.
  - "caption": fixed caption, used when there is no trace (or as fallback for steps with
    no reasoning text yet).
  - "title": chapter title drawn in the caption band under the RL panel.

Usage:
  .venv/bin/python -m agents.make_comparison_gif \
      --segments assets/comparison_segments.json \
      --out assets/comparison
"""

import argparse
import bisect
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
MARGIN, HEADER_H, CAPTION_H = 16, 40, 90   # caption band fits 3 wrapped lines + chapter title
CANVAS_W = MARGIN + PANEL_W + MARGIN + PANEL_W + MARGIN   # 688
CANVAS_H = HEADER_H + PANEL_H + CAPTION_H                 # 418

RL_LABEL = "RL - PPO"
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


def wrap_caption(text: str, width: int = 34, max_lines: int = 3) -> list:
    """Wrap text to fit the caption panel width, marking truncation honestly.

    width=34 chars fits within the ~320 px right-panel margin at the 15-px
    mono caption font. If textwrap drops lines beyond max_lines, the last
    kept line gets a trailing "…" (trimmed to stay within width) so readers
    know the caption was cut, instead of silently losing the ellipsis that
    shorten_thought already appended.
    """
    all_lines = textwrap.wrap(text, width=width)
    lines = all_lines[:max_lines]
    if len(all_lines) > max_lines and lines and not lines[-1].endswith("…"):
        last = lines[-1]
        if len(last) >= width:
            last = last[:width - 1]
        lines[-1] = last + "…"
    return lines


def upscale2x(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def compose_frame(rl_frame: np.ndarray, llm_frame: np.ndarray,
                  caption_lines: list, title: str = "") -> np.ndarray:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)
    label_font, caption_font = _font(18), _font(15)

    lx, rx = MARGIN, MARGIN + PANEL_W + MARGIN
    draw.text((lx, 10), RL_LABEL, fill=(120, 200, 255), font=label_font)
    draw.text((rx, 10), LLM_LABEL, fill=(255, 190, 120), font=label_font)

    canvas.paste(Image.fromarray(upscale2x(rl_frame)), (lx, HEADER_H))
    canvas.paste(Image.fromarray(upscale2x(llm_frame)), (rx, HEADER_H))

    y = HEADER_H + PANEL_H + 10
    if title:  # chapter title lives under the RL panel, left of the reasoning caption
        draw.text((lx, y), title, fill=(150, 150, 160), font=caption_font)
    for line in caption_lines:
        draw.text((rx, y), line, fill=(230, 230, 230), font=caption_font)
        y += 22
    return np.asarray(canvas, dtype=np.uint8)


def load_frames(dir_path: str, start: int, end: int) -> list:
    """[(source frame index, frame)] for the frames present in [start, end)."""
    frames = []
    for i in range(start, end):
        path = os.path.join(dir_path, f"frame_{i:05d}.png")
        if os.path.exists(path):
            frames.append((i, iio.imread(path)[:, :, :3]))
    return frames


def load_trace_thoughts(trace_path: str) -> list:
    """[(step, thought)] sorted by step, one entry per trace line with reasoning text.

    agents/llm/run.py writes one trace line AND one recorded frame per ReAct step, so these
    steps line up 1:1 with the source frame indices of the LLM footage.
    """
    out = []
    with open(trace_path) as f:
        for line in f:
            row = json.loads(line)
            thought = (row.get("thought") or "").strip()
            if "step" in row and thought:
                out.append((row["step"], thought))
    out.sort()
    return out


def thought_at(thoughts: list, step: int) -> str:
    """The most recent reasoning text at or before `step` ("" if none yet)."""
    i = bisect.bisect_right(thoughts, (step, "￿"))
    return thoughts[i - 1][1] if i else ""


def build_montage(segments: list, rl_dir: str, llm_dir: str, fps: int) -> list:
    out = []
    for seg in segments:
        n = int(round(seg["seconds"] * fps))
        seg_rl_dir = seg.get("rl_dir", rl_dir)
        seg_llm_dir = seg.get("llm_dir", llm_dir)
        rl_frames = load_frames(seg_rl_dir, *seg["rl"])
        if not rl_frames:
            raise ValueError(f"rl segment range {seg['rl']} yielded no frames in {seg_rl_dir}")
        llm_frames = load_frames(seg_llm_dir, *seg["llm"])
        if not llm_frames:
            raise ValueError(f"llm segment range {seg['llm']} yielded no frames in {seg_llm_dir}")

        thoughts = load_trace_thoughts(seg["llm_trace"]) if "llm_trace" in seg else None
        # Hold each caption for caption_seconds so it can actually be read: the LLM reasons
        # once per frame, which is far faster than a viewer reads.
        hold = max(1, int(round(seg.get("caption_seconds", 2.0) * fps)))
        title = seg.get("title", "")

        rl_pick = np.linspace(0, len(rl_frames) - 1, n).round().astype(int)
        llm_pick = np.linspace(0, len(llm_frames) - 1, n).round().astype(int)
        for k in range(n):
            text = seg.get("caption", "")
            if thoughts is not None:
                anchor_step = llm_frames[llm_pick[(k // hold) * hold]][0]
                text = thought_at(thoughts, anchor_step) or text
            caption = wrap_caption(shorten_thought(text))
            out.append(compose_frame(rl_frames[rl_pick[k]][1], llm_frames[llm_pick[k]][1],
                                     caption, title))
    return out


def export(frames: list, out_base: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    gif_path, mp4_path = out_base + ".gif", out_base + ".mp4"
    iio.imwrite(gif_path, np.stack(frames), plugin="pillow", loop=0,
                duration=int(round(1000 / fps)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", gif_path,
         "-c:v", "libopenh264", "-movflags", "+faststart", "-pix_fmt", "yuv420p", mp4_path],
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
