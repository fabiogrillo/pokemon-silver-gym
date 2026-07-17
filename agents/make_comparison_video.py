"""Render the FULL navigation comparison video at true 4x speed (no frame subsampling).

Complements agents/make_comparison_gif.py (the short highlight montage): this renders every
recorded env-step exactly once, so with one step = 24 game frames = 0.4 s, playback at 10 fps
is exactly 4.0x real time — slow enough to actually follow the gameplay. Frames stream
straight into ffmpeg instead of materializing a GIF first (a 10-minute GIF would not fit in
memory).

Usage:
  .venv/bin/python -m agents.make_comparison_video   # writes assets/comparison_full.mp4

The output is intentionally gitignored (~80 MB); it is built locally and hosted off-repo.
"""

import subprocess

import numpy as np

from agents.make_comparison_gif import (
    CANVAS_H, CANVAS_W, compose_frame, load_frames, load_trace_thoughts,
    shorten_thought, thought_at, wrap_caption,
)

FPS = 10  # 10 shown steps/s * 0.4 s game time per step = 4.0x real time

CHAPTERS = [
    {
        "title": "Ch.1 New Bark -> Violet Gym",
        "rl_dir": "runs/comparison_footage/rl_corridor_091", "rl": (0, 5352),
        "llm_dir": "runs/comparison_footage/llm_corridor",
        "llm": (0, 676), "llm_trace": "runs/llm_logs/run_1783971285.jsonl",
    },
    {
        "title": "Ch.2 Violet Gym: 2 trainers + Falkner",
        "rl_dir": "runs/comparison_footage/rl", "rl": (0, 843),
        "llm_dir": "runs/comparison_footage/llm_gym",
        "llm": (1, 984), "llm_trace": "runs/llm_logs/run_1783953416.jsonl",
    },
]

OUT = "assets/comparison_full.mp4"


def main():
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CANVAS_W}x{CANVAS_H}",
         "-r", str(FPS), "-i", "-",
         "-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE,
    )

    t = 0.0
    for ch in CHAPTERS:
        print(f"[render] {ch['title']} starts at t={t:.1f}s", flush=True)
        rl_frames = load_frames(ch["rl_dir"], *ch["rl"])
        llm_frames = load_frames(ch["llm_dir"], *ch["llm"])
        thoughts = load_trace_thoughts(ch["llm_trace"])
        n = len(rl_frames)  # one output frame per RL env-step -> exact 4x
        hold = 2 * FPS      # captions hold 2 s so the reasoning is readable
        llm_pick = np.linspace(0, len(llm_frames) - 1, n).round().astype(int)
        for k in range(n):
            anchor_step = llm_frames[llm_pick[(k // hold) * hold]][0]
            text = thought_at(thoughts, anchor_step)
            caption = wrap_caption(shorten_thought(text))
            frame = compose_frame(rl_frames[k][1], llm_frames[llm_pick[k]][1],
                                  caption, ch["title"])
            ff.stdin.write(frame.tobytes())
        t += n / FPS
        del rl_frames, llm_frames

    ff.stdin.close()
    ff.wait()
    print(f"[render] done, total {t:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
