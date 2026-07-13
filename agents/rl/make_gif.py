"""
Render a watchable GIF of a trained CNN policy playing, at a controlled (slow) speed.

Why this exists separately from `evaluate_cnn --gif`: the in-env GIF capture samples every 3rd
env-step and renders at 12 fps, i.e. ~10x real time — too fast to see what the agent is doing.
This script captures the full-resolution Game Boy screen and renders at a chosen `--speed`
(default 2x, the cap requested for demo clips).

Speed math: 1 env-step advances the emulator 16 frames = 16/60 s = 0.2667 s of game time. To play
back at `speed` x real time, each captured frame is shown for (0.2667 / speed) s. `--frame-skip K`
keeps every K-th env-step (smaller file) while preserving the playback speed by scaling the per-frame
duration. Clip length on screen ≈ (max_steps * 0.2667 / speed) seconds — so a long in-game journey at
<=2x is inherently a long clip; use a smaller --max-steps for a short demo, larger to show the full task.

Usage:
  .venv/bin/python -m agents.rl.make_gif \
      --model runs/checkpoints/agent_090/agent_090_50000000_steps.zip \
      --state saves/start.state --max-steps 4000 --speed 2 --out runs/progress_gifs/agent_090.gif
"""

import argparse
import os
import numpy as np
import imageio.v3 as iio
import torch
from stable_baselines3 import PPO
from agents.rl.evaluate_cnn import build_vec_env, checkpoint_visited_obs

SEC_PER_ENV_STEP = 16.0 / 60.0  # 16 emulator frames per env-step at 60 fps


def make_gif(model_path, state_path, out_path, max_steps, speed, frame_skip, deterministic):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gif] model={os.path.basename(model_path)} state={os.path.basename(state_path)} "
          f"max_steps={max_steps} speed={speed}x frame_skip={frame_skip}")
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False,
                             visited_obs=checkpoint_visited_obs(model_path))
    model = PPO.load(model_path, device=device)

    obs = vec.reset()
    frames = []
    for step in range(max_steps):
        if step % frame_skip == 0:
            frames.append(env.pyboy.pyboy.screen.ndarray[:, :, :3].copy())  # full-res GB screen
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, _ = vec.step(action)
        if dones[0]:
            frames.append(env.pyboy.pyboy.screen.ndarray[:, :, :3].copy())
            break
    vec.close()

    # per-frame display time so playback = `speed` x real time, given each frame spans `frame_skip` env-steps
    duration_ms = int(round(frame_skip * SEC_PER_ENV_STEP / speed * 1000))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iio.imwrite(out_path, np.stack(frames), plugin="pillow", loop=0, duration=duration_ms)
    clip_s = len(frames) * duration_ms / 1000.0
    print(f"[gif] wrote {out_path}: {len(frames)} frames, {duration_ms} ms/frame -> {clip_s:.1f}s clip "
          f"({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", default="saves/start.state")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--speed", type=float, default=2.0, help="playback x real-time (<=2 recommended)")
    p.add_argument("--frame-skip", type=int, default=2, help="capture every K-th env-step (smaller file)")
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    make_gif(a.model, a.state, a.out, a.max_steps, a.speed, a.frame_skip, a.deterministic)
