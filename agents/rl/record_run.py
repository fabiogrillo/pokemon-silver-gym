"""Record per-step frames + RAM metadata of a trained CNN policy for montage footage.

Unlike make_gif.py (which renders a final GIF directly), this dumps raw material:
one PNG per env-step plus a meta.jsonl (map, position, battle flag, badge bit) so a
composer can pick highlight segments by step index afterwards.

Usage:
  .venv/bin/python -m agents.rl.record_run \
      --model runs/checkpoints/agent_087/agent_087_final.zip \
      --state saves/violet_city_gym.state \
      --out runs/comparison_footage/rl --max-steps 2000
"""

import argparse
import json
import os

import imageio.v3 as iio
import torch
from stable_baselines3 import PPO

from agents.rl.evaluate_cnn import build_vec_env


def record(model_path, state_path, out_dir, max_steps, deterministic):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False)
    model = PPO.load(model_path, device=device)

    obs = vec.reset()
    steps = 0
    with open(os.path.join(out_dir, "meta.jsonl"), "w", buffering=1) as f:
        for step in range(max_steps):
            iio.imwrite(os.path.join(out_dir, f"frame_{step:05d}.png"),
                        env.pyboy.pyboy.screen.ndarray[:, :, :3].copy())
            s = env.ram_reader.read_all()
            f.write(json.dumps({
                "step": step, "map": [s["map_bank"], s["map_number"]],
                "pos": [s["local_x"], s["local_y"]], "battle": s["battle_type"],
                "zephyr": s["zephyr"],
            }) + "\n")
            steps = step + 1
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, _ = vec.step(action)
            if dones[0]:
                break
    vec.close()
    print(f"[record] wrote {steps} frames + meta.jsonl to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", default="saves/violet_city_gym.state")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    record(a.model, a.state, a.out, a.max_steps, a.deterministic)
