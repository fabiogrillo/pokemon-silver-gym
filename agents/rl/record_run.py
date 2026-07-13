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

from agents.rl.evaluate_cnn import build_vec_env, checkpoint_visited_obs


def record(model_path, state_path, out_dir, max_steps, deterministic):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False,
                             visited_obs=checkpoint_visited_obs(model_path))
    model = PPO.load(model_path, device=device)

    # DummyVecEnv auto-resets the underlying env INSIDE step_wait() the moment
    # terminated/truncated is True, before control returns to this loop. That means
    # by the time vec.step() returns with dones[0]==True, env.pyboy already reflects
    # the NEXT episode's reset state (full HP, badge_count back to 0, spawn position) —
    # querying RAM/screen after the fact silently records reset artifacts instead of the
    # actual badge-won / death frame. Monkeypatch env.step to snapshot the screen + RAM
    # at the true terminal instant, while it's still live, before the wrapper's reset.
    terminal_capture = {}
    orig_step = env.step

    def _step_and_capture(action):
        result = orig_step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            terminal_capture["screen"] = env.pyboy.pyboy.screen.ndarray[:, :, :3].copy()
            terminal_capture["ram"] = env.ram_reader.read_all()
        return result

    env.step = _step_and_capture

    def _row(step, s):
        return {
            "step": step, "map": [s["map_bank"], s["map_number"]],
            "pos": [s["local_x"], s["local_y"]], "battle": s["battle_type"],
            "zephyr": s["zephyr"], "hp_ratio": s["hp_ratio"],
            "badge_count": s["badge_count"],
        }

    obs = vec.reset()
    steps = 0
    with open(os.path.join(out_dir, "meta.jsonl"), "w", buffering=1) as f:
        for step in range(max_steps):
            iio.imwrite(os.path.join(out_dir, f"frame_{step:05d}.png"),
                        env.pyboy.pyboy.screen.ndarray[:, :, :3].copy())
            s = env.ram_reader.read_all()
            f.write(json.dumps(_row(step, s)) + "\n")
            steps = step + 1
            action, _ = model.predict(obs, deterministic=deterministic)
            terminal_capture.clear()
            obs, _, dones, _ = vec.step(action)
            if dones[0]:
                # Use the pre-reset snapshot captured inside _step_and_capture, NOT a
                # fresh read here (the env has already been reset by this point).
                if terminal_capture:
                    iio.imwrite(os.path.join(out_dir, f"frame_{steps:05d}.png"),
                                terminal_capture["screen"])
                    f.write(json.dumps(_row(steps, terminal_capture["ram"])) + "\n")
                    steps += 1
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
