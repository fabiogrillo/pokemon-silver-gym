"""
Play a pretrained checkpoint and report its journey toward Falkner (default: the gym agent).

This is the Dockerized demo entrypoint (see /Dockerfile): it loads a checkpoint, runs one (or a few)
episodes from a save state (default saves/violet_city_gym.state via the STATE env var), prints the
milestones reached (waypoints, trainers, badge), and optionally writes the map-trajectory overlay
(agents/rl/visualize_map) and a watchable gameplay GIF (agents/rl/make_gif).

The ROM is NOT bundled in the image (legal reasons) — mount it at runtime. Inference runs fine on CPU.

Usage (local):
  python -m agents.rl.play --model runs/checkpoints/agent_090/agent_090_50000000_steps.zip --map --gif

Usage (Docker): see Dockerfile / README. Env vars MODEL, STATE, MAX_STEPS act as defaults.
"""

import argparse
import os

import torch
from stable_baselines3 import PPO

from agents.rl.evaluate_cnn import build_vec_env, checkpoint_visited_obs
from agents.rl import map_layout as ml

WAYPOINT_NAMES = ["start", "cherrygrove", "route30_gate", "route31", "violet", "gym"]


def play(model_path, state_path, max_steps, deterministic, make_map, make_gif_clip):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[play] model={os.path.basename(model_path)} state={os.path.basename(state_path)} "
          f"device={device} max_steps={max_steps}")

    model = PPO.load(model_path, device=device)
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False,
                             visited_obs="visited" in model.observation_space.spaces)
    obs = vec.reset()

    positions = []
    best_waypoint = 0
    badge = False
    last_announced = 0
    for step in range(max_steps):
        ram = env.ram_reader.read_all()
        positions.append((ram["map_bank"], ram["map_number"], ram["local_x"], ram["local_y"]))

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = vec.step(action)
        info = infos[0]

        wp = int(info.get("max_waypoint", 0))
        if wp > last_announced:                       # announce each newly-reached waypoint
            last_announced = wp
            print(f"[play] step {step:6d} → reached {WAYPOINT_NAMES[wp]}")
        best_waypoint = max(best_waypoint, wp)
        if info.get("zephyr"):
            badge = True
            print(f"[play] step {step:6d} → 🏅 ZEPHYR BADGE — Falkner defeated!")
        if dones[0]:
            break
    vec.close()

    print(f"[play] done: furthest = {WAYPOINT_NAMES[best_waypoint]} | badge = {badge} | steps = {len(positions)}")

    if make_map:
        from agents.rl.visualize_map import render
        out = os.path.join("runs", "maps", f"play_{os.path.splitext(os.path.basename(model_path))[0]}")
        render(positions, out, title=os.path.basename(model_path))

    if make_gif_clip:
        from agents.rl.make_gif import make_gif
        out = os.path.join("runs", "play_gifs", f"{os.path.splitext(os.path.basename(model_path))[0]}.gif")
        make_gif(model_path, state_path, out, max_steps, speed=2.0, frame_skip=2, deterministic=deterministic)

    return badge, best_waypoint


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL"), help="checkpoint .zip (or MODEL env var)")
    p.add_argument("--state", default=os.environ.get("STATE", "saves/egg_delivered_clean.state"))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "20000")))
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--map", action="store_true", help="also write the trajectory/heatmap overlay PNG+GIF")
    p.add_argument("--gif", action="store_true", help="also write a watchable gameplay GIF")
    a = p.parse_args()
    if not a.model:
        raise SystemExit("Provide --model <ckpt.zip> or set the MODEL env var")
    play(a.model, a.state, a.max_steps, a.deterministic, a.map, a.gif)
