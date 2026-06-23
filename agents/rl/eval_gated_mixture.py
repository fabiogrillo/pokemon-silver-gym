"""
Gated mixture-of-experts eval (agent_064, distillation STEP 1 — validate composition).

Zero training: run a PICKUP teacher while the agent has no egg, switch to a DELIVERY teacher the
instant the egg is picked up (gated on the egg RAM bit, read every step). If this composite DELIVERS
from start.state, the two skills compose and the only barrier was packing them into one shared-RL net
→ proceed to BC-distill the mixture into one policy. If it does NOT compose even hand-gated, that's a
deeper problem (e.g. the pickup teacher leaves the agent in a pose the delivery teacher can't continue).

Teachers (same obs space, egg_marker OFF):
  pickup  = agent_053 (reliable start→Cherrygrove→Mr.Pokemon's→pickup)
  deliver = agent_062 (reliable carry→Elm delivery in its frontier episodes)

Run:
  .venv/bin/python -m agents.rl.eval_gated_mixture \
      --pickup runs/checkpoints/agent_053/agent_053_79999872_steps.zip \
      --deliver runs/checkpoints/agent_062/agent_062_39999936_steps.zip \
      --episodes 20 --max-steps 20000
"""

import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from agents.rl.evaluate_cnn import build_vec_env, WAYPOINT_NAMES


def evaluate_gated(pickup_path, deliver_path, n_episodes, state_path, max_steps):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[gated] pickup  teacher: {pickup_path}")
    print(f"[gated] deliver teacher: {deliver_path}")
    print(f"[gated] {n_episodes} eps from {state_path}, stochastic, cap {max_steps}\n")

    vec, _ = build_vec_env(state_path, gif_dir=None, watch=False)
    m_pickup = PPO.load(pickup_path, device=device)
    m_deliver = PPO.load(deliver_path, device=device)

    received = delivered = 0
    wp_counts = {n: 0 for n in WAYPOINT_NAMES}
    obs = vec.reset()
    needs_reset = False

    for ep in range(1, n_episodes + 1):
        if needs_reset:
            obs = vec.reset()
            needs_reset = False
        egg = False          # start.state: no egg → pickup teacher first
        ep_received = ep_delivered = False
        max_wp = 0
        switches = 0
        steps = 0
        while True:
            model = m_deliver if egg else m_pickup
            action, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = vec.step(action)
            steps += 1
            info = infos[0]
            new_egg = bool(info.get("egg_received", False))
            if new_egg != egg:
                switches += 1
            egg = new_egg
            ep_received = ep_received or new_egg
            ep_delivered = ep_delivered or bool(info.get("egg_delivered", False))
            max_wp = max(max_wp, int(info.get("max_waypoint", 0)))
            if dones[0]:
                break
            if max_steps is not None and steps >= max_steps:
                needs_reset = True
                break

        received += int(ep_received)
        delivered += int(ep_delivered)
        wp_counts[WAYPOINT_NAMES[max_wp]] += 1
        print(f"Ep {ep:2d} | steps={steps:6d} | egg={'R' if ep_received else '-'}"
              f"{'D' if ep_delivered else ' '} | wp={WAYPOINT_NAMES[max_wp]:13s} | switches={switches}")

    print(f"\n=== Gated mixture over {n_episodes} eps from start.state ===")
    print(f"  Egg received:  {received}/{n_episodes}")
    print(f"  Egg DELIVERED: {delivered}/{n_episodes}   <-- the composition test")
    print(f"  Waypoints: " + ", ".join(f"{n}={c}" for n, c in wp_counts.items() if c))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pickup", required=True)
    p.add_argument("--deliver", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--state", default="saves/start.state")
    p.add_argument("--max-steps", type=int, default=20000)
    a = p.parse_args()
    evaluate_gated(a.pickup, a.deliver, a.episodes, a.state, a.max_steps)
