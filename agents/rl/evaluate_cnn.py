"""
Evaluation script for CnnPolicy models trained via train_cnn.py.

Replicates the training pipeline (VecFrameStack + VecTransposeImage) because the
CNN expects (12, 72, 80) uint8 input — 4 stacked frames × 3 RGB channels, transposed
to channels-first for PyTorch. VecNormalize is intentionally omitted: in this project
we only normalize rewards (norm_obs=False), and the policy network operates on
unnormalized pixel obs identically in eval and training.

Usage:
    python -m agents.rl.evaluate_cnn --model runs/checkpoints/agent_090/agent_090_50000000_steps.zip
    python -m agents.rl.evaluate_cnn --model <path> --episodes 20 --gif --deterministic
    python -m agents.rl.evaluate_cnn --model <path> --state saves/violet_city.state
"""

import argparse
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from env.pokemon_env_cnn import PokemonEnvCNN
from agents.rl import config


def checkpoint_visited_obs(model_path: str) -> bool:
    """Whether the checkpoint was trained with the visited-coordinates Dict-obs key.

    Checkpoints from before VISITED_OBS (e.g. the gym agent) and after it have incompatible
    observation spaces; peeking at the space saved inside the zip lets every CLI tool build a
    matching env automatically instead of taking a flag the caller can get wrong.
    """
    data, _, _ = load_from_zip_file(model_path, device="cpu")
    return "visited" in getattr(data["observation_space"], "spaces", {})


def build_vec_env(state_path: str, gif_dir: str | None, watch: bool, speed: int = 0,
                  visited_obs: bool = False):
    """Build the same wrapper stack used in training, minus VecNormalize.

    If watch=True, the underlying PyBoy opens an SDL2 window so you can see
    the agent play in real time. Use DummyVecEnv only (single process), since
    SDL2 + SubprocVecEnv is unreliable.

    `visited_obs` must match the checkpoint being loaded (see checkpoint_visited_obs).

    Returns (vec_env, underlying_env) so the caller can read the badge bit
    from RAM directly at episode end.
    """
    ref = []

    def _init():
        e = PokemonEnvCNN(
            config.ROM_PATH, state_path, headless=not watch,
            gif_dir=gif_dir,
            gif_every_n_episodes=1 if gif_dir else 10**9,
            gif_prefix="eval",
            visited_obs=visited_obs,
            # confine_to_gym / confine_to_corridor are intentionally NOT passed here (both default
            # False in PokemonEnvCNN): eval must never confine the agent — we need the true,
            # un-clipped success/behavior signal, same as today's confine_to_gym handling.
        )
        # Throttle the emulator so a watched run is viewable (0 = unbounded, 1 = realtime, 2 = 2×).
        if watch and speed > 0:
            e.pyboy.pyboy.set_emulation_speed(speed)
        ref.append(e)
        return e

    vec = DummyVecEnv([_init])
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    return vec, ref[0]


ACTION_NAMES = ["up", "down", "left", "right", "a", "b", "start", "select"]
WAYPOINT_NAMES = ["start", "cherrygrove", "route30_gate", "route31", "violet", "gym"]


def evaluate(model_path: str, n_episodes: int, state_path: str,
             deterministic: bool, gif_dir: str | None,
             watch: bool, max_steps: int | None, log_path: str | None = None,
             speed: int = 0):
    if not torch.cuda.is_available():
        print("[device] CUDA not available — CNN inference on CPU will be slow but works.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"[device] CUDA → {torch.cuda.get_device_name(0)}")

    print(f"[eval] model:     {model_path}")
    print(f"[eval] state:     {state_path}")
    print(f"[eval] episodes:  {n_episodes}")
    print(f"[eval] deterministic: {deterministic}")
    print(f"[eval] gif_dir:   {gif_dir or '(disabled)'}")
    print(f"[eval] watch:     {watch} (SDL2 window)")
    print(f"[eval] max_steps: {max_steps or 'env default'}")
    print()

    vec, underlying = build_vec_env(state_path, gif_dir, watch, speed,
                                    visited_obs=checkpoint_visited_obs(model_path))
    model = PPO.load(model_path, env=vec, device=device)

    results = []
    obs = vec.reset()
    needs_reset = False  # True when the previous episode ended via --max-steps cap (no env done)

    for ep in range(1, n_episodes + 1):
        if needs_reset:
            # With --max-steps we cut the episode WITHOUT the env terminating: reset explicitly
            # so per-episode counters (battles, tiles, egg flags) start clean.
            obs = vec.reset()
            needs_reset = False
        total_reward = 0.0
        steps = 0
        max_tiles = 0
        battle_steps = 0
        action_counts = np.zeros(len(ACTION_NAMES), dtype=int)
        done_flag = False

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_counts[int(action[0])] += 1
            obs, rewards, dones, infos = vec.step(action)
            total_reward += float(rewards[0])
            steps += 1

            info = infos[0]
            max_tiles = max(max_tiles, info.get("visited_tiles", 0))
            battle_steps += int(info.get("in_battle", 0))

            if dones[0]:
                done_flag = True
                break
            if max_steps is not None and steps >= max_steps:
                break

        # VecEnv auto-resets BEFORE returning the final obs of an episode, but infos[0]
        # still carries the terminal info dict of the just-ended episode.
        info = infos[0]
        got_badge = bool(info.get("zephyr", False))
        if done_flag:
            end_cause = "badge" if got_badge else ("death" if info.get("hp_ratio", 1) <= 0 else "truncated")
        else:
            end_cause = "step_cap"
            needs_reset = True

        battle_pct = (battle_steps / steps * 100) if steps else 0.0
        top_actions = ", ".join(
            f"{ACTION_NAMES[i]}:{action_counts[i] / steps * 100:.0f}%"
            for i in np.argsort(action_counts)[::-1][:4]
        )
        ep_result = {
            "episode": ep,
            "reward":  round(total_reward, 2),
            "steps":   steps,
            "tiles":   int(max_tiles),
            "badge":   got_badge,
            "in_battle_pct": round(battle_pct, 1),
            "end_cause": end_cause,
            "max_waypoint": WAYPOINT_NAMES[int(info.get("max_waypoint", 0))],
            "egg_received": bool(info.get("egg_received", False)),
            "egg_delivered": bool(info.get("egg_delivered", False)),
            "route_trainers_beaten": int(info.get("route_trainers_beaten", 0)),
            "gym_trainers_beaten": int(info.get("gym_trainers_beaten", 0)),
            "battles_won": int(info.get("battles_won", 0)),
            "battles_fled": int(info.get("battles_fled", 0)),
            "battles_lost": int(info.get("battles_lost", 0)),
            "lead_level": int(info.get("lead_level", 0)),
            "action_counts": {ACTION_NAMES[i]: int(c) for i, c in enumerate(action_counts)},
        }
        results.append(ep_result)

        print(
            f"Ep {ep:3d} | reward={total_reward:+8.1f} | steps={steps:5d} | tiles={max_tiles:4d} | "
            f"in_battle={battle_pct:5.1f}% | wp={ep_result['max_waypoint']:13s} | "
            f"egg={'D' if ep_result['egg_delivered'] else ('R' if ep_result['egg_received'] else '-')} | "
            f"W/F/L={ep_result['battles_won']}/{ep_result['battles_fled']}/{ep_result['battles_lost']} | "
            f"lv={ep_result['lead_level']:2d} | end={end_cause}"
        )
        print(f"        top actions: {top_actions}")

        if log_path:
            import json
            with open(log_path, "a") as f:
                f.write(json.dumps({"model": os.path.basename(model_path),
                                    "state": os.path.basename(state_path),
                                    "deterministic": deterministic, **ep_result}) + "\n")

    vec.close()

    badge_rate = sum(r["badge"] for r in results) / n_episodes
    avg_reward = np.mean([r["reward"] for r in results])
    std_reward = np.std([r["reward"] for r in results])
    avg_steps  = np.mean([r["steps"] for r in results])
    std_steps  = np.std([r["steps"] for r in results])
    avg_tiles  = np.mean([r["tiles"] for r in results])
    std_tiles  = np.std([r["tiles"] for r in results])
    avg_battle = np.mean([r["in_battle_pct"] for r in results])

    print()
    print(f"Summary over {n_episodes} episodes from {os.path.basename(state_path)}:")
    print(f"  Badge obtained: {badge_rate*100:.1f}%  ({sum(r['badge'] for r in results)}/{n_episodes})")
    print(f"  Egg received:   {sum(r['egg_received'] for r in results)}/{n_episodes}   delivered: {sum(r['egg_delivered'] for r in results)}/{n_episodes}")
    print(f"  Reached: " + ", ".join(f"{w}={sum(WAYPOINT_NAMES.index(r['max_waypoint']) >= i for r in results)}/{n_episodes}"
                                     for i, w in enumerate(WAYPOINT_NAMES) if i > 0))
    print(f"  Battles W/F/L:  {sum(r['battles_won'] for r in results)}/{sum(r['battles_fled'] for r in results)}/{sum(r['battles_lost'] for r in results)}")
    print(f"  Average reward: {avg_reward:+.1f} ± {std_reward:.1f}")
    print(f"  Average steps:  {avg_steps:.0f} ± {std_steps:.0f}")
    print(f"  Average tiles:  {avg_tiles:.0f} ± {std_tiles:.0f}")
    print(f"  Average in_battle: {avg_battle:.1f}%")
    if log_path:
        print(f"  Per-episode JSONL appended to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved .zip model (e.g. runs/checkpoints/agent_090/agent_090_50000000_steps.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--state", type=str, default=config.STATE_PATH,
                        help=f"Save state to evaluate from (default: {config.STATE_PATH})")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (no action sampling)")
    parser.add_argument("--gif", action="store_true",
                        help="Capture a GIF per episode under runs/eval_gifs/")
    parser.add_argument("--watch", action="store_true",
                        help="Open SDL2 window to watch the agent play live")
    parser.add_argument("--speed", type=int, default=2,
                        help="Emulation speed when --watch (0=unbounded, 1=realtime, 2=2x). Default 2.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Truncate each episode after N steps (useful with --watch for quick previews)")
    parser.add_argument("--log", nargs="?", const="auto", default=None,
                        help="Append per-episode JSONL results to a file (default: runs/eval_logs/<model>_<ts>.jsonl)")
    args = parser.parse_args()

    gif_dir = "runs/eval_gifs" if args.gif else None
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    log_path = None
    if args.log:
        if args.log == "auto":
            import time
            os.makedirs("runs/eval_logs", exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.model))[0]
            log_path = f"runs/eval_logs/{stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        else:
            log_path = args.log

    evaluate(args.model, args.episodes, args.state, args.deterministic,
             gif_dir, args.watch, args.max_steps, log_path, args.speed)
