"""
Evaluation script for CnnPolicy models trained via train_cnn.py.

Replicates the training pipeline (VecFrameStack + VecTransposeImage) because the
CNN expects (12, 72, 80) uint8 input — 4 stacked frames × 3 RGB channels, transposed
to channels-first for PyTorch. VecNormalize is intentionally omitted: in PPO_CNN_*
we only normalize rewards (norm_obs=False), and the policy network operates on
unnormalized pixel obs identically in eval and training.

Usage:
    python -m agents.rl.evaluate_cnn --model runs/checkpoints/PPO_CNN_2/PPO_CNN_2_20000000_steps.zip
    python -m agents.rl.evaluate_cnn --model <path> --episodes 20 --gif --deterministic
    python -m agents.rl.evaluate_cnn --model <path> --state saves/violet_city.state
"""

import argparse
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from env.pokemon_env_cnn import PokemonEnvCNN
from agents.rl import config


def build_vec_env(state_path: str, gif_dir: str | None, watch: bool):
    """Build the same wrapper stack used in training, minus VecNormalize.

    If watch=True, the underlying PyBoy opens an SDL2 window so you can see
    the agent play in real time. Use DummyVecEnv only (single process), since
    SDL2 + SubprocVecEnv is unreliable.

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
        )
        ref.append(e)
        return e

    vec = DummyVecEnv([_init])
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    return vec, ref[0]


def evaluate(model_path: str, n_episodes: int, state_path: str,
             deterministic: bool, gif_dir: str | None,
             watch: bool, max_steps: int | None):
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

    vec, underlying = build_vec_env(state_path, gif_dir, watch)
    model = PPO.load(model_path, env=vec, device=device)

    results = []
    obs = vec.reset()

    for ep in range(1, n_episodes + 1):
        total_reward = 0.0
        steps = 0
        max_tiles = 0
        battle_steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec.step(action)
            total_reward += float(rewards[0])
            steps += 1

            info = infos[0]
            max_tiles = max(max_tiles, info.get("visited_tiles", 0))
            battle_steps += int(info.get("in_battle", 0))

            if dones[0]:
                break
            if max_steps is not None and steps >= max_steps:
                break

        # VecEnv auto-resets BEFORE returning the final obs of an episode,
        # so reading the badge bit now would reflect the NEW episode, not the one
        # just completed. Workaround: peek at the previous episode's final info.
        # infos[0] still carries the terminal info dict from the just-ended episode.
        got_badge = bool(infos[0].get("zephyr", False))
        # Fallback if terminal_observation key isn't present
        if "TimeLimit.truncated" in infos[0]:
            truncated = infos[0]["TimeLimit.truncated"]
        else:
            truncated = False

        battle_pct = (battle_steps / steps * 100) if steps else 0.0
        results.append({
            "episode": ep,
            "reward":  total_reward,
            "steps":   steps,
            "tiles":   max_tiles,
            "badge":   got_badge,
            "in_battle_pct": battle_pct,
        })

        print(
            f"Ep {ep:3d} | reward={total_reward:+8.1f} | steps={steps:5d} | "
            f"tiles={max_tiles:4d} | in_battle={battle_pct:5.1f}% | "
            f"badge={'YES' if got_badge else 'no'}"
        )

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
    print(f"  Average reward: {avg_reward:+.1f} ± {std_reward:.1f}")
    print(f"  Average steps:  {avg_steps:.0f} ± {std_steps:.0f}")
    print(f"  Average tiles:  {avg_tiles:.0f} ± {std_tiles:.0f}")
    print(f"  Average in_battle: {avg_battle:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved .zip model (e.g. runs/checkpoints/PPO_CNN_2/PPO_CNN_2_20000000_steps.zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--state", type=str, default=config.STATE_PATH,
                        help=f"Save state to evaluate from (default: {config.STATE_PATH})")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (no action sampling)")
    parser.add_argument("--gif", action="store_true",
                        help="Capture a GIF per episode under runs/eval_gifs/")
    parser.add_argument("--watch", action="store_true",
                        help="Open SDL2 window to watch the agent play live")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Truncate each episode after N steps (useful with --watch for quick previews)")
    args = parser.parse_args()

    gif_dir = "runs/eval_gifs" if args.gif else None
    if gif_dir:
        os.makedirs(gif_dir, exist_ok=True)

    evaluate(args.model, args.episodes, args.state, args.deterministic,
             gif_dir, args.watch, args.max_steps)
