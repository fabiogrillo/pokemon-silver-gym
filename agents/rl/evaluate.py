import argparse
import numpy as np
from stable_baselines3 import PPO
from env.pokemon_env_mlp import PokemonEnvMLP
from agents.rl import config


def evaluate(model_path: str, n_episodes: int, render: bool):
    env = PokemonEnvMLP(config.ROM_PATH, config.STATE_PATH, headless=not render)
    model = PPO.load(model_path, env=env, device="cpu")

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        steps = 0
        max_tiles = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            max_tiles = max(max_tiles, info.get("visited_tiles", 0))

        # Check if the badge was obtained by reading the RAM state at the end of the episode
        got_badge = bool(env.ram_reader.read_all()["zephyr"])

        results.append({
            "episode": ep + 1,
            "reward":  total_reward,
            "steps":   steps,
            "tiles":   max_tiles,
            "badge":   got_badge,
        })

        print(f"Ep {ep+1:3d} | reward={total_reward:+.1f} | steps={steps:5d} | tiles={max_tiles:4d} | badge={'YES' if got_badge else 'no'}")

    env.close()

    # % summary statistics
    badge_rate = sum(r["badge"] for r in results) / n_episodes
    avg_reward = np.mean([r["reward"] for r in results])
    std_reward = np.std([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    std_steps = np.std([r["steps"] for r in results])
    avg_tiles = np.mean([r["tiles"] for r in results])
    std_tiles = np.std([r["tiles"] for r in results])
    print(f"\nSummary over {n_episodes} episodes:")
    print(f"Badge obtained: {badge_rate*100:.1f}%")
    print(f"Average reward: {avg_reward:+.1f} ± {std_reward:.1f}")
    print(f"Average steps: {avg_steps:.1f} ± {std_steps:.1f}")
    print(f"Average tiles: {avg_tiles:.1f} ± {std_tiles:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default=f"{config.MODEL_DIR}/ppo_pokemon_final", help="Path to the saved model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--render",   action="store_true", help="Render the game window during evaluation")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.render)
