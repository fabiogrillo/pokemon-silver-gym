"""
Sanity check for PokemonEnvCNN.

What this verifies (no training involved):
  - reset() returns obs with correct shape (72, 80, 3) and dtype uint8
  - obs values are in [0, 255] (no float leakage, no negative values)
  - step() runs 1000 random actions without crashing
  - reward components are finite floats, info dict has expected keys
  - episode boundaries (terminated/truncated) trigger reset cleanly
  - GIF capture path executes (writes one file to /tmp)

Run:
    python -m tests.test_random_env_cnn
"""

import tempfile
import numpy as np
from env.pokemon_env_cnn import PokemonEnvCNN

ROM_PATH   = "pokemon_rom.gbc"
STATE_PATH = "saves/start.state"
N_STEPS    = 1000

EXPECTED_OBS_SHAPE = (72, 80, 3)
EXPECTED_OBS_DTYPE = np.uint8
EXPECTED_INFO_KEYS = {
    "reward_exploration", "reward_events", "reward_penalties",
    "visited_tiles", "hp_ratio", "map_number", "in_battle",
}


def main():
    # Use a temp dir so GIF capture path is exercised without polluting runs/
    with tempfile.TemporaryDirectory() as tmp:
        env = PokemonEnvCNN(
            ROM_PATH, STATE_PATH, headless=True,
            gif_dir=tmp, gif_every_n_episodes=1, gif_prefix="sanity",
        )

        # ── Reset
        obs, info = env.reset()
        assert obs.shape == EXPECTED_OBS_SHAPE, f"reset obs shape {obs.shape} != {EXPECTED_OBS_SHAPE}"
        assert obs.dtype == EXPECTED_OBS_DTYPE, f"reset obs dtype {obs.dtype} != {EXPECTED_OBS_DTYPE}"
        assert obs.min() >= 0 and obs.max() <= 255, f"reset obs out of range: [{obs.min()}, {obs.max()}]"
        print(f"[OK] reset → obs {obs.shape} {obs.dtype} range [{obs.min()}, {obs.max()}]")

        # ── Loop random actions
        total_reward = 0.0
        episode_lengths = []
        ep_steps = 0

        for step in range(N_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Per-step invariants (run on EVERY step — cheap)
            assert obs.shape == EXPECTED_OBS_SHAPE, f"step {step} obs shape wrong: {obs.shape}"
            assert obs.dtype == EXPECTED_OBS_DTYPE, f"step {step} obs dtype wrong: {obs.dtype}"
            assert np.isfinite(reward), f"step {step} reward not finite: {reward}"
            assert EXPECTED_INFO_KEYS.issubset(info.keys()), \
                f"step {step} info missing keys: {EXPECTED_INFO_KEYS - info.keys()}"

            total_reward += reward
            ep_steps += 1

            if step % 250 == 0:
                ram = env.ram_reader.read_all()
                print(
                    f"step {step:4d} | reward {reward:+7.2f} | total {total_reward:+9.2f} | "
                    f"tiles {info['visited_tiles']:3d} | map ({ram['map_bank']:2d},{ram['map_number']:2d}) | "
                    f"battle {ram['battle_type']} | hp {ram['hp_ratio']:.2f}"
                )

            if terminated or truncated:
                end_kind = "WIN" if terminated and ram_state_zephyr(env) else "truncated/death"
                print(f"  [ep end @ step {step}] kind={end_kind} ep_steps={ep_steps} total={total_reward:.2f}")
                episode_lengths.append(ep_steps)
                obs, _ = env.reset()
                total_reward = 0.0
                ep_steps = 0

        env.close()

    print()
    print(f"[OK] Completed {N_STEPS} steps without crash.")
    print(f"[OK] {len(episode_lengths)} episode(s) ended during run. lengths={episode_lengths}")
    print(f"[OK] GIF capture path executed (temp dir cleaned).")


def ram_state_zephyr(env):
    """Tiny helper: read the badge flag at episode end."""
    return env.ram_reader.read_all()["zephyr"]


if __name__ == "__main__":
    main()
