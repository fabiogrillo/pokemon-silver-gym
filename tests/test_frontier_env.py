"""
Integration smoke test for the Go-Explore frontier reset wired into PokemonEnvCNN (agent_059).
PyBoy-backed (loads the real ROM/states) — validates the code paths the unit tests can't:

  A. wrapper save_state_bytes() / reset_from_bytes() round-trip via the env's reset path
  B. reset() samples an archived cell when p_frontier=1 and the archive is non-empty, tags the
     episode (_from_frontier) so info["from_start"] flips to False (→ logs under front/)
  C. the harvest write-path: stepping in an egg-carrying / gym state populates the shared archive

Run:
    .venv/bin/python -m tests.test_frontier_env
"""

import tempfile
import numpy as np
from env.pokemon_env_cnn import PokemonEnvCNN, MAX_STEPS

ROM_PATH = "pokemon_rom.gbc"


def _check_obs(obs):
    assert isinstance(obs, dict) and set(obs) == {"image", "vector"}, "obs not Dict(image,vector)"
    assert obs["image"].shape == (72, 80, 3) and obs["image"].dtype == np.uint8
    assert obs["vector"].shape == (11,)


def test_reset_from_frontier_sets_flags_and_obs():
    with tempfile.TemporaryDirectory() as d:
        env = PokemonEnvCNN(ROM_PATH, "saves/start.state", headless=True, gif_dir=None,
                            frontier_root=d, p_frontier=1.0)
        try:
            # Empty archive → even at p_frontier=1 the reset must FALL BACK to start.state cleanly.
            obs, _ = env.reset()
            _check_obs(obs)
            assert env._from_frontier is False, "empty archive must fall back to start (not frontier)"

            # Seed one cell from the live emulator, then force a frontier reset.
            env.frontier.add("seed_cell", 100, env.pyboy.save_state_bytes)
            assert len(env.frontier) == 1
            obs, _ = env.reset()
            _check_obs(obs)
            assert env._from_frontier is True, "non-empty archive at p=1 must reset from frontier"

            # A frontier episode must report from_start=False so it logs under front/, not the nav/ gate.
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["from_start"] is False, "frontier episode must not count toward the start gate"
        finally:
            env.close()


def test_harvest_populates_archive_from_egg_state():
    # mid_route30.state holds the egg (egg_received) → the harvest condition fires → cells are written.
    with tempfile.TemporaryDirectory() as d:
        env = PokemonEnvCNN(ROM_PATH, "saves/mid_route30.state", headless=True, gif_dir=None,
                            frontier_root=d, p_frontier=0.0)
        try:
            env.reset()
            for _ in range(40):
                env.step(env.action_space.sample())
            assert len(env.frontier) > 0, "stepping in an egg-carrying state must harvest frontier cells"
            # The harvested state must be loadable and yield a well-formed obs (round-trip through disk).
            data = env.frontier.sample()
            assert data is not None
            screen = env.pyboy.reset_from_bytes(data)
            assert screen.shape[0] > 0
        finally:
            env.close()


def test_frontier_episode_uses_short_cap():
    # agent_062: a frontier episode truncates at frontier_max_steps; a start episode keeps MAX_STEPS.
    with tempfile.TemporaryDirectory() as d:
        env = PokemonEnvCNN(ROM_PATH, "saves/start.state", headless=True, gif_dir=None,
                            frontier_root=d, p_frontier=1.0, frontier_max_steps=20)
        try:
            env.reset()  # empty archive → falls back to start → FULL cap
            assert env._from_frontier is False and env._max_steps == MAX_STEPS
            env.frontier.add("seed", 2, env.pyboy.save_state_bytes)
            env.reset()  # archive non-empty, p=1 → frontier episode → SHORT cap
            assert env._from_frontier is True and env._max_steps == 20
            trunc = False
            for _ in range(20):
                _, _, _, trunc, _ = env.step(0)  # action 0 = 'up' (won't deliver from start)
                if trunc:
                    break
            assert trunc and env.steps == 20, f"frontier episode must truncate at the cap; steps={env.steps}"
        finally:
            env.close()


def test_egg_state_marker_in_obs():
    # agent_063: with egg_marker=True the obs image's top-left 8x8 corner encodes the egg quest state.
    # start.state = no egg → BLACK; mid_route30.state = egg received (carrying, undelivered) → RED.
    # (Default is egg_marker=False — the warm-marker run failed; teachers see a clean obs.)
    env = PokemonEnvCNN(ROM_PATH, "saves/start.state", headless=True, gif_dir=None, egg_marker=True)
    try:
        obs, _ = env.reset()
        assert (obs["image"][0:8, 0:8] == np.array([0, 0, 0])).all(), "no-egg corner must be black"
    finally:
        env.close()
    env2 = PokemonEnvCNN(ROM_PATH, "saves/mid_route30.state", headless=True, gif_dir=None, egg_marker=True)
    try:
        obs, _ = env2.reset()
        assert (obs["image"][0:8, 0:8] == np.array([255, 0, 0])).all(), "carrying corner must be red"
    finally:
        env2.close()
    # default OFF: the corner must be the raw game pixels, NOT forced black
    env3 = PokemonEnvCNN(ROM_PATH, "saves/mid_route30.state", headless=True, gif_dir=None)
    try:
        obs, _ = env3.reset()
        assert not (obs["image"][0:8, 0:8] == np.array([255, 0, 0])).all(), "marker must be OFF by default"
    finally:
        env3.close()


def main():
    for t in (test_reset_from_frontier_sets_flags_and_obs,
              test_harvest_populates_archive_from_egg_state,
              test_frontier_episode_uses_short_cap,
              test_egg_state_marker_in_obs):
        t()
        print(f"  ✓ {t.__name__}")
    print("ALL FRONTIER ENV INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
