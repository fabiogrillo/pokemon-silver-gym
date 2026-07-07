import os
import pytest

from env.pokemon_env_cnn import PokemonEnvCNN

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"

requires_rom = pytest.mark.skipif(
    not (os.path.exists(ROM) and os.path.exists(STATE)), reason="ROM/state not available")


@requires_rom
def test_visited_obs_off_by_default_keeps_obs_space_identical():
    """With the flag OFF the observation space must be BIT-IDENTICAL to today (old checkpoints
    must keep loading) — exact obs-key set equality, not just 'visited' absent."""
    env = PokemonEnvCNN(ROM, STATE, headless=True)
    obs, _ = env.reset()
    assert set(obs.keys()) == {"image", "vector"}
    assert set(env.observation_space.spaces.keys()) == {"image", "vector"}
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_visited_obs_on_adds_visited_key(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, visited_obs=True)
    assert set(env.observation_space.spaces.keys()) == {"image", "vector", "visited"}
    env.reset()
    obs, *_ = env.step(0)
    visited = obs["visited"]
    assert visited.dtype.kind == "u"  # uint8
    assert visited.shape == (48, 48)
    # The tile the player currently stands on is always episode-visited (added in step()/reset()),
    # so the crop's center (player position) must be marked.
    assert visited[24, 24] == 1
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_visited_crop_detransposes_east_walk(monkeypatch):
    """THE de-transposition regression test. env/ram_reader.py's local_x/local_y fields are swapped
    vs their names (local_x holds wYCoord, local_y holds wXCoord — see agents/rl/map_layout.py's
    ram_to_image_px docstring). Walking true-EAST means incrementing RAM's local_y field. The crop
    must place those tiles along increasing COLUMNS at a constant row — this test FAILS if the crop
    is indexed crop[x][y] (swapped) instead of crop[row=true_y][col=true_x]."""
    env = PokemonEnvCNN(ROM, STATE, headless=True, visited_obs=True)
    env.reset()
    real_read = env.ram_reader.read_all
    base = real_read()
    fixed_ram_local_x = base["local_x"]   # true_y — held constant (walking due east only)
    start_ram_local_y = base["local_y"]   # true_x — incremented to simulate the east walk

    def make_fake(offset):
        def fake_read():
            s = real_read()
            s["local_x"] = fixed_ram_local_x
            s["local_y"] = start_ram_local_y + offset
            return s
        return fake_read

    obs = None
    for offset in range(1, 4):
        monkeypatch.setattr(env.ram_reader, "read_all", make_fake(offset))
        obs, *_ = env.step(0)

    visited = obs["visited"]
    center_row = 24
    # 4 visited true-x positions (start + 3 east steps) land on columns 21..24, all on the SAME row.
    for col in range(21, 25):
        assert visited[center_row, col] == 1, f"expected east-walk mark at col {col}"
    # A swapped implementation would instead place the trail along the ROW axis at a fixed column —
    # assert that pattern is NOT what we see (rows 21..23 at the center column must stay unmarked,
    # since true_y never changed).
    for row in range(21, 24):
        assert visited[row, center_row] == 0, f"unexpected mark at row {row} (axes look swapped)"
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_dyn_budget_base_override_honored(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, dynamic_episode_budget=True, dyn_budget_base=32768)
    env.reset()
    assert env._max_steps == 32768
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 26, 3   # Cherrygrove = waypoint 1
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    env.step(0)
    assert env._max_steps == 65536   # 32768 * (1 + 1)
    env.pyboy.pyboy.stop(save=False)
