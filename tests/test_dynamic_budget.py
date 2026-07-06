import os
import pytest

from env.pokemon_env_cnn import PokemonEnvCNN

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"

requires_rom = pytest.mark.skipif(
    not (os.path.exists(ROM) and os.path.exists(STATE)), reason="ROM/state not available")


@requires_rom
def test_budget_starts_at_base_and_grows_on_waypoint(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, dynamic_episode_budget=True)
    env.reset()
    assert env._max_steps == 16384
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 26, 3   # Cherrygrove = waypoint 1
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    env.step(0)
    assert env._max_steps == 32768
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_budget_off_by_default(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True)
    env.reset()
    assert env._max_steps == 65536
    env.pyboy.pyboy.stop(save=False)
