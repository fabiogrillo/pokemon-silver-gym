import os
import pytest

from env.pokemon_env_cnn import PokemonEnvCNN
from env.rewards import CORRIDOR_LEGAL, CORRIDOR_WHITELIST

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"

requires_rom = pytest.mark.skipif(
    not (os.path.exists(ROM) and os.path.exists(STATE)), reason="ROM/state not available")


def test_corridor_legal_superset_of_whitelist():
    assert CORRIDOR_WHITELIST <= CORRIDOR_LEGAL
    assert (24, 5) in CORRIDOR_LEGAL   # Elm's lab
    assert (10, 7) in CORRIDOR_LEGAL   # gym
    assert (26, 11) in CORRIDOR_LEGAL  # Violet gatehouse
    assert (10, 10) in CORRIDOR_LEGAL  # Violet Pokemon Center (heal before the gym)
    assert (3, 70) not in CORRIDOR_LEGAL  # Dark Cave stays illegal


@requires_rom
def test_confinement_terminates_on_illegal_map(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True, confine_to_corridor=True)
    env.reset()
    # Simulate the RAM reporting an off-corridor map (Dark Cave) without walking there:
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 3, 70
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True
    env.pyboy.pyboy.stop(save=False)


@requires_rom
def test_no_confinement_by_default(monkeypatch):
    env = PokemonEnvCNN(ROM, STATE, headless=True)
    env.reset()
    real_read = env.ram_reader.read_all
    def fake_read():
        s = real_read()
        s["map_bank"], s["map_number"] = 3, 70
        return s
    monkeypatch.setattr(env.ram_reader, "read_all", fake_read)
    _, _, terminated, _, _ = env.step(0)
    assert terminated is False
    env.pyboy.pyboy.stop(save=False)
