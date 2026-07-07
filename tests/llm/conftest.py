import os
import pytest
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"
GYM_STATE = "saves/violet_city_gym.state"
LEDGE_STATE = "saves/crossing.state"


@pytest.fixture
def emulator():
    if not (os.path.exists(ROM) and os.path.exists(STATE)):
        pytest.skip("ROM or save state not available")
    wrapper = PyBoyWrapper(ROM, STATE, headless=True)
    reader = RAMReader(wrapper.pyboy)
    yield wrapper, reader
    wrapper.pyboy.stop(save=False)


@pytest.fixture
def gym_emulator():
    """Same as `emulator`, but boots saves/violet_city_gym.state (inside the gym map, bank 10 num
    7) -- used by navigate_to tests that need assets/collision/gym.json's coordinate space."""
    if not (os.path.exists(ROM) and os.path.exists(GYM_STATE)):
        pytest.skip("ROM or gym save state not available")
    wrapper = PyBoyWrapper(ROM, GYM_STATE, headless=True)
    reader = RAMReader(wrapper.pyboy)
    yield wrapper, reader
    wrapper.pyboy.stop(save=False)


@pytest.fixture
def ledge_emulator():
    """Same as `emulator`, but boots saves/crossing.state -- Route 30 (bank 26, num 1), true (x, y)
    = (6, 3), a few tiles from a real HOP_DOWN ledge at (5, 2). Used by the directional-ledge
    regression tests to exercise a real one-way hop against the actual emulator, not just the
    static grid."""
    if not (os.path.exists(ROM) and os.path.exists(LEDGE_STATE)):
        pytest.skip("ROM or ledge save state not available")
    wrapper = PyBoyWrapper(ROM, LEDGE_STATE, headless=True)
    reader = RAMReader(wrapper.pyboy)
    yield wrapper, reader
    wrapper.pyboy.stop(save=False)