import os
import pytest
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader

ROM = "pokemon_rom.gbc"
STATE = "saves/egg_delivered_clean.state"


@pytest.fixture
def emulator():
    if not (os.path.exists(ROM) and os.path.exists(STATE)):
        pytest.skip("ROM or save state not available")
    wrapper = PyBoyWrapper(ROM, STATE, headless=True)
    reader = RAMReader(wrapper.pyboy)
    yield wrapper, reader
    wrapper.pyboy.stop(save=False)