"""Live NPC sprite reads (agents/llm/sprites.py) against the real emulator.

The gym save state is a good probe: the map has stationary trainers whose object structs are
loaded, and the RAM reader gives an independent read of the player's position to check the
+4 coordinate convention against.
"""
import pytest

from agents.llm import sprites
from agents.llm.pathfind import load_grids


def _player_true_xy(reader):
    s = reader.read_all()
    return s["local_y"], s["local_x"]  # un-swap: ram_reader's fields hold (wY, wX)


def test_gym_npcs_are_reported_on_walkable_grid_tiles(gym_emulator):
    wrapper, reader = gym_emulator
    tiles = sprites.live_npc_tiles(wrapper.pyboy)
    assert tiles, "the gym map has trainer sprites, none were read"
    grid = load_grids()[(10, 7)]
    for x, y in tiles:
        assert 0 <= x < grid.width and 0 <= y < grid.height


def test_player_tile_is_excluded(gym_emulator):
    wrapper, reader = gym_emulator
    assert _player_true_xy(reader) not in sprites.live_npc_tiles(wrapper.pyboy)


def test_player_struct_confirms_coordinate_convention(gym_emulator):
    """Slot 0 (the player) must match the RAM reader's position after the -4 border offset —
    the live check that MapX/MapY really are true axes + 4, with no x/y swap."""
    wrapper, reader = gym_emulator
    mem = wrapper.pyboy.memory
    base = sprites.OBJECT_STRUCTS
    struct_xy = (mem[base + 0x10] - 4, mem[base + 0x11] - 4)
    assert struct_xy == _player_true_xy(reader)


def test_empty_slots_are_skipped(gym_emulator):
    """Far fewer than 12 NPCs are loaded in the gym — empty slots must not add ghost tiles."""
    wrapper, reader = gym_emulator
    assert len(sprites.live_npc_tiles(wrapper.pyboy)) < 12
