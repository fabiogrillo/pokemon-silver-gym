from agents.llm.config import LLMConfig
from agents.llm.tools import execute_tool

CFG = LLMConfig()


def test_move_changes_position_or_map(emulator):
    wrapper, reader = emulator
    before = reader.read_all()
    obs = execute_tool("move", {"direction": "down", "steps": 5}, wrapper, reader, CFG)
    after = reader.read_all()
    assert obs["ok"] is True
    moved = (before["local_x"], before["local_y"]) != (after["local_x"], after["local_y"])
    map_changed = (before["map_number"] != after["map_number"])
    assert moved or map_changed or obs["stopped_early"]


def test_get_state_returns_text(emulator):
    wrapper, reader = emulator
    obs = execute_tool("get_state", {}, wrapper, reader, CFG)
    assert obs["ok"] is True
    assert "Map" in obs["note"]


def test_press_runs(emulator):
    wrapper, reader = emulator
    obs = execute_tool("press", {"button": "a"}, wrapper, reader, CFG)
    assert obs["ok"] is True