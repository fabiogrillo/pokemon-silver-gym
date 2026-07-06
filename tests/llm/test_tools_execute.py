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


def test_navigate_to_walks_north_in_gym(gym_emulator):
    # saves/violet_city_gym.state starts inside the gym map (bank 10, num 7), true (x, y) = (5, 12).
    # Straight north (same x) is open for 2 tiles (assets/collision/gym.json row 9 col 5 is a wall,
    # confirmed against the live emulator: pressing "up" repeatedly from this exact save moves
    # local_x 12 -> 11 -> 10 and then stops) -- so the goal here is 2 tiles north, a target that is
    # both grid- and emulator-verified walkable and reachable in a straight line.
    wrapper, reader = gym_emulator
    before = reader.read_all()
    # RAM local_x/local_y are swapped relative to their names (agents/rl/map_layout.ram_to_image_px);
    # TRUE (x, y) is (local_y, local_x).
    true_x, true_y = before["local_y"], before["local_x"]
    goal = (true_x, true_y - 2)

    obs = execute_tool("navigate_to", {"x": goal[0], "y": goal[1]}, wrapper, reader, CFG)

    after = reader.read_all()
    after_true = (after["local_y"], after["local_x"])
    assert obs["ok"] is True
    assert after_true[1] < true_y  # moved toward the target (north = smaller true y)
    assert after_true == goal