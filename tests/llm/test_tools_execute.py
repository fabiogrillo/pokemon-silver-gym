from agents.llm import pathfind
from agents.llm.config import LLMConfig
from agents.llm.tools import _true_xy, execute_tool

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


def test_navigate_to_east_changes_x_not_y(emulator):
    # Regression test for an axis mix-up between pathfind.py's A*-step->button conversion and the
    # executor's TRUE (x, y) re-reads: EAST must change x only. saves/egg_delivered_clean.state
    # spawns in New Bark Town (bank 24, num 4) at TRUE (6, 6); (8, 6) is grid-walkable and
    # live-emulator-verified clear (no NPC in the way, unlike two tiles south -- see the south
    # test below).
    wrapper, reader = emulator
    before = reader.read_all()
    true_x, true_y = _true_xy(before)
    goal = (true_x + 2, true_y)

    grid = pathfind.load_grids()[(before["map_bank"], before["map_number"])]
    assert grid.is_walkable(*goal), "test setup: goal must be grid-walkable"

    obs = execute_tool("navigate_to", {"x": goal[0], "y": goal[1]}, wrapper, reader, CFG)

    after_true = _true_xy(reader.read_all())
    assert obs["ok"] is True
    assert obs["stopped_early"] is False
    assert after_true == goal
    assert after_true[0] == true_x + 2  # x moved east
    assert after_true[1] == true_y      # y unchanged


def test_navigate_to_south_changes_y_not_x(emulator):
    # Regression test (south leg) for the same axis mix-up: SOUTH must change y only.
    #
    # Two tiles due south of the raw spawn tile, (6, 8), is grid-walkable but is where a live NPC
    # in this exact save state stands (confirmed via probe_walkable and OAM sprite inspection --
    # this is what actually caused the reported "path blocked at (6, 7)" bug, not an axis error).
    # A static collision grid has no notion of that NPC, so testing directly off the spawn tile
    # would be flaky. Sidestep it: first move a few tiles east (test above proves east is exact
    # and NPC-free), then verify the south leg from there.
    wrapper, reader = emulator
    spawn = reader.read_all()
    spawn_x, spawn_y = _true_xy(spawn)

    setup = execute_tool("navigate_to", {"x": spawn_x + 4, "y": spawn_y}, wrapper, reader, CFG)
    assert setup["ok"] is True and setup["stopped_early"] is False

    before = reader.read_all()
    true_x, true_y = _true_xy(before)
    goal = (true_x, true_y + 2)

    grid = pathfind.load_grids()[(before["map_bank"], before["map_number"])]
    assert grid.is_walkable(*goal), "test setup: goal must be grid-walkable"

    obs = execute_tool("navigate_to", {"x": goal[0], "y": goal[1]}, wrapper, reader, CFG)

    after_true = _true_xy(reader.read_all())
    assert obs["ok"] is True
    assert obs["stopped_early"] is False
    assert after_true == goal
    assert after_true[0] == true_x        # x unchanged
    assert after_true[1] == true_y + 2    # y moved south