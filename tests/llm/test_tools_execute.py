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


def test_navigate_to_border_tile_crosses_into_next_map(emulator):
    # Regression test for the "parked on the border" bug: New Bark Town's west edge has 4
    # grid-walkable tiles (assets/collision/new_bark.json col 0, rows 8/9/12/13 -- see
    # agents/llm/legs.border_exits((24, 4), "west")), but only rows 8-9 are an actual Route 29
    # connection; rows 12-13 are grid-walkable local terrain that's a dead end against the map
    # edge (live-emulator-verified: repeated "left" presses from (0, 12)/(0, 13) never move the
    # player or change the map). (0, 9) is both grid-walkable AND a real connection, so it's the
    # target here.
    #
    # GSC only triggers a map connection when the player steps PAST the edge -- one more "left"
    # press while standing on the border tile -- and A* can't plan an out-of-grid goal, so
    # `navigate_to` used to stop exactly at (0, 9) and never actually leave the map (confirmed by
    # running this exact call against the pre-fix executor: it reports "navigated to (0, 9)" and
    # stays on map (24, 4) forever). The fix (agents/llm/tools._finish_at_goal +
    # pathfind.border_exit_direction) presses the outward direction a few more times once the A*
    # target itself is a border tile.
    wrapper, reader = emulator
    before = reader.read_all()
    assert (before["map_bank"], before["map_number"]) == (24, 4)  # New Bark Town

    obs = execute_tool("navigate_to", {"x": 0, "y": 9}, wrapper, reader, CFG)

    after = reader.read_all()
    assert obs["ok"] is True
    assert (after["map_bank"], after["map_number"]) == (24, 3)  # Route 29 -- an actual transition
    assert obs["stopped_early"] is True


def test_navigate_to_hops_real_route_30_ledge(ledge_emulator):
    # Emulator regression for the directional-ledge fix (agents/llm/pathfind.py's astar() hop
    # edges + ledge_recovery_direction). saves/crossing.state starts on Route 30 (bank 26, num 1)
    # at true (6, 3); assets/collision/route_30.json has a real HOP_DOWN ledge at (5, 2), foothold
    # (5, 1), landing (5, 4) (two tiles past the ledge, past a non-walkable "buffer" tile at
    # (5, 3) -- see extract_collision.py's module docstring on why the real hop distance is three
    # tiles from the foothold, not two).
    #
    # A single navigate_to call must cross the ledge cleanly, matching how a real player would
    # play through it -- confirmed directly against the emulator during this task: before the
    # ledge_recovery_direction fix, cfg.frames_per_press (24 frames) sometimes wasn't enough to
    # finish the whole hop animation in one press, and the live player was observed parked exactly
    # on the ledge tile (5, 2) or its buffer tile (5, 3) -- both non-walkable in the static grid --
    # after which pathfind.plan()'s first-line is_walkable(start) check failed for literally any
    # goal, reproducing the "no path to (0, 7)" x334 symptom this task set out to fix.
    wrapper, reader = ledge_emulator
    before = reader.read_all()
    assert (before["map_bank"], before["map_number"]) == (26, 1)  # Route 30
    assert _true_xy(before) == (6, 3)

    grid = pathfind.load_grids()[(26, 1)]
    assert grid.ledges.get((5, 2)) == ["down"]
    assert grid.is_walkable(5, 4)

    obs = execute_tool("navigate_to", {"x": 5, "y": 4}, wrapper, reader, CFG)

    after_true = _true_xy(reader.read_all())
    assert obs["ok"] is True
    assert after_true == (5, 4)


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