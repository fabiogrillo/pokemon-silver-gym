"""Tests for the probe-verified greedy fallback (agents/llm/tools.py, _execute_navigate docstring
point 5): pathfind.plan() returning None from the LIVE position used to be treated as terminal
("no path"), which is what produced the 957x and 334x "no path" stalls -- a roaming NPC sitting
on/adjacent to the player blocks the static grid's only known route, even though the live emulator
would happily report an open direction if actually asked.

Two layers, per the TDD approach agreed for this task:
1. `_greedy_direction` is a pure function (open_dirs + current + goal -> direction) -- exercised
   exhaustively here with no emulator needed.
2. One real-emulator test forces `pathfind.plan()` to fail exactly once and confirms
   `_execute_navigate` takes a real, probe-verified step instead of bailing, then finishes the leg
   normally once planning resumes -- proving the new code path is actually wired in and does not
   regress the happy path.
"""
from agents.llm import pathfind
from agents.llm import tools as tools_mod
from agents.llm.config import LLMConfig
from agents.llm.tools import _greedy_direction, _true_xy, execute_tool

CFG = LLMConfig()


# ---------------------------------------------------------------------------
# 1. Pure _greedy_direction tests (no emulator)
# ---------------------------------------------------------------------------

def test_greedy_direction_empty_open_dirs_returns_none():
    assert _greedy_direction([], (0, 0), (5, 5)) is None


def test_greedy_direction_single_open_direction_is_taken_even_if_forced():
    # Goal is due east/south of current; the only open direction ("left") doesn't reduce distance
    # at all, but the fallback must still make forward progress with whatever is open rather than
    # returning nothing.
    assert _greedy_direction(["left"], (0, 0), (5, 5)) == "left"


def test_greedy_direction_picks_the_clear_distance_winner():
    # goal is due east (dx=5, dy=0): "right" strictly reduces Manhattan distance (4 vs. current 5),
    # "up" strictly increases it (6) since dy=0 already -- not a tie, "right" must win outright.
    assert _greedy_direction(["up", "right"], (0, 0), (5, 0)) == "right"


def test_greedy_direction_tie_prefers_larger_remaining_delta_x_axis():
    # goal (5, 2): stepping "right" or "down" each reduce total Manhattan distance from 7 to 6
    # (a tie) -- the remaining x delta (5) is larger than the remaining y delta (2), so "right"
    # (the x-axis move) must win the tie-break.
    assert _greedy_direction(["down", "right"], (0, 0), (5, 2)) == "right"


def test_greedy_direction_tie_prefers_larger_remaining_delta_y_axis():
    # Mirror of the above with the axes swapped: goal (2, 5) -- "down" (y-axis) must win.
    assert _greedy_direction(["down", "right"], (0, 0), (2, 5)) == "down"


def test_greedy_direction_full_tie_falls_back_to_canonical_direction_order():
    # goal (3, 3): "down" and "right" tie on both distance AND remaining delta (dx == dy == 3).
    # The deterministic fallback is DIRECTIONS order ("up", "down", "left", "right"), so "down"
    # (which precedes "right") must win.
    assert _greedy_direction(["right", "down"], (0, 0), (3, 3)) == "down"


def test_greedy_direction_handles_goal_to_the_west_and_north():
    # current (5, 5), goal (0, 0): "up" and "left" tie on both distance and remaining delta
    # (|dx| == |dy| == 5); canonical order picks "up" (precedes "left").
    assert _greedy_direction(["up", "down", "left", "right"], (5, 5), (0, 0)) == "up"


def test_greedy_direction_ignores_directions_not_in_open_dirs():
    # "right" would win on distance alone, but it isn't reported open -- must not be picked.
    assert _greedy_direction(["left"], (0, 0), (5, 0)) == "left"


# ---------------------------------------------------------------------------
# 2. Emulator regression: navigate_to survives a forced plan() failure
# ---------------------------------------------------------------------------

def test_navigate_to_recovers_via_greedy_fallback_when_plan_fails_once(emulator, monkeypatch):
    # saves/egg_delivered_clean.state spawns in New Bark Town (bank 24, num 4) at TRUE (6, 6);
    # (8, 6) is grid-walkable and live-emulator-verified clear (see
    # test_navigate_to_east_changes_x_not_y). Force pathfind.plan() to return None on its FIRST
    # call only -- simulating the live "no path from here" failure this task fixes (e.g. a roaming
    # NPC sitting on the only route the static grid knows about) -- and confirm the executor takes
    # one real, probe-verified step toward the goal instead of reporting "no path" immediately,
    # then completes the leg normally once planning resumes on the next iteration.
    wrapper, reader = emulator
    before = reader.read_all()
    true_x, true_y = _true_xy(before)
    goal = (true_x + 2, true_y)

    grid = pathfind.load_grids()[(before["map_bank"], before["map_number"])]
    assert grid.is_walkable(*goal), "test setup: goal must be grid-walkable"

    real_plan = pathfind.plan
    calls = {"n": 0}

    def fake_plan(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # forced failure: exercises the greedy-fallback branch exactly once
        return real_plan(*a, **kw)

    monkeypatch.setattr(tools_mod.pathfind, "plan", fake_plan)

    obs = execute_tool("navigate_to", {"x": goal[0], "y": goal[1]}, wrapper, reader, CFG)

    after_true = _true_xy(reader.read_all())
    assert obs["ok"] is True
    assert after_true == goal
    assert calls["n"] > 1  # plan() really was retried (and succeeded) after the forced failure
