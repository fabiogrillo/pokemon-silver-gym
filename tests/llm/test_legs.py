from agents.llm import legs

# Same GRIDS mapping as tests/llm/test_collision_grids.py -- the frozen set of committed grids.
_GRID_PATHS = {
    (24, 4): "assets/collision/new_bark.json", (24, 3): "assets/collision/route_29.json",
    (26, 3): "assets/collision/cherrygrove.json", (26, 1): "assets/collision/route_30.json",
    (26, 2): "assets/collision/route_31.json", (10, 5): "assets/collision/violet_city.json",
    (10, 7): "assets/collision/gym.json",
}

# side -> declared border side for each of the 4 plain-exit legs. The other 3 legs are documented
# overrides, not border exits: Route 31's crossing is gated by the Violet Gatehouse door (grid data
# shows an empty "west" border -- see legs._ROUTE_31_GATE_DOOR), Violet City's is the gym door, and
# the gym's is Falkner's tile.
_BORDER_LEG_SIDES = {
    (24, 4): "west", (24, 3): "west", (26, 3): "north", (26, 1): "north",
}


def _synthetic_grid():
    # 4x3 grid, TRUE axes, grid[y][x]:
    #   row0: 0 1 1 0
    #   row1: 1 1 0 1
    #   row2: 0 1 1 0
    return {
        "bank": 0, "num": 0, "width": 4, "height": 3,
        "walkable": [
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
    }


def test_border_exits_west():
    assert legs.border_exits(_synthetic_grid(), "west") == [(0, 1)]


def test_border_exits_east():
    assert legs.border_exits(_synthetic_grid(), "east") == [(3, 1)]


def test_border_exits_north():
    assert legs.border_exits(_synthetic_grid(), "north") == [(1, 0), (2, 0)]


def test_border_exits_south():
    assert legs.border_exits(_synthetic_grid(), "south") == [(1, 2), (2, 2)]


def test_border_exits_rejects_unknown_side():
    import pytest
    with pytest.raises(ValueError):
        legs.border_exits(_synthetic_grid(), "up")


def test_legs_cover_the_full_corridor_order_in_the_declared_map_order():
    # 4 grid-derived border legs + 3 documented overrides (gatehouse door, gym door, Falkner) =
    # 7 legs, one per committed collision grid (no grid is committed for the (26, 11) gatehouse
    # interior itself, see agents/llm/extract_collision.py's MAPS list -- legs.py's scope tracks
    # the committed grids).
    assert [leg.map_key for leg in legs.LEGS] == [
        (24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (10, 5), (10, 7),
    ]


def test_every_leg_target_is_walkable_in_its_own_grid():
    import json
    for leg in legs.LEGS:
        with open(_GRID_PATHS[leg.map_key]) as f:
            g = json.load(f)
        x, y = leg.target
        assert 0 <= x < g["width"] and 0 <= y < g["height"], leg
        assert g["walkable"][y][x] == 1, leg


def test_border_leg_targets_land_on_their_declared_side():
    import json
    for leg in legs.LEGS:
        side = _BORDER_LEG_SIDES.get(leg.map_key)
        if side is None:
            continue  # override leg (gym door / Falkner), not a border exit -- see below
        with open(_GRID_PATHS[leg.map_key]) as f:
            g = json.load(f)
        assert leg.target in legs.border_exits(g, side)


def test_route_31_border_west_is_empty_which_is_why_it_needs_an_override():
    import json
    with open("assets/collision/route_31.json") as f:
        g = json.load(f)
    assert legs.border_exits(g, "west") == []


def test_route_31_override_is_the_gatehouse_door_from_pret_warp_data():
    leg = legs.LegTracker().current(26, 2)
    assert leg.target == legs._ROUTE_31_GATE_DOOR


def test_violet_city_override_is_the_gym_door_from_pret_warp_data():
    leg = legs.LegTracker().current(10, 5)
    assert leg.target == legs._GYM_DOOR


def test_gym_override_is_the_top_center_of_the_topmost_walkable_row():
    import json
    with open("assets/collision/gym.json") as f:
        g = json.load(f)
    leg = legs.LegTracker().current(10, 7)
    tx, ty = leg.target
    # Falkner's tile must be on the topmost row that has any walkable tile at all.
    topmost_walkable_y = min(y for y, row in enumerate(g["walkable"]) if any(row))
    assert ty == topmost_walkable_y


def test_leg_tracker_current_transitions_across_the_leg_order():
    tracker = legs.LegTracker()
    for leg in legs.LEGS:
        bank, num = leg.map_key
        assert tracker.current(bank, num) is leg


def test_leg_tracker_current_is_none_off_route():
    tracker = legs.LegTracker()
    assert tracker.current(99, 99) is None


def test_goal_note_contains_target_coordinates():
    tracker = legs.LegTracker()
    leg = legs.LEGS[0]
    note = tracker.goal_note(leg.map_key[0], leg.map_key[1], 5, 5)
    tx, ty = leg.target
    assert str(tx) in note and str(ty) in note
    assert leg.name in note


def test_goal_note_off_route_mentions_the_previous_leg():
    tracker = legs.LegTracker()
    first_leg = legs.LEGS[0]
    tracker.goal_note(first_leg.map_key[0], first_leg.map_key[1], 1, 1)  # establish "previous"
    note = tracker.goal_note(99, 99, 0, 0)
    assert "return to the corridor" in note.lower()
    assert first_leg.name in note


def test_goal_note_off_route_with_no_prior_leg_still_says_return_to_corridor():
    tracker = legs.LegTracker()
    note = tracker.goal_note(99, 99, 0, 0)
    assert "return to the corridor" in note.lower()


def test_completed_count_counts_consecutive_visited_legs_from_the_start():
    tracker = legs.LegTracker()
    visited = {legs.LEGS[0].map_key, legs.LEGS[1].map_key, legs.LEGS[2].map_key}
    assert tracker.completed_count(visited) == 3


def test_completed_count_stops_at_first_gap():
    tracker = legs.LegTracker()
    visited = {legs.LEGS[0].map_key, legs.LEGS[2].map_key}  # skipped leg index 1
    assert tracker.completed_count(visited) == 1


def test_completed_count_zero_when_nothing_visited():
    tracker = legs.LegTracker()
    assert tracker.completed_count(set()) == 0
