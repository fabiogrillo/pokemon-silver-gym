import json

from agents.llm import legs
from agents.rl.map_layout import MAP_INFO

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

_OUTWARD = {"west": (-1, 0), "east": (1, 0), "north": (0, -1), "south": (0, 1)}


def _load_grid(map_key):
    with open(_GRID_PATHS[map_key]) as f:
        return json.load(f)


def _counterpart_is_walkable(map_key, tile, side, next_map):
    """Independent re-derivation of the counterpart check (grids + MAP_INFO offsets only, no
    legs.py internals): whether stepping one tile outward off `tile` on `map_key`'s `side` lands
    on a walkable tile of `next_map`'s grid."""
    ngrid = _load_grid(next_map)
    off, noff = MAP_INFO[map_key].offset, MAP_INFO[next_map].offset
    dx, dy = _OUTWARD[side]
    nx = off[0] + tile[0] + dx - noff[0]
    ny = off[1] + tile[1] + dy - noff[1]
    in_bounds = 0 <= nx < ngrid["width"] and 0 <= ny < ngrid["height"]
    return in_bounds and ngrid["walkable"][ny][nx] == 1


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


def test_every_border_leg_target_has_a_walkable_counterpart_on_its_next_map():
    # The core property of the fix: a GSC border crossing only succeeds if the ARRIVAL tile on the
    # neighboring map is also walkable, so every plain border-exit leg's resolved target must have
    # a walkable counterpart on its (now explicit) next_map. Re-derived independently from the two
    # grids + MAP_INFO offsets (see _counterpart_is_walkable above), not via legs.py's own helper.
    for leg in legs.LEGS:
        side = _BORDER_LEG_SIDES.get(leg.map_key)
        if side is None:
            continue  # override leg (doors / Falkner), not a border-adjacency crossing
        assert leg.next_map is not None, leg
        assert _counterpart_is_walkable(leg.map_key, leg.target, side, leg.next_map), leg


def test_new_bark_west_target_has_a_walkable_route_29_counterpart():
    # Regression test for the "parked on the edge" bug: New Bark Town's west border tiles at
    # y in {12, 13} are locally walkable but their Route 29 counterparts are trees -- pressing
    # left there does nothing, forever. The old plain local median picked exactly (0, 12). The
    # resolved target must be one of the border tiles whose counterpart IS walkable (the property,
    # not a magic number -- live-verified for (0, 9) by
    # tests/llm/test_tools_execute.py::test_navigate_to_border_tile_crosses_into_next_map).
    leg = legs.LegTracker().current(24, 4)
    assert leg.next_map == (24, 3)  # Route 29, now explicit on the Leg
    assert _counterpart_is_walkable((24, 4), leg.target, "west", (24, 3)), leg.target
    # And specifically NOT one of the live-verified dead-end tiles.
    assert leg.target not in {(0, 12), (0, 13)}


def test_new_bark_dead_end_border_tiles_fail_the_counterpart_check():
    # Documents the bug's mechanism with the committed data: (0, 12)/(0, 13) are in
    # border_exits(new_bark, "west") (locally walkable) yet their Route 29 counterparts are not
    # walkable -- exactly the tiles the fix must exclude.
    g = _load_grid((24, 4))
    exits = legs.border_exits(g, "west")
    for dead_end in [(0, 12), (0, 13)]:
        assert dead_end in exits
        assert not _counterpart_is_walkable((24, 4), dead_end, "west", (24, 3))


def test_median_border_exit_falls_back_to_plain_median_without_a_neighbor_grid():
    # When next_map has no committed collision grid (the (26, 11) gatehouse interior case) there
    # is nothing to verify the counterpart against, so the resolver keeps the pre-fix behavior:
    # the plain median of the locally-walkable border tiles.
    grid = _synthetic_grid()
    target = legs._median_border_exit(grid, "north", (0, 0), (99, 99), grids={})
    assert target == (2, 0)  # median of [(1, 0), (2, 0)]


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


def test_goal_note_gatehouse_interior_returns_static_exit_message():
    # (26, 11) is the Violet Gatehouse interior -- no collision grid is committed for it (see the
    # module docstring's scope note), so it has no leg of its own and would otherwise fall through
    # to the generic off-route message. It IS on the route (between Route 31's leg and Violet
    # City's), so it needs its own static, hand-written exit instruction instead.
    tracker = legs.LegTracker()
    note = tracker.goal_note(26, 11, 5, 6)
    assert note == (
        "You are inside the Route 31 gatehouse: walk WEST (a few tiles) to exit into Violet City."
    )


def test_goal_note_gatehouse_interior_does_not_disturb_last_on_route_tracking():
    # Passing through the gatehouse should not clobber the "last on-route leg" memory used by the
    # generic off-route message -- Route 31 is still the last real leg the player was on.
    tracker = legs.LegTracker()
    route_31_leg = legs.LegTracker().current(26, 2)
    tracker.goal_note(26, 2, 4, 7)  # establish Route 31 as last-on-route
    tracker.goal_note(26, 11, 5, 6)  # pass through the gatehouse
    note = tracker.goal_note(99, 99, 0, 0)  # now truly off-route
    assert route_31_leg.name in note
