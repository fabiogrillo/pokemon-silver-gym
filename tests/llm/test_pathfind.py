from agents.llm import pathfind

_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _grid(rows):
    return pathfind.Grid(bank=0, num=0, width=len(rows[0]), height=len(rows), walkable=rows)


def _replay(grid, start, path):
    x, y = start
    for step in path:
        dx, dy = _DELTA[step]
        x, y = x + dx, y + dy
        assert grid.is_walkable(x, y), f"path steps onto a non-walkable tile at ({x}, {y})"
    return x, y


def test_astar_straight_line():
    grid = _grid([[1] * 5 for _ in range(5)])
    path = pathfind.astar(grid, (0, 0), (4, 0))
    assert path == ["right"] * 4


def test_astar_same_cell_returns_empty_path():
    grid = _grid([[1]])
    assert pathfind.astar(grid, (0, 0), (0, 0)) == []


def test_astar_wall_detour_longer_than_manhattan_distance():
    # Only gap in the row-2 wall is at x=4; raw Manhattan distance (0,0)->(0,4) is 4, but the
    # detour via x=4 forces 12 steps (4 right + 4 down + 4 left).
    grid = _grid([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    path = pathfind.astar(grid, (0, 0), (0, 4))
    assert path is not None
    assert _replay(grid, (0, 0), path) == (0, 4)
    assert len(path) == 12


def test_astar_unreachable_returns_none():
    grid = _grid([
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ])
    assert pathfind.astar(grid, (0, 0), (0, 2)) is None


def test_astar_rejects_wall_start_or_goal():
    grid = _grid([
        [1, 0],
        [1, 1],
    ])
    assert pathfind.astar(grid, (1, 0), (0, 0)) is None  # start is a wall
    assert pathfind.astar(grid, (0, 0), (1, 0)) is None  # goal is a wall


def test_astar_rejects_out_of_bounds_goal():
    grid = _grid([[1, 1], [1, 1]])
    assert pathfind.astar(grid, (0, 0), (5, 5)) is None


def test_load_grids_finds_real_assets():
    grids = pathfind.load_grids()
    assert (10, 7) in grids  # assets/collision/gym.json
    gym = grids[(10, 7)]
    assert gym.width == 10 and gym.height == 16
    assert isinstance(gym.walkable, list) and len(gym.walkable) == 16


def test_plan_same_map_uses_real_gym_grid():
    grids = pathfind.load_grids()
    # gym.json row index 1 is all walkable (see assets/collision/gym.json) -> straight line.
    path = pathfind.plan(10, 7, (1, 1), (5, 1), grids=grids)
    assert path == ["right"] * 4


def test_plan_unknown_map_returns_none():
    assert pathfind.plan(999, 999, (0, 0), (1, 1)) is None


# ---------------------------------------------------------------------------------------------
# Directional ledge hops (agents/llm/pathfind.py's astar() neighbor expansion + ledge_recovery_
# direction()). Landing is TWO tiles past the ledge tile (three from the foothold), not one --
# every real ledge in every extracted map has a non-walkable "buffer" tile immediately past it
# and open ground only two tiles past it (see extract_collision.py's module docstring; verified
# exhaustively, 77/77 ledges, zero exceptions).

def _ledge_grid(ledges):
    """5x5 grid, all floor except a wall column at x=2 except where a ledge punches through it,
    and a wall at x=3 for the "buffer" tile every real ledge has immediately past it."""
    rows = [[1] * 5 for _ in range(5)]
    for x in (2, 3):
        for y in range(5):
            rows[y][x] = 0
    for (x, y) in ledges:
        rows[y][x] = 0  # ledge tiles are always non-walkable in the matrix too
    return pathfind.Grid(bank=0, num=0, width=5, height=5, walkable=rows, ledges=ledges)


def _replay_with_hops(grid, start, path):
    """Like _replay(), but a direction that steps into a ledge tile jumps two tiles further (see
    astar()'s neighbor expansion) instead of the usual one."""
    x, y = start
    for step in path:
        dx, dy = _DELTA[step]
        adjacent = (x + dx, y + dy)
        if adjacent in grid.ledges:
            x, y = adjacent[0] + 2 * dx, adjacent[1] + 2 * dy
        else:
            x, y = adjacent
        assert grid.is_walkable(x, y), f"path steps onto a non-walkable tile at ({x}, {y})"
    return x, y


def test_astar_hops_ledge_in_allowed_direction():
    # Column x=2 is a wall except (2, 2), a HOP_RIGHT ledge; x=3 is the buffer wall (also 0 in the
    # matrix, per the real-game layout); landing (4, 2) is open floor two tiles past the ledge.
    grid = _ledge_grid({(2, 2): ["right"]})
    path = pathfind.astar(grid, (1, 2), (4, 2))
    assert path == ["right"]  # one button press covers foothold -> ledge -> buffer -> landing
    assert _replay_with_hops(grid, (1, 2), path) == (4, 2)


def test_astar_rejects_ledge_against_its_direction():
    # Same ledge, but approaching from the landing side moving left (into the ledge from the
    # "wrong" side) must not be treated as a valid hop -- COLL_HOP_RIGHT only permits FACE_RIGHT.
    grid = _ledge_grid({(2, 2): ["right"]})
    assert pathfind.astar(grid, (4, 2), (1, 2)) is None


def test_astar_ledge_hop_landing_must_be_walkable():
    # If the tile two past the ledge isn't walkable, the hop (and therefore the whole route) isn't
    # available.
    rows = [[1] * 5 for _ in range(5)]
    for x in (2, 3, 4):
        for y in range(5):
            rows[y][x] = 0
    ledges = {(2, 2): ["right"]}
    rows[2][2] = 0
    grid = pathfind.Grid(bank=0, num=0, width=5, height=5, walkable=rows, ledges=ledges)
    assert pathfind.astar(grid, (1, 2), (4, 2)) is None


def test_astar_diagonal_ledge_allows_either_listed_direction():
    # A corner ledge (e.g. COLL_HOP_DOWN_RIGHT) permits a hop from either named direction. Only the
    # ledge tile and its two buffer tiles (one per allowed direction) are walled off, so both a
    # horizontal foothold (1, 2) and a vertical foothold (2, 1) stay open.
    rows = [[1] * 5 for _ in range(5)]
    rows[2][2] = 0  # the ledge tile itself
    rows[2][3] = 0  # buffer for the "right" hop
    rows[3][2] = 0  # buffer for the "down" hop
    ledges = {(2, 2): ["down", "right"]}
    grid = pathfind.Grid(bank=0, num=0, width=5, height=5, walkable=rows, ledges=ledges)

    path_right = pathfind.astar(grid, (1, 2), (4, 2))
    assert path_right == ["right"]
    assert _replay_with_hops(grid, (1, 2), path_right) == (4, 2)
    path_down = pathfind.astar(grid, (2, 1), (2, 4))
    assert path_down == ["down"]
    assert _replay_with_hops(grid, (2, 1), path_down) == (2, 4)


def test_ledge_recovery_direction_on_ledge_tile():
    grid = _ledge_grid({(2, 2): ["right"]})
    assert pathfind.ledge_recovery_direction(grid, (2, 2)) == "right"


def test_ledge_recovery_direction_on_buffer_tile():
    grid = _ledge_grid({(2, 2): ["right"]})
    assert pathfind.ledge_recovery_direction(grid, (3, 2)) == "right"


def test_ledge_recovery_direction_none_on_walkable_tile():
    grid = _ledge_grid({(2, 2): ["right"]})
    assert pathfind.ledge_recovery_direction(grid, (1, 2)) is None
    assert pathfind.ledge_recovery_direction(grid, (4, 2)) is None


def test_plan_route29_no_path_regression():
    # Task regression target: plan((24, 3), (10, 11) -> (0, 7)) must return a real path. Note
    # (verify, don't assume): this exact coordinate pair already had a path BEFORE the directional-
    # ledge fix too (Route 29's walkable area was already a single connected component without any
    # hop edges -- checked exhaustively with a flood fill, with and without hop edges, across every
    # map that has ledges: none of them change connectivity). So the live trace's repeated
    # "no path to (0, 7)" was NOT actually caused by a sealed static grid; seeledge_recovery_
    # direction's docstring and agents/llm/tools._execute_navigate bug 3 for the real mechanism
    # this task ended up fixing (the live player landing mid-hop on a ledge/buffer tile, which
    # pathfind.plan() had no node for and would previously fail "no path" from forever). This test
    # still asserts the literal regression target the task specified.
    grids = pathfind.load_grids()
    path = pathfind.plan(24, 3, (10, 11), (0, 7), grids=grids)
    assert path is not None
    assert _replay(grids[(24, 3)], (10, 11), path) == (0, 7)
