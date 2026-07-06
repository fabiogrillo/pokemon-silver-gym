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
