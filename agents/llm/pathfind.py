"""L1b: A* router over the walkability grids extracted by agents/llm/extract_collision.py.

Grid convention (matches assets/collision/*.json): `walkable[y][x]`, row-major, TRUE axes (x grows
EAST, y grows SOUTH) -- see extract_collision.py's module docstring. Directions returned here use
the same (dx, dy) convention as extract_collision.py's `_DIR_DELTA` and agents/llm/config.py's
SYSTEM_PROMPT ("moving up decreases y").

Scope (v1, YAGNI): `plan()` only routes within a single map. Cross-map legs (e.g. Route 31 ->
Violet City) are the harness's job later -- composing multiple same-map plans at map-boundary
warps is out of scope for this task.
"""

import glob
import heapq
import json
import os
from dataclasses import dataclass

COLLISION_DIR = "assets/collision"

# direction name -> (dx, dy) in TRUE (east, south) axes.
DIR_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass(frozen=True)
class Grid:
    bank: int
    num: int
    width: int
    height: int
    walkable: list  # row-major grid[y][x], truthy = walkable

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and bool(self.walkable[y][x])


def load_grids(collision_dir: str = COLLISION_DIR) -> dict:
    """Load every assets/collision/*.json grid into {(bank, num): Grid}."""
    grids = {}
    for path in sorted(glob.glob(os.path.join(collision_dir, "*.json"))):
        with open(path) as f:
            g = json.load(f)
        key = (g["bank"], g["num"])
        grids[key] = Grid(bank=g["bank"], num=g["num"], width=g["width"], height=g["height"],
                           walkable=g["walkable"])
    return grids


def astar(grid: Grid, start_xy: tuple, goal_xy: tuple, blocked: frozenset = frozenset()) -> list | None:
    """A* from start_xy to goal_xy (both TRUE (x, y)) over `grid`.

    `blocked` is an extra set of TRUE (x, y) tiles to treat as non-walkable on top of the static
    grid -- used by the executor (agents/llm/tools._execute_navigate) to route around a tile that
    the grid calls walkable but that a live NPC/sprite is actually standing on (the static
    collision grid has no notion of dynamic obstacles; see extract_collision.py's module
    docstring on scope).

    Returns a list of direction strings ("up"/"down"/"left"/"right") to press in order, an empty
    list if start == goal, or None if unreachable (including a wall/out-of-bounds start or goal).
    """
    if not grid.is_walkable(*start_xy) or not grid.is_walkable(*goal_xy) or goal_xy in blocked:
        return None
    if start_xy == goal_xy:
        return []

    def heuristic(xy):
        return abs(xy[0] - goal_xy[0]) + abs(xy[1] - goal_xy[1])

    open_heap = [(heuristic(start_xy), 0, start_xy)]
    came_from: dict = {}  # xy -> (prev_xy, direction)
    best_g = {start_xy: 0}
    closed = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_xy:
            path = []
            node = current
            while node in came_from:
                prev, direction = came_from[node]
                path.append(direction)
                node = prev
            path.reverse()
            return path
        closed.add(current)
        for direction, (dx, dy) in DIR_DELTA.items():
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in closed or nxt in blocked or not grid.is_walkable(*nxt):
                continue
            tentative = g + 1
            if tentative < best_g.get(nxt, float("inf")):
                best_g[nxt] = tentative
                came_from[nxt] = (current, direction)
                heapq.heappush(open_heap, (tentative + heuristic(nxt), tentative, nxt))
    return None


def border_exit_direction(grid: Grid, xy: tuple, last_direction: str | None = None) -> str | None:
    """Return the direction that walks OUT of `grid` from `xy`, if `xy` sits on a map border.

    GSC map connections only trigger when the player steps PAST the map edge -- one more
    directional press while standing on the border tile. `astar()`/`plan()` can't route into
    out-of-grid coordinates (there's no grid there), so a leg that targets a border tile as an
    exit stops exactly AT the edge and never actually crosses. The executor
    (agents/llm/tools._execute_navigate) calls this after arriving at a same-map A* target to
    decide whether that one extra outward press is needed.

    A tile can be on up to two borders at once (a corner). Tie-break: prefer the axis matching
    `last_direction` (the direction of the final A* step that reached `xy`), since that's the axis
    the leg was actually navigating along -- e.g. arriving at a corner by walking `down` means the
    leg cares about the south connection, not the perpendicular one. If `last_direction` doesn't
    resolve it (unknown, or doesn't match either border axis -- e.g. the caller was already
    standing on the tile with no A* step taken), fall back to a fixed, deterministic candidate
    order (left, right, up, down); this is an arbitrary but stable choice for the ambiguous case.

    Returns None if `xy` isn't on any border.
    """
    x, y = xy
    candidates = []
    if x == 0:
        candidates.append("left")
    if x == grid.width - 1:
        candidates.append("right")
    if y == 0:
        candidates.append("up")
    if y == grid.height - 1:
        candidates.append("down")
    if not candidates:
        return None
    if last_direction in candidates:
        return last_direction
    return candidates[0]


def plan(bank: int, num: int, start_true_xy: tuple, goal_true_xy: tuple,
         grids: dict | None = None, blocked: frozenset = frozenset()) -> list | None:
    """Same-map A* plan: (bank, num) select the grid, start/goal are TRUE (x, y).

    `grids` lets callers reuse an already-loaded `load_grids()` dict; defaults to a fresh load.
    `blocked` is forwarded to `astar()` (see its docstring) to route around dynamic obstacles.
    Returns None if the map isn't in the collision set or the target is unreachable.
    """
    grids = grids if grids is not None else load_grids()
    grid = grids.get((bank, num))
    if grid is None:
        return None
    return astar(grid, start_true_xy, goal_true_xy, blocked=blocked)
