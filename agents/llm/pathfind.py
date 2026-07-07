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
from dataclasses import dataclass, field

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
    # {(x, y): [direction, ...]} -- TRUE-(x,y) ledge tiles and the facing(s) that may hop them (see
    # extract_collision.py's module docstring). A ledge tile is ALWAYS 0 in `walkable` (it's never
    # a legal place to stop); `astar()` uses this side-channel to route a hop straight through it
    # instead of treating it as a dead wall. Defaults to {} so pre-v2 grids/synthetic test grids
    # that never pass `ledges` behave exactly like before.
    ledges: dict = field(default_factory=dict)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and bool(self.walkable[y][x])


def _parse_ledges(raw: dict) -> dict:
    """{'y,x': ['down', ...]} (JSON, string keys) -> {(x, y): ['down', ...]} (TRUE (x, y) tuples)."""
    ledges = {}
    for key, dirs in raw.items():
        y_str, x_str = key.split(",")
        ledges[(int(x_str), int(y_str))] = list(dirs) if isinstance(dirs, list) else [dirs]
    return ledges


def load_grids(collision_dir: str = COLLISION_DIR) -> dict:
    """Load every assets/collision/*.json grid into {(bank, num): Grid}."""
    grids = {}
    for path in sorted(glob.glob(os.path.join(collision_dir, "*.json"))):
        with open(path) as f:
            g = json.load(f)
        key = (g["bank"], g["num"])
        grids[key] = Grid(bank=g["bank"], num=g["num"], width=g["width"], height=g["height"],
                           walkable=g["walkable"], ledges=_parse_ledges(g.get("ledges", {})))
    return grids


def astar(grid: Grid, start_xy: tuple, goal_xy: tuple, blocked: frozenset = frozenset()) -> list | None:
    """A* from start_xy to goal_xy (both TRUE (x, y)) over `grid`.

    `blocked` is an extra set of TRUE (x, y) tiles to treat as non-walkable on top of the static
    grid -- used by the executor (agents/llm/tools._execute_navigate) to route around a tile that
    the grid calls walkable but that a live NPC/sprite is actually standing on (the static
    collision grid has no notion of dynamic obstacles; see extract_collision.py's module
    docstring on scope).

    `grid.ledges` tiles are never entered as a plain step (they're always 0 in `walkable`). Instead,
    a ledge tile L with allowed direction(s) D is crossed as a single hop: moving in direction
    d in D from the tile immediately opposite L (i.e. `L - d`) lands three tiles away, at `L + 2*d`
    -- the real game always places a non-walkable "buffer" tile at `L + d` that the hop clears
    (see the neighbor expansion below for why) -- see extract_collision.py's module docstring.
    Entering L from any other side, or in any other direction, is not a valid move (matches the
    one-directional real-game hop).

    Returns a list of direction strings ("up"/"down"/"left"/"right") to press in order, an empty
    list if start == goal, or None if unreachable (including a wall/out-of-bounds start or goal).
    Each returned direction corresponds to one planned button press, but a ledge-hop direction can
    take the executor two real presses to fully resolve (one to step onto the ledge tile itself,
    one more to complete the jump over the buffer tile) -- see agents/llm/tools._execute_navigate's
    ledge-escape handling for how the executor stays correct either way.
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
            adjacent = (current[0] + dx, current[1] + dy)
            ledge_dirs = grid.ledges.get(adjacent)
            if ledge_dirs is not None:
                # `adjacent` is a ledge tile -- always non-walkable in the matrix, so it's never a
                # stop-over node. It's only traversable as a hop: entered from the foothold
                # OPPOSITE its allowed direction(s). The landing tile is TWO tiles past the ledge
                # (three tiles from the foothold), not one: every single ledge in every extracted
                # map (77/77, checked exhaustively) has a non-walkable "buffer" tile immediately
                # past it and open ground only two tiles past it -- i.e. every ledge block is built
                # as [foothold][ledge][buffer wall][landing]. This matches pokegold's engine
                # (engine/overworld/player_movement.asm .TryStep/.TryJump): the buffer tile is what
                # makes an ordinary step fail and fall through to the jump check in the first
                # place, and the jump animation (STEP_LEDGE) is sized to clear exactly that one
                # tile, landing on the far side of it -- the game never re-checks collision on the
                # buffer tile itself, which is why it's safe to skip over here without a
                # walkability check.
                if direction not in ledge_dirs or adjacent in blocked:
                    continue
                nxt = (adjacent[0] + 2 * dx, adjacent[1] + 2 * dy)
                if nxt in closed or nxt in blocked or not grid.is_walkable(*nxt):
                    continue
            else:
                nxt = adjacent
                if nxt in closed or nxt in blocked or not grid.is_walkable(*nxt):
                    continue
            tentative = g + 1
            if tentative < best_g.get(nxt, float("inf")):
                best_g[nxt] = tentative
                came_from[nxt] = (current, direction)
                heapq.heappush(open_heap, (tentative + heuristic(nxt), tentative, nxt))
    return None


def ledge_recovery_direction(grid: Grid, xy: tuple) -> str | None:
    """If `xy` is a real, live position the A* graph has no node for because it's mid-hop over a
    ledge, return the direction to press to keep going. None if `xy` is ordinary plannable ground.

    Two tiles of a ledge hop are 0 in `grid.walkable` and have no A* node (see astar()'s docstring):
    the ledge tile L itself, and the non-walkable "buffer" tile at `L + direction` that the hop
    clears. `cfg.frames_per_press` isn't guaranteed to cover a whole hop's animation in one press
    (same root cause as the walk-cycle overshoot documented in agents/llm/tools._execute_navigate),
    so the live player can end up standing on EITHER of those two tiles mid-hop -- confirmed
    directly against the emulator. Both recover the same way: press the ledge's own hop direction
    again (exactly what completes/re-triggers the jump in the real game too).

    Ordinary walkable tiles always return None here even if they happen to sit at the same offset
    as some ledge's buffer tile, because a buffer tile is by construction never walkable (see
    extract_collision.py's module docstring: 77/77 ledges checked, zero exceptions) -- so a
    walkable `xy` can never actually be one, and checking `is_walkable` first avoids ever
    second-guessing an ordinary in-progress route.
    """
    if grid.is_walkable(*xy):
        return None
    ledge_dirs = grid.ledges.get(xy)
    if ledge_dirs:
        return ledge_dirs[0]
    for direction, (dx, dy) in DIR_DELTA.items():
        candidate_ledge = (xy[0] - dx, xy[1] - dy)
        dirs = grid.ledges.get(candidate_ledge)
        if dirs and direction in dirs:
            return direction
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
