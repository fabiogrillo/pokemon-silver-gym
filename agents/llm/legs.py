"""L4: harness-owned corridor leg checklist (LLM-3, docs/superpowers/specs/2026-07-06-final-attempt
-findings.md's L4). The HARNESS (not the model) tracks progress along the egg-delivered -> Falkner
corridor as an ordered list of "legs" (one per map), each with a single target tile to reach. Every
turn's prompt carries only the current leg's goal -- this is the fix for LLM-2's observed failure
mode (50/56 navigate_to calls perseverated on one mid-town coordinate; the model can adopt the tool
but not pick strategic targets).

Grid convention (matches agents/llm/pathfind.py and extract_collision.py): `walkable[y][x]`,
row-major, TRUE axes (x grows EAST, y grows SOUTH). Any live-RAM position fed into this module must
already be un-swapped by the caller -- env/ram_reader.py's local_x/local_y fields are swapped
relative to their names (see agents/rl/map_layout.ram_to_image_px's docstring); this module never
touches RAM directly, so that's the caller's responsibility.

Leg targets are derived from the committed collision grids (assets/collision/*.json), not
hand-typed:
  - 4 legs are plain border exits: the median walkable tile on the map's named edge (west/east/
    north/south), computed by `border_exits()` below.
  - 3 legs are documented overrides for targets that are NOT plain border exits: the Route 31 ->
    Violet City gatehouse door, the Violet City gym door, and Falkner's tile inside the gym (see
    _ROUTE_31_GATE_DOOR, _GYM_DOOR and `_top_center_walkable()` below). The plan that seeded this
    module expected only the latter two as overrides ("Route 31 -> west" was expected to be a
    plain border exit like the other 4), but border_exits() on the committed route_31.json grid
    returns an empty list for "west" -- the border there is a solid gatehouse wall except for its
    2-tile door, confirmed against pret's disassembly (see _ROUTE_31_GATE_DOOR below). Grid data
    wins over the plan's assumption; this discrepancy is what the override documents.

Scope note: the plan's corridor order also names a (26, 11) Violet Gatehouse leg, but no collision
grid is committed for it -- `extract_collision.py`'s MAPS list only covers the 7 maps below (the
gatehouse interior was out of scope for L1a). Since this module's whole point is "targets derived
from the committed collision grids", the gatehouse has no leg of its own here; LEGS instead ends
Route 31's leg at the gatehouse's door (see _ROUTE_31_GATE_DOOR), covering the 7 maps that do have
a grid, in corridor order.
"""

import glob
import json
import os
from dataclasses import dataclass

COLLISION_DIR = "assets/collision"

# side -> which axis is fixed on that border, and at which coordinate.
_SIDES = ("west", "east", "north", "south")


def border_exits(grid: dict, side: str) -> list[tuple[int, int]]:
    """Walkable tiles on one border of `grid` (a raw collision-grid dict: width/height/walkable),
    true (x, y). `side` is one of "west" (x == 0), "east" (x == width - 1), "north" (y == 0),
    "south" (y == height - 1)."""
    if side not in _SIDES:
        raise ValueError(f"unknown side {side!r}, expected one of {_SIDES}")
    width, height, walkable = grid["width"], grid["height"], grid["walkable"]
    if side == "west":
        return [(0, y) for y in range(height) if walkable[y][0]]
    if side == "east":
        x = width - 1
        return [(x, y) for y in range(height) if walkable[y][x]]
    if side == "north":
        return [(x, 0) for x in range(width) if walkable[0][x]]
    y = height - 1  # side == "south"
    return [(x, y) for x in range(width) if walkable[y][x]]


def _median_border_target(grid: dict, side: str) -> tuple[int, int]:
    """The middle walkable tile (by position along the border) on `grid`'s `side` -- a single,
    reproducible target picked from the grid itself rather than hand-typed."""
    exits = sorted(border_exits(grid, side))
    if not exits:
        raise ValueError(f"grid ({grid['bank']}, {grid['num']}) has no walkable tile on side {side!r}")
    return exits[len(exits) // 2]


def _top_center_walkable(grid: dict) -> tuple[int, int]:
    """Override for the Falkner leg: the top-center walkable tile of the gym grid. Falkner stands
    at the north end of VIOLET_GYM's interior, past the trainer gauntlet; the topmost walkable row
    (row 0 is the map's wall border) with its median x is that tile, computed straight from the
    committed grid -- no hand-typed coordinate."""
    walkable = grid["walkable"]
    for y, row in enumerate(walkable):
        xs = [x for x, v in enumerate(row) if v]
        if xs:
            return xs[len(xs) // 2], y
    raise ValueError(f"grid ({grid['bank']}, {grid['num']}) has no walkable tile at all")


# Override for the Violet City -> gym leg: the gym building's door sits mid-map, not on Violet
# City's border, so it can't come from `border_exits()`. Derived from pret's map source (the same
# technique extract_collision.py uses for the grids themselves -- Silver shares Gold's map data for
# this map, see extract_collision.py's module docstring):
#
#   https://raw.githubusercontent.com/pret/pokegold/master/maps/VioletCity.asm
#   def_warp_events
#   warp_event 18, 17, VIOLET_GYM, 1
#
# warp_event's args are (X, Y, destination map, destination warp id) in true (east, south) tile
# coordinates already -- this is disassembly source data, not a live-RAM read, so no un-swap is
# needed (contrast agents/rl/map_layout.ram_to_image_px, which documents the RAM-only swap).
# Confirmed walkable in the committed grid: assets/collision/violet_city.json["walkable"][17][18] == 1.
_GYM_DOOR: tuple[int, int] = (18, 17)

# Override for the Route 31 -> Violet City leg. The plan's leg order calls this a plain "west"
# border exit, and pret's map attributes do list `connection west, VioletCity` for Route 31 -- but
# probing border_exits() against the committed grid at import time raised
# "grid (26, 2) has no walkable tile on side 'west'": assets/collision/route_31.json's x in
# {0, 1, 2, 3} is a solid wall on every row. The `connection` macro is only a camera-scroll link;
# the real, walkable crossing is the 2-tile Violet Gatehouse door a few tiles in from the edge
# (pret's maps/Route31.asm):
#
#   def_warp_events
#   warp_event  4,  6, ROUTE_31_VIOLET_GATE, 3
#   warp_event  4,  7, ROUTE_31_VIOLET_GATE, 4
#
# ROUTE_31_VIOLET_GATE is the (26, 11) gatehouse interior; no collision grid is committed for it
# (see the module docstring's scope note), so this leg's target is the door INTO the gatehouse
# rather than a tile inside it -- same treatment as _GYM_DOOR above, one map short. Median of the
# two door tiles; confirmed walkable in the committed grid:
# assets/collision/route_31.json["walkable"][6][4] == 1 and ["walkable"][7][4] == 1.
_ROUTE_31_GATE_DOOR: tuple[int, int] = (4, 7)

# (map_key, display name, border side, per-turn hint) for the 4 plain border-exit legs, in
# corridor order (New Bark -> ... -> Route 30, the last map with a genuine walkable border
# crossing -- Route 31's crossing is the gatehouse-door override above).
_CORRIDOR = [
    ((24, 4), "New Bark Town", "west", "cross into Route 29"),
    ((24, 3), "Route 29", "west", "cross into Cherrygrove City"),
    ((26, 3), "Cherrygrove City", "north", "cross into Route 30"),
    ((26, 1), "Route 30", "north", "cross into Route 31"),
]


@dataclass(frozen=True)
class Leg:
    map_key: tuple[int, int]
    name: str
    target: tuple[int, int]
    hint: str


def _load_grids(collision_dir: str = COLLISION_DIR) -> dict:
    """{(bank, num): raw grid dict} for every committed assets/collision/*.json grid."""
    grids = {}
    for path in sorted(glob.glob(os.path.join(collision_dir, "*.json"))):
        with open(path) as f:
            g = json.load(f)
        grids[(g["bank"], g["num"])] = g
    return grids


def _build_legs() -> list[Leg]:
    grids = _load_grids()
    legs = [
        Leg(map_key, name, _median_border_target(grids[map_key], side), hint)
        for map_key, name, side, hint in _CORRIDOR
    ]
    legs.append(Leg((26, 2), "Route 31", _ROUTE_31_GATE_DOOR,
                     "enter the gatehouse toward Violet City"))
    legs.append(Leg((10, 5), "Violet City", _GYM_DOOR, "enter Falkner's Gym through the gym door"))
    legs.append(Leg((10, 7), "Falkner's Gym",
                     _top_center_walkable(grids[(10, 7)]),
                     "challenge Falkner for the Zephyr Badge"))
    return legs


LEGS: list[Leg] = _build_legs()


class LegTracker:
    """Tracks progress along `legs` (default: the module-level LEGS) as the harness reports the
    player's current map each turn. Stateless for lookups (`current`); `goal_note` remembers the
    last on-route leg so it can still name a leg to return to once the player wanders off-route."""

    def __init__(self, legs: list[Leg] = LEGS):
        self.legs = legs
        self._last_on_route: Leg | None = None

    def current(self, bank: int, num: int) -> Leg | None:
        """The first leg whose map matches (bank, num), or None if off-route."""
        for leg in self.legs:
            if leg.map_key == (bank, num):
                return leg
        return None

    def goal_note(self, bank: int, num: int, x: int, y: int) -> str:
        """One-line per-turn goal for the prompt: leg name, current position, target tile, and a
        navigate_to instruction. Off-route (map not on the corridor): tells the model to return to
        the corridor, naming the last leg it was on (if any)."""
        leg = self.current(bank, num)
        if leg is not None:
            self._last_on_route = leg
            tx, ty = leg.target
            return (
                f"Leg '{leg.name}': you are at ({x}, {y}); target ({tx}, {ty}) -- {leg.hint}. "
                f"Call navigate_to({tx}, {ty})."
            )
        if self._last_on_route is not None:
            ptx, pty = self._last_on_route.target
            return (
                f"Off-route (not on the corridor): return to the corridor -- previous leg "
                f"'{self._last_on_route.name}' at ({ptx}, {pty})."
            )
        return "Off-route (not on the corridor): return to the corridor."

    def completed_count(self, visited_map_ids) -> int:
        """How many legs, counted from the start of the corridor, have been visited so far.
        Counts consecutively from leg 0 and stops at the first not-yet-visited leg, matching the
        corridor's fixed order (a later map being visited without an earlier one is not "progress"
        -- e.g. it can't happen on this linear route without a skip/teleport)."""
        visited = set(visited_map_ids)
        count = 0
        for leg in self.legs:
            if leg.map_key not in visited:
                break
            count += 1
        return count
