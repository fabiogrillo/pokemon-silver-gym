"""
Map stitching layout for the Whidden-style trajectory/heatmap overlay (see visualize_map.py).

The game only exposes LOCAL coordinates per map (map_bank, map_number, local_x, local_y from
env/ram_reader.py). To plot a single world-space trajectory across maps we need a table that places
each local map inside one global canvas:

    global_tile = MAP_INFO[(bank, num)].offset + (local_x, local_y)
    global_px   = global_tile * TILE_PX

Scope: only the ~10 maps of the egg-delivered → Falkner corridor are needed (this is the whole
objective of the RL v2 re-baseline), so this is a small, hand-laid schematic — NOT a full-Johto
reconstruction. The offsets below are TOPOLOGICALLY faithful to Johto's geography (New Bark in the
south-east, the route running west then north to Violet City) but are PLACEHOLDER pixel positions.

Calibration (to make them pixel-accurate against a real stitched map image):
  1. Drop a stitched Johto-corridor image at assets/maps/johto_corridor.png.
  2. Walk the corridor with `python tests/save_state.py <name>` — it prints
     `MAP CHANGED -> (bank,num) local=(x,y)` at every boundary.
  3. For each map, note the local (x,y) of a recognisable landmark, read its pixel position in the
     image, and set offset so that offset + local lands on it. Only ~10 maps, so this is quick.

Until that image exists, visualize_map.py renders onto a generated schematic canvas (labelled
rectangles), so the tool is runnable immediately.
"""

from dataclasses import dataclass, field

# Pixels per in-game tile on the global canvas (schematic). Small to keep the canvas compact.
TILE_PX = 6


@dataclass(frozen=True)
class MapBox:
    name: str
    offset: tuple[int, int]   # (gx, gy) top-left of this map on the global canvas, in TILES
    size: tuple[int, int]     # (w, h) of this map, in TILES — generously sized for schematic placement
    color: tuple[int, int, int] = field(default=(60, 70, 90))  # schematic fill color


# (bank, number) -> MapBox. (bank, number) match the constants in env/rewards.py.
# Layout: north is up, west is left — mirrors the real Johto corridor. Interiors (Elm's lab, the gym,
# Mr. Pokemon's house, the gatehouse) are placed as small boxes next to their parent overworld map.
MAP_INFO: dict[tuple[int, int], MapBox] = {
    (10, 5):  MapBox("Violet City",      offset=(0,  18), size=(34, 30), color=(70, 100, 80)),
    (10, 7):  MapBox("Falkner's Gym",    offset=(8,  4),  size=(12, 12), color=(120, 80, 70)),
    (26, 11): MapBox("Violet Gatehouse", offset=(8,  50), size=(8,  6),  color=(90, 90, 70)),
    (26, 2):  MapBox("Route 31",         offset=(18, 58), size=(36, 18), color=(80, 95, 70)),
    (26, 1):  MapBox("Route 30 (gate)",  offset=(44, 74), size=(24, 34), color=(80, 95, 70)),
    (26, 10): MapBox("Mr. Pokemon's",    offset=(74, 78), size=(10, 10), color=(90, 90, 70)),
    (26, 3):  MapBox("Cherrygrove City", offset=(38, 110), size=(30, 24), color=(70, 100, 80)),
    (24, 3):  MapBox("Route 29",         offset=(74, 116), size=(40, 18), color=(80, 95, 70)),
    (24, 4):  MapBox("New Bark Town",    offset=(118, 112), size=(22, 22), color=(70, 100, 80)),
    (24, 5):  MapBox("Elm's Lab",        offset=(120, 138), size=(10, 10), color=(110, 90, 120)),
    (3, 70):  MapBox("Dark Cave",        offset=(60, 40), size=(10, 10), color=(50, 50, 60)),  # off-path
}

# Ordered corridor (for legends / progression labels), south-start to gym.
CORRIDOR_ORDER = [(24, 5), (24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (26, 11), (10, 5), (10, 7)]


def to_global_px(bank: int, num: int, local_x: int, local_y: int) -> tuple[int, int] | None:
    """Map a (bank, num, local_x, local_y) RAM reading to a global canvas pixel, or None if the map
    is not in the corridor layout (e.g. an interior we did not place — caller should skip it)."""
    box = MAP_INFO.get((bank, num))
    if box is None:
        return None
    gx = (box.offset[0] + local_x) * TILE_PX
    gy = (box.offset[1] + local_y) * TILE_PX
    return gx, gy


def canvas_size() -> tuple[int, int]:
    """(width, height) in pixels large enough to contain every placed map box."""
    max_w = max((b.offset[0] + b.size[0]) for b in MAP_INFO.values())
    max_h = max((b.offset[1] + b.size[1]) for b in MAP_INFO.values())
    pad = 4
    return (max_w + pad) * TILE_PX, (max_h + pad) * TILE_PX
