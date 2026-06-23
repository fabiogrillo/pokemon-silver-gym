# Map assets for the trajectory overlay

`agents/rl/visualize_map.py` draws the agent's path on a background image of the New Bark → Violet
(Falkner) corridor.

- **`johto_corridor.png`** (optional): a stitched image of the corridor maps. If present, it is used
  as the background; otherwise a labelled **schematic** canvas is generated automatically, so the
  visualizer works without this file.

To add a real background:
1. Obtain a stitched Johto image of the corridor (community map resources, or compose per-map images
   from a wiki). Save it here as `johto_corridor.png`.
2. Calibrate the per-map pixel offsets in `agents/rl/map_layout.py` so `offset + local` lands on the
   right spot in the image — walk the corridor with `python tests/save_state.py <name>`, which prints
   `MAP CHANGED -> (bank,num) local=(x,y)` at every boundary, and align a few landmark tiles.
