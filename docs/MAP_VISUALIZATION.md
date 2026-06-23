# Map visualization

Replay a trained checkpoint and render **where the agent went**, overlaid on a map of the
New Bark → Violet City (Falkner) corridor. Inspired by PWhiddy's
[pokerl-map-viz](https://github.com/PWhiddy/pokerl-map-viz) and the coordinate-based exploration
of [PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments).

## How it works

The game only exposes **local** coordinates per map in RAM (`env/ram_reader.py`):

| field | RAM | meaning |
|-------|-----|---------|
| `map_bank`  | `0xDA00` | map group |
| `map_number`| `0xDA01` | map id within group |
| `local_x`   | `0xDA02` | x on the current map |
| `local_y`   | `0xDA03` | y on the current map |

To draw a single world-space path we convert each reading to a **global** canvas pixel:

```
global_tile = MAP_OFFSETS[(bank, num)] + (local_x, local_y)     # agents/rl/map_layout.py
global_px   = global_tile * TILE_PX
```

`agents/rl/visualize_map.py` rolls out one episode (reusing the eval wrapper stack from
`evaluate_cnn.build_vec_env`), reads the position each step, maps it to canvas pixels, and renders:

- **`<out>.png`** — visitation **heatmap** + the **trajectory** polyline (blue = early → red = late,
  green start dot, white end dot).
- **`<out>.gif`** — the trajectory revealed step by step.

## Usage

```bash
# one checkpoint
python -m agents.rl.visualize_map \
  --model runs/checkpoints/agent_076/agent_076_5000000_steps.zip \
  --state saves/egg_delivered_clean.state --max-steps 8000 --out runs/maps/agent_076

# progression montage: one PNG per checkpoint in a run dir + a combined GIF
python -m agents.rl.visualize_map --all-checkpoints runs/checkpoints/agent_076 \
  --state saves/egg_delivered_clean.state --out runs/maps/agent_076
```

## Background image (optional) and calibration

By default a labelled **schematic** canvas is generated from the map boxes in
`agents/rl/map_layout.py`, so the tool runs with no extra assets. The boxes are topologically faithful
to Johto (New Bark south-east; the route runs west then north to Violet) but are **placeholder** pixel
positions.

To use a real stitched map:

1. Save a stitched corridor image at `assets/maps/johto_corridor.png` (it is auto-detected and
   resized to the canvas).
2. Calibrate the offsets in `map_layout.py` so `offset + local` lands on the right spot:
   - Walk the corridor with `python tests/save_state.py <name>` — it prints
     `MAP CHANGED -> (bank,num) local=(x,y)` at every boundary.
   - For each of the ~10 corridor maps, align a recognisable landmark tile against the image and set
     its `offset`.

Only ~10 maps need offsets (the whole objective is the egg-delivered → Falkner corridor), so this is a
short, one-time calibration — not a full-Johto reconstruction.
