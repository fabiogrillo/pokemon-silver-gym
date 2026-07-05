# Map assets

- `johto_full.png` — community-stitched full Johto overworld map (Pokémon Gold/Silver),
  16 px per in-game tile (GBC native). Used ONLY as the background for the trajectory
  visualizations (`agents/rl/visualize_map.py`); no game assets are redistributed beyond
  this fan-made reference image. If you are the author and want attribution or removal,
  open an issue.
- `corridor_agent080.png` — generated overlay (trajectory + heatmap) produced by
  `visualize_map.py` for `runs/checkpoints/agent_080/agent_080_final.zip` (the RL v2 corridor
  checkpoint, warm-started from agent_079's breakthrough; see `training_log.md`), replayed from
  `saves/egg_delivered_clean.state`. Safe to regenerate at any time.
