# RL-vs-LLM Comparison GIF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `assets/comparison.gif` (+ `assets/comparison.mp4`), a ~12–15 s side-by-side highlight montage: left panel the RL agent (PPO `agent_087`), right panel the LLM agent (qwen3-vl:8b) with its real (shortened) reasoning text captioned below.

**Architecture:** Two thin recorder paths dump per-step PNG frames + metadata JSONL (RL: new `agents/rl/record_run.py` reusing `build_vec_env`; LLM: a `frame` argument added to the existing `on_step` callback + `--record-dir` in `agents/llm/run.py`). A new composer `agents/make_comparison_gif.py` reads both frame dirs, resamples paired highlight segments to a common length, composes labeled side-by-side frames with a caption strip (PIL), and exports GIF (imageio) + MP4 (system ffmpeg).

**Tech Stack:** Python 3.12+ (`.venv`), PyBoy 2.7.1, stable-baselines3 (PPO), Pillow, imageio, numpy, system `ffmpeg` (verified at `/usr/bin/ffmpeg`), pytest.

## Global Constraints

- Chat with the user in Italian; ALL repo code/docs/comments/commit messages in English. (Project rule.)
- The env layer (`env/`) MUST NOT change.
- LLM thoughts in the caption are REAL trace extracts, only shortened/truncated — never invented.
- GIF file size < 5 MB (LinkedIn GIF limit is 8 MB; MP4 is the primary upload).
- Canvas dimensions must be even numbers (ffmpeg yuv420p requirement).
- Footage dirs live under `runs/comparison_footage/` (untracked); final assets under `assets/` (tracked).
- ROM at `pokemon_rom.gbc`; both recordings start from `saves/violet_city_gym.state`.
- RL checkpoint: `runs/checkpoints/agent_087/agent_087_final.zip`.
- Run everything with the venv python: `.venv/bin/python`; tests: `.venv/bin/python -m pytest <path> -v` from repo root.
- LLM recording requires `ollama serve` with `qwen3-vl:8b` pulled.
- `pyboy.screen.ndarray` is RGBA `(144, 160, 4)` — drop alpha (`[:, :, :3]`) before saving/composing.

---

## File Structure

```
agents/rl/record_run.py          # NEW: RL frames + meta.jsonl recorder (reuses build_vec_env)
agents/llm/agent.py              # MODIFY: pass the captured frame to on_step
agents/llm/run.py                # MODIFY: on_step accepts frame; --record-dir saves PNGs
agents/make_comparison_gif.py    # NEW: montage composer (helpers + CLI)
assets/comparison_segments.json  # NEW: curated highlight segment indices + captions
assets/comparison.gif|.mp4       # NEW: final deliverables
tests/llm/test_record_hook.py    # NEW: frame reaches on_step (stub client + emulator)
tests/test_comparison_gif.py     # NEW: unit tests for composer helpers
```

---

### Task 1: LLM loop — pass the frame to `on_step` + `--record-dir`

**Files:**
- Modify: `agents/llm/agent.py` (the `on_step` call at ~line 161)
- Modify: `agents/llm/run.py` (callback signature + new CLI flag)
- Test: `tests/llm/test_record_hook.py`

**Interfaces:**
- Consumes: `ReActAgent.run(wrapper, reader, on_step=None)`; inside the loop a frame is already captured at ~line 90 as `frame = wrapper.pyboy.screen.ndarray.copy()` (RGBA).
- Produces: `on_step(step: int, state: dict, out: dict, obs: dict, frame: np.ndarray)` — 5th positional arg, the RGBA frame for that step. `run.py --record-dir DIR` writes `DIR/frame_{step:05d}.png` (RGB) per step.

- [ ] **Step 1: Write the failing test**

`tests/llm/test_record_hook.py` (the `emulator` fixture already exists in `tests/llm/conftest.py` and skips cleanly without the ROM):

```python
from agents.llm.config import LLMConfig
from agents.llm.agent import ReActAgent


class StubClient:
    """Offline stand-in for OllamaClient: always presses `a`."""
    def chat(self, messages, tools):
        return {"thought": "stub", "tool_name": "press", "args": {"button": "a"}, "tokens": 0}


def test_on_step_receives_frame(emulator):
    wrapper, reader = emulator
    cfg = LLMConfig()
    cfg.max_steps = 2
    agent = ReActAgent(cfg)
    agent.client = StubClient()  # no network

    frames = []

    def on_step(step, state, out, obs, frame):
        frames.append(frame)

    agent.run(wrapper, reader, on_step=on_step)
    assert len(frames) == 2
    assert frames[0].shape[0] == 144 and frames[0].shape[1] == 160  # GB screen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/llm/test_record_hook.py -v`
Expected: FAIL with `TypeError: on_step() missing 1 required positional argument: 'frame'` (the loop calls `on_step(step, state, out, obs)`).

- [ ] **Step 3: Implement — pass the frame through**

In `agents/llm/agent.py`, the loop already captures `frame = wrapper.pyboy.screen.ndarray.copy()` before probing. Change the callback invocation (~line 161) from:

```python
            if on_step:
                on_step(step, state, out, obs)
```

to:

```python
            if on_step:
                on_step(step, state, out, obs, frame)
```

In `agents/llm/run.py`, update the callback and add the flag. Replace the argparse block and `on_step` with:

```python
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="open PyBoy window + print reasonings")
    ap.add_argument("--record-dir", default=None,
                    help="also save one PNG per step (frame_00000.png ...) for montage footage")
    args = ap.parse_args()
```

and (inside `main`, replacing the current `on_step`; add `import imageio.v3 as iio` at the top):

```python
        if args.record_dir:
            os.makedirs(args.record_dir, exist_ok=True)

        def on_step(step, state, out, obs, frame):
            if args.watch:
                thought = out["thought"] or "(no reasoning text)"
                print(f"[{step}] {thought}\n -> {out['tool_name']}({out['args']}) | "
                      f"{obs['note']} map {state['map_bank']}-{state['map_number']} "
                      f"@({state['local_x']},{state['local_y']}) "
                      f"battle={state['battle_type']} tokens={out['tokens']}")
            if args.record_dir:
                iio.imwrite(os.path.join(args.record_dir, f"frame_{step:05d}.png"),
                            frame[:, :, :3])
            f.write(json.dumps({
                "step": step, "map": [state["map_bank"], state["map_number"]],
                "pos": [state["local_x"], state["local_y"]], "battle": state["battle_type"],
                "thought": out["thought"][:300], "tool": out["tool_name"], "args": out["args"],
                "obs": obs["note"], "tokens": out["tokens"],
            }) + "\n")
```

NOTE: `argparse` must run BEFORE `path`/log setup only if it doesn't already; keep the existing order otherwise. Do not change anything else in the file.

- [ ] **Step 4: Run tests to verify they pass (full LLM suite for regressions)**

Run: `.venv/bin/python -m pytest tests/llm -v`
Expected: all pass (24 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add agents/llm/agent.py agents/llm/run.py tests/llm/test_record_hook.py
git commit -m "feat(llm): expose per-step frame to on_step and add --record-dir footage dump"
```

---

### Task 2: RL footage recorder

**Files:**
- Create: `agents/rl/record_run.py`

**Interfaces:**
- Consumes: `build_vec_env(state_path, gif_dir, watch)` from `agents/rl/evaluate_cnn.py` (returns `(vec, env)`; `env.pyboy.pyboy.screen.ndarray` is the full-res screen; `env.ram_reader.read_all()` returns the state dict with `map_bank`, `map_number`, `local_x`, `local_y`, `battle_type`, `zephyr`).
- Produces: `OUT_DIR/frame_{step:05d}.png` (RGB, one per env-step) + `OUT_DIR/meta.jsonl` (one line per step: `{"step", "map", "pos", "battle", "zephyr"}`) — the same field names the LLM trace uses, so the composer can treat both uniformly.

This is an integration script (same precedent as `agents/rl/make_gif.py`: no unit test; verified by a smoke run).

- [ ] **Step 1: Write record_run.py**

```python
"""Record per-step frames + RAM metadata of a trained CNN policy for montage footage.

Unlike make_gif.py (which renders a final GIF directly), this dumps raw material:
one PNG per env-step plus a meta.jsonl (map, position, battle flag, badge bit) so a
composer can pick highlight segments by step index afterwards.

Usage:
  .venv/bin/python -m agents.rl.record_run \
      --model runs/checkpoints/agent_087/agent_087_final.zip \
      --state saves/violet_city_gym.state \
      --out runs/comparison_footage/rl --max-steps 2000
"""

import argparse
import json
import os

import imageio.v3 as iio
import torch
from stable_baselines3 import PPO

from agents.rl.evaluate_cnn import build_vec_env


def record(model_path, state_path, out_dir, max_steps, deterministic):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec, env = build_vec_env(state_path, gif_dir=None, watch=False)
    model = PPO.load(model_path, device=device)

    obs = vec.reset()
    steps = 0
    with open(os.path.join(out_dir, "meta.jsonl"), "w", buffering=1) as f:
        for step in range(max_steps):
            iio.imwrite(os.path.join(out_dir, f"frame_{step:05d}.png"),
                        env.pyboy.pyboy.screen.ndarray[:, :, :3].copy())
            s = env.ram_reader.read_all()
            f.write(json.dumps({
                "step": step, "map": [s["map_bank"], s["map_number"]],
                "pos": [s["local_x"], s["local_y"]], "battle": s["battle_type"],
                "zephyr": s["zephyr"],
            }) + "\n")
            steps = step + 1
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, dones, _ = vec.step(action)
            if dones[0]:
                break
    vec.close()
    print(f"[record] wrote {steps} frames + meta.jsonl to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--state", default="saves/violet_city_gym.state")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    record(a.model, a.state, a.out, a.max_steps, a.deterministic)
```

- [ ] **Step 2: Smoke-run 30 steps**

Run:
```bash
.venv/bin/python -m agents.rl.record_run \
    --model runs/checkpoints/agent_087/agent_087_final.zip \
    --state saves/violet_city_gym.state \
    --out runs/comparison_footage/smoke --max-steps 30
```
Expected: prints `[record] wrote 30 frames + meta.jsonl ...`; `ls runs/comparison_footage/smoke | head` shows `frame_00000.png ...` and `meta.jsonl` has 30 lines. Then `rm -rf runs/comparison_footage/smoke`.

- [ ] **Step 3: Commit**

```bash
git add agents/rl/record_run.py
git commit -m "feat(rl): raw footage recorder (per-step frames + RAM metadata)"
```

---

### Task 3: Composer helpers — text + frame composition

**Files:**
- Create: `agents/make_comparison_gif.py`
- Test: `tests/test_comparison_gif.py`

**Interfaces:**
- Produces (consumed by Task 4):
  - `shorten_thought(text: str, max_len: int = 120) -> str` — collapse whitespace, cut at a word boundary, append `…` when cut.
  - `wrap_caption(text: str, width: int = 56, max_lines: int = 2) -> list[str]`.
  - `upscale2x(frame: np.ndarray) -> np.ndarray` — nearest-neighbor 2x (crisp pixel art).
  - `compose_frame(rl_frame, llm_frame, caption_lines: list[str]) -> np.ndarray` — one montage canvas frame, RGB uint8, `(400, 688, 3)`.
  - Layout constants: `PANEL_W=320, PANEL_H=288, MARGIN=16, HEADER_H=40, CAPTION_H=72, CANVAS_W=688, CANVAS_H=400`; labels `RL_LABEL = "RL - PPO (agent_087)"`, `LLM_LABEL = "LLM - qwen3-vl:8b (ReAct)"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_comparison_gif.py`:

```python
import numpy as np
from agents.make_comparison_gif import (
    shorten_thought, wrap_caption, upscale2x, compose_frame,
    CANVAS_W, CANVAS_H,
)


def test_shorten_thought_collapses_and_truncates():
    txt = "I  should\n move   up because " + "x" * 200
    out = shorten_thought(txt, max_len=40)
    assert len(out) <= 41  # 40 + ellipsis char
    assert "\n" not in out and "  " not in out
    assert out.endswith("…")


def test_shorten_thought_short_text_untouched():
    assert shorten_thought("Press A to advance.", max_len=120) == "Press A to advance."


def test_wrap_caption_max_two_lines():
    lines = wrap_caption("word " * 50, width=20, max_lines=2)
    assert len(lines) == 2
    assert all(len(l) <= 20 for l in lines)


def test_upscale2x_doubles_dims():
    f = np.arange(144 * 160 * 3, dtype=np.uint8).reshape(144, 160, 3)
    up = upscale2x(f)
    assert up.shape == (288, 320, 3)
    assert (up[0, 0] == f[0, 0]).all() and (up[1, 1] == f[0, 0]).all()  # nearest


def test_compose_frame_shape_and_even_dims():
    rl = np.zeros((144, 160, 3), dtype=np.uint8)
    llm = np.full((144, 160, 3), 255, dtype=np.uint8)
    canvas = compose_frame(rl, llm, ["thinking about the door"])
    assert canvas.shape == (CANVAS_H, CANVAS_W, 3)
    assert CANVAS_W % 2 == 0 and CANVAS_H % 2 == 0  # ffmpeg yuv420p
    assert canvas.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_comparison_gif.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.make_comparison_gif'`.

- [ ] **Step 3: Implement the helpers**

`agents/make_comparison_gif.py`:

```python
"""Compose the RL-vs-LLM side-by-side highlight montage (GIF + MP4).

Reads two footage dirs (agents/rl/record_run.py and agents/llm/run.py --record-dir),
a segments JSON (curated step ranges + real-thought captions), and renders:
left panel RL, right panel LLM with its reasoning captioned underneath.

Usage:
  .venv/bin/python -m agents.make_comparison_gif \
      --segments assets/comparison_segments.json \
      --out assets/comparison
"""

import argparse
import json
import os
import re
import subprocess
import textwrap

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Layout (all even — ffmpeg yuv420p needs even dims)
PANEL_W, PANEL_H = 320, 288          # 2x the 160x144 GB screen
MARGIN, HEADER_H, CAPTION_H = 16, 40, 72
CANVAS_W = MARGIN + PANEL_W + MARGIN + PANEL_W + MARGIN   # 688
CANVAS_H = HEADER_H + PANEL_H + CAPTION_H                 # 400

RL_LABEL = "RL - PPO (agent_087)"
LLM_LABEL = "LLM - qwen3-vl:8b (ReAct)"

_FONT_CANDIDATES = [
    "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.otf",
    "/usr/share/fonts/google-noto-vf/NotoSansMono[wght].ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]


def _font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def shorten_thought(text: str, max_len: int = 120) -> str:
    """Collapse whitespace and cut at a word boundary. Real text in, shorter real text out."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= max_len:
        return clean
    cut = clean[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def wrap_caption(text: str, width: int = 56, max_lines: int = 2) -> list:
    return textwrap.wrap(text, width=width)[:max_lines]


def upscale2x(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def compose_frame(rl_frame: np.ndarray, llm_frame: np.ndarray,
                  caption_lines: list) -> np.ndarray:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)
    label_font, caption_font = _font(18), _font(15)

    lx, rx = MARGIN, MARGIN + PANEL_W + MARGIN
    draw.text((lx, 10), RL_LABEL, fill=(120, 200, 255), font=label_font)
    draw.text((rx, 10), LLM_LABEL, fill=(255, 190, 120), font=label_font)

    canvas.paste(Image.fromarray(upscale2x(rl_frame)), (lx, HEADER_H))
    canvas.paste(Image.fromarray(upscale2x(llm_frame)), (rx, HEADER_H))

    y = HEADER_H + PANEL_H + 10
    for line in caption_lines:
        draw.text((rx, y), line, fill=(230, 230, 230), font=caption_font)
        y += 22
    return np.asarray(canvas, dtype=np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_comparison_gif.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/make_comparison_gif.py tests/test_comparison_gif.py
git commit -m "feat: comparison-GIF composer helpers (labels, caption, 2x panels)"
```

---

### Task 4: Montage builder + CLI (GIF + MP4 export)

**Files:**
- Modify: `agents/make_comparison_gif.py`
- Test: `tests/test_comparison_gif.py` (add tests)

**Interfaces:**
- Consumes: Task 3 helpers; footage dirs with `frame_{step:05d}.png`; segments JSON.
- Produces:
  - `load_frames(dir_path: str, start: int, end: int) -> list[np.ndarray]` — reads `frame_{i:05d}.png` for `i in [start, end)`, skipping missing indices.
  - `resample(frames: list, target_len: int) -> list` — `np.linspace` index resampling so both sides of a segment share a duration.
  - `build_montage(segments: list[dict], rl_dir: str, llm_dir: str, fps: int) -> list[np.ndarray]`.
  - CLI `main()` reading `--segments`, `--out` (basename without extension) → writes `<out>.gif` + `<out>.mp4` (via system ffmpeg).
- Segments JSON schema (one object per montage segment, played in order):

```json
{
  "fps": 10,
  "segments": [
    {"seconds": 6, "rl": [0, 300], "llm": [5, 40],
     "caption": "real shortened thought text shown under the LLM panel"}
  ]
}
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_comparison_gif.py`:

```python
import imageio.v3 as iio
from agents.make_comparison_gif import load_frames, resample, build_montage


def _write_frames(dir_path, n):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = np.full((144, 160, 3), i * 10 % 255, dtype=np.uint8)
        iio.imwrite(str(dir_path / f"frame_{i:05d}.png"), frame)


def test_load_frames_range_and_missing_ok(tmp_path):
    _write_frames(tmp_path / "f", 5)
    frames = load_frames(str(tmp_path / "f"), 1, 8)  # 8 > available: skip missing
    assert len(frames) == 4
    assert frames[0].shape == (144, 160, 3)


def test_resample_stretches_and_shrinks():
    frames = [np.full((1, 1, 3), i, dtype=np.uint8) for i in range(4)]
    assert len(resample(frames, 8)) == 8
    shrunk = resample(frames, 2)
    assert len(shrunk) == 2
    assert shrunk[0][0, 0, 0] == 0 and shrunk[-1][0, 0, 0] == 3  # keeps endpoints


def test_build_montage_lengths_and_shape(tmp_path):
    _write_frames(tmp_path / "rl", 6)
    _write_frames(tmp_path / "llm", 3)
    segments = [{"seconds": 1, "rl": [0, 6], "llm": [0, 3], "caption": "hello"}]
    frames = build_montage(segments, str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
    assert len(frames) == 4  # 1 s * 4 fps
    from agents.make_comparison_gif import CANVAS_H, CANVAS_W
    assert frames[0].shape == (CANVAS_H, CANVAS_W, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_comparison_gif.py -v`
Expected: FAIL with `ImportError` for `load_frames`/`resample`/`build_montage`.

- [ ] **Step 3: Implement builder + CLI**

Append to `agents/make_comparison_gif.py`:

```python
def load_frames(dir_path: str, start: int, end: int) -> list:
    frames = []
    for i in range(start, end):
        path = os.path.join(dir_path, f"frame_{i:05d}.png")
        if os.path.exists(path):
            frames.append(iio.imread(path)[:, :, :3])
    return frames


def resample(frames: list, target_len: int) -> list:
    idx = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return [frames[i] for i in idx]


def build_montage(segments: list, rl_dir: str, llm_dir: str, fps: int) -> list:
    out = []
    for seg in segments:
        n = int(round(seg["seconds"] * fps))
        rl = resample(load_frames(rl_dir, *seg["rl"]), n)
        llm = resample(load_frames(llm_dir, *seg["llm"]), n)
        caption = wrap_caption(shorten_thought(seg.get("caption", "")))
        out.extend(compose_frame(r, l, caption) for r, l in zip(rl, llm))
    return out


def export(frames: list, out_base: str, fps: int) -> None:
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    gif_path, mp4_path = out_base + ".gif", out_base + ".mp4"
    iio.imwrite(gif_path, np.stack(frames), plugin="pillow", loop=0,
                duration=int(round(1000 / fps)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", gif_path,
         "-movflags", "+faststart", "-pix_fmt", "yuv420p", mp4_path],
        check=True,
    )
    for p in (gif_path, mp4_path):
        print(f"[montage] wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="segments JSON (fps, segments[])")
    ap.add_argument("--rl-dir", default="runs/comparison_footage/rl")
    ap.add_argument("--llm-dir", default="runs/comparison_footage/llm")
    ap.add_argument("--out", default="assets/comparison", help="output basename (no extension)")
    args = ap.parse_args()
    with open(args.segments) as f:
        spec = json.load(f)
    frames = build_montage(spec["segments"], args.rl_dir, args.llm_dir, spec.get("fps", 10))
    export(frames, args.out, spec.get("fps", 10))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test file**

Run: `.venv/bin/python -m pytest tests/test_comparison_gif.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/make_comparison_gif.py tests/test_comparison_gif.py
git commit -m "feat: montage builder + GIF/MP4 export CLI for the RL-vs-LLM comparison"
```

---

### Task 5: Record real footage, curate segments, render the deliverables

This is the curation/integration task — commands + acceptance criteria, no unit tests.

**Files:**
- Create: `assets/comparison_segments.json`
- Create (artifacts): `assets/comparison.gif`, `assets/comparison.mp4`

**Interfaces:**
- Consumes: everything above. Requires `ollama serve` + `qwen3-vl:8b` running for the LLM recording.

- [ ] **Step 1: Record RL footage (~2 min wall clock)**

```bash
.venv/bin/python -m agents.rl.record_run \
    --model runs/checkpoints/agent_087/agent_087_final.zip \
    --state saves/violet_city_gym.state \
    --out runs/comparison_footage/rl --max-steps 2000
```
Expected: ~840–1000 frames; last meta.jsonl lines show `"zephyr": true` (badge won).

- [ ] **Step 2: Record LLM footage (~30–40 min at ~6 s/step — run and wait)**

```bash
.venv/bin/python -m agents.llm.run --record-dir runs/comparison_footage/llm
```
Expected: frames + a JSONL trace in `runs/llm_logs/` whose steps align with the frame indices. The interesting arc (first battle won → door bouncing) happens within the first ~200 steps.

- [ ] **Step 3: Curate the 4 highlights into 2 paired segments**

Find the step ranges from the metadata (battle flag transitions and position stalls):

```bash
# RL: battle windows and badge step
.venv/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("runs/comparison_footage/rl/meta.jsonl")]
for i in range(1, len(rows)):
    if rows[i]["battle"] != rows[i-1]["battle"]:
        print("RL battle edge at step", rows[i]["step"], "->", rows[i]["battle"])
print("badge at:", next((r["step"] for r in rows if r["zephyr"]), None))
EOF

# LLM: battle edges + where the position freezes (door bouncing) + thought texts
.venv/bin/python - <<'EOF'
import glob, json
trace = sorted(glob.glob("runs/llm_logs/run_*.jsonl"))[-1]
rows = [json.loads(l) for l in open(trace) if "summary" not in l]
for i in range(1, len(rows)):
    if rows[i]["battle"] != rows[i-1]["battle"]:
        print("LLM battle edge at step", rows[i]["step"], "->", rows[i]["battle"])
for r in rows:
    if r["thought"]:
        print(r["step"], "|", r["thought"][:110])
EOF
```

Write `assets/comparison_segments.json` — pairing per the approved design (segment 1: RL navigating ↔ LLM winning its battle; segment 2: RL beating Falkner/badge ↔ LLM bouncing at the door). Captions MUST be real thought extracts from the trace (shortened only). Template to fill with the found indices:

```json
{
  "fps": 10,
  "segments": [
    {"seconds": 7, "rl": [RL_NAV_START, RL_NAV_END], "llm": [LLM_BATTLE_START, LLM_BATTLE_END],
     "caption": "<real thought during the battle, e.g. about choosing an attack>"},
    {"seconds": 7, "rl": [RL_FALKNER_START, RL_BADGE_STEP + 20], "llm": [LLM_STUCK_START, LLM_STUCK_END],
     "caption": "<real thought while stuck at the door>"}
  ]
}
```

- [ ] **Step 4: Render + verify**

```bash
.venv/bin/python -m agents.make_comparison_gif \
    --segments assets/comparison_segments.json --out assets/comparison
xdg-open assets/comparison.gif
```
Acceptance: total clip ~12–15 s; both labels readable; caption legible at ~50% zoom (LinkedIn feed size); badge moment visible on the left while the LLM bounces on the right; `assets/comparison.gif` < 5 MB (if larger: lower fps to 8, shorten segments, or add `frame-skip` by widening the resample). `assets/comparison.mp4` plays in a browser.

- [ ] **Step 5: Commit the deliverables**

```bash
git add assets/comparison_segments.json assets/comparison.gif assets/comparison.mp4
git commit -m "feat: RL-vs-LLM comparison montage (GIF + MP4)"
```

---

### Task 6: Phase sync (project rule)

**Files:**
- Modify: `README.md` (repo)
- Modify: `/home/fabio/Projects/FAANG-Job-Search/FAANG-Job-Search/Planning/Active-Roadmap.md`
- Modify: `/home/fabio/Projects/FAANG-Job-Search/FAANG-Job-Search/Planning/Weekly-Logs/2026-Q2/Week-27-Jun29-Jul5.md` (current week)

- [ ] **Step 1: README — embed the GIF**

Add near the top of `README.md` (right after the title/intro paragraph):

```markdown
![RL vs LLM — same gym, same goal](assets/comparison.gif)
```

- [ ] **Step 2: FAANG sync**

In `Active-Roadmap.md` Week 11, check off the plots/GIF item as partially done:
change `- [ ] Generate comparison plots + best GIFs from both agents *(→ carry to W12)*` to
`- [x] Comparison montage GIF+MP4 (assets/comparison.gif) ✅ 2026-07-0X — plots/map GIFs → W12`.
Add one line to the current weekly log under the Pokémon section noting the montage was produced.

- [ ] **Step 3: Commit (repo only; the vault is not a git repo — verify before attempting)**

```bash
git add README.md
git commit -m "docs: embed RL-vs-LLM comparison montage in README"
```

---

## Self-Review

**Spec coverage (Phase A):** RL footage (Task 2 + 5.1), LLM footage via on_step frame + --record-dir (Task 1 + 5.2), highlight selection by step index with config (Task 5.3, `assets/comparison_segments.json`), composition with labels + shortened real thoughts (Tasks 3–4), 12–15 s / 10–12 fps (segments JSON), GIF < 5 MB + MP4 primary (Task 5.4 acceptance), README/roadmap sync (Task 6). No gaps found.

**Placeholder scan:** Task 5.3 JSON contains UPPERCASE index placeholders by design — they are curation inputs discovered at runtime from the traces, with the exact commands to find them provided. No other TBDs.

**Type consistency:** `on_step(step, state, out, obs, frame)` used in Tasks 1 and 5; `load_frames/resample/build_montage/compose_frame` signatures match between Tasks 3, 4 and tests; `frame_{step:05d}.png` naming shared by Tasks 1, 2, 4.

**Note for executor:** Task 5 requires `ollama serve` with `qwen3-vl:8b` and takes ~40 min of recording; everything else is fast and offline.
