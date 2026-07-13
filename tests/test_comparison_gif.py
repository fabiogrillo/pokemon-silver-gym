import json

import numpy as np
import pytest
import imageio.v3 as iio
from agents.make_comparison_gif import (
    shorten_thought, wrap_caption, upscale2x, compose_frame,
    CANVAS_W, CANVAS_H, load_frames, load_trace_thoughts, thought_at, build_montage,
)


def test_shorten_thought_collapses_and_truncates():
    txt = "I  should\n move   up because " + "x" * 200
    out = shorten_thought(txt, max_len=40)
    assert len(out) <= 41  # 40 + ellipsis char
    assert "\n" not in out and "  " not in out
    assert out.endswith("…")


def test_shorten_thought_short_text_untouched():
    assert shorten_thought("Press A to advance.", max_len=120) == "Press A to advance."


def test_wrap_caption_max_lines():
    lines = wrap_caption("word " * 50, width=20, max_lines=2)
    assert len(lines) == 2
    assert all(len(l) <= 20 for l in lines)


def test_wrap_caption_marks_dropped_lines_with_ellipsis():
    lines = wrap_caption("word " * 50, width=20, max_lines=2)
    assert len(lines) == 2
    assert lines[-1].endswith("…")


def test_wrap_caption_no_ellipsis_when_nothing_dropped():
    lines = wrap_caption("short caption", width=34, max_lines=2)
    assert lines == ["short caption"]


def test_upscale2x_doubles_dims():
    f = np.arange(144 * 160 * 3, dtype=np.uint8).reshape(144, 160, 3)
    up = upscale2x(f)
    assert up.shape == (288, 320, 3)
    assert (up[0, 0] == f[0, 0]).all() and (up[1, 1] == f[0, 0]).all()  # nearest


def test_compose_frame_shape_and_even_dims():
    rl = np.zeros((144, 160, 3), dtype=np.uint8)
    llm = np.full((144, 160, 3), 255, dtype=np.uint8)
    canvas = compose_frame(rl, llm, ["thinking about the door"], title="Chapter 1")
    assert canvas.shape == (CANVAS_H, CANVAS_W, 3)
    assert CANVAS_W % 2 == 0 and CANVAS_H % 2 == 0  # ffmpeg yuv420p
    assert canvas.dtype == np.uint8


def _write_frames(dir_path, n):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        frame = np.full((144, 160, 3), i * 10 % 255, dtype=np.uint8)
        iio.imwrite(str(dir_path / f"frame_{i:05d}.png"), frame)


def _write_trace(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_frames_range_and_missing_ok(tmp_path):
    _write_frames(tmp_path / "f", 5)
    frames = load_frames(str(tmp_path / "f"), 1, 8)  # 8 > available: skip missing
    assert len(frames) == 4
    assert [i for i, _ in frames] == [1, 2, 3, 4]  # keeps source frame indices
    assert frames[0][1].shape == (144, 160, 3)


def test_load_trace_thoughts_skips_empty_and_summary(tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [
        {"step": 0, "thought": "first idea"},
        {"step": 3, "thought": ""},          # empty reasoning: no caption entry
        {"step": 5, "thought": "second idea"},
        {"summary": {"steps": 6}},           # summary line: not a step
    ])
    assert load_trace_thoughts(str(p)) == [(0, "first idea"), (5, "second idea")]


def test_thought_at_returns_latest_at_or_before():
    thoughts = [(0, "first"), (5, "second"), (9, "third")]
    assert thought_at(thoughts, 0) == "first"
    assert thought_at(thoughts, 4) == "first"
    assert thought_at(thoughts, 5) == "second"
    assert thought_at(thoughts, 100) == "third"
    assert thought_at([(3, "later")], 1) == ""  # nothing said yet


def test_build_montage_lengths_and_shape(tmp_path):
    _write_frames(tmp_path / "rl", 6)
    _write_frames(tmp_path / "llm", 3)
    segments = [{"seconds": 1, "rl": [0, 6], "llm": [0, 3], "caption": "hello"}]
    frames = build_montage(segments, str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
    assert len(frames) == 4  # 1 s * 4 fps
    assert frames[0].shape == (CANVAS_H, CANVAS_W, 3)


def test_build_montage_live_trace_captions_change_over_time(tmp_path):
    _write_frames(tmp_path / "rl", 8)
    _write_frames(tmp_path / "llm", 8)
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, [{"step": i, "thought": f"turn {i} reasoning"} for i in range(8)])
    segments = [{"seconds": 2, "rl": [0, 8], "llm": [0, 8],
                 "llm_trace": str(trace), "caption_seconds": 0.5}]
    with_trace = build_montage(segments, str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
    fixed = build_montage(
        [{"seconds": 2, "rl": [0, 8], "llm": [0, 8], "caption": "static"}],
        str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
    assert len(with_trace) == len(fixed) == 8
    # The caption band (below the panels) must change as the reasoning advances...
    band = slice(CANVAS_H - 90, CANVAS_H)
    assert any((with_trace[0][band] != f[band]).any() for f in with_trace[1:])
    # ...whereas a fixed caption renders the band identically on every frame.
    assert all((fixed[0][band] == f[band]).all() for f in fixed[1:])


def test_build_montage_per_segment_footage_dirs(tmp_path):
    _write_frames(tmp_path / "rl_a", 3)
    _write_frames(tmp_path / "llm_a", 3)
    _write_frames(tmp_path / "rl_b", 3)
    _write_frames(tmp_path / "llm_b", 3)
    segments = [
        {"seconds": 1, "rl": [0, 3], "llm": [0, 3], "caption": "a"},
        {"seconds": 1, "rl": [0, 3], "llm": [0, 3], "caption": "b",
         "rl_dir": str(tmp_path / "rl_b"), "llm_dir": str(tmp_path / "llm_b")},
    ]
    frames = build_montage(segments, str(tmp_path / "rl_a"), str(tmp_path / "llm_a"), fps=2)
    assert len(frames) == 4


def test_build_montage_empty_range_raises_clear_error(tmp_path):
    _write_frames(tmp_path / "rl", 3)
    (tmp_path / "llm").mkdir()
    segments = [{"seconds": 1, "rl": [0, 3], "llm": [10, 20], "caption": "x"}]
    with pytest.raises(ValueError, match="llm"):
        build_montage(segments, str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
