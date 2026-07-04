import numpy as np
import pytest
import imageio.v3 as iio
from agents.make_comparison_gif import (
    shorten_thought, wrap_caption, upscale2x, compose_frame,
    CANVAS_W, CANVAS_H, load_frames, resample, build_montage,
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


def test_build_montage_empty_range_raises_clear_error(tmp_path):
    _write_frames(tmp_path / "rl", 3)
    (tmp_path / "llm").mkdir()
    segments = [{"seconds": 1, "rl": [0, 3], "llm": [10, 20], "caption": "x"}]
    with pytest.raises(ValueError, match="llm"):
        build_montage(segments, str(tmp_path / "rl"), str(tmp_path / "llm"), fps=4)
