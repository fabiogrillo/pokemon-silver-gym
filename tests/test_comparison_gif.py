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
