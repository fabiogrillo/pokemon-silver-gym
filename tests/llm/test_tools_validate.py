import pytest
from agents.llm.config import LLMConfig
from agents.llm.tools import validate_tool_call, ToolValidationError

CFG = LLMConfig()


def test_move_clamps_steps_and_lowercases_direction():
    name, args = validate_tool_call("move", {"direction": "UP", "steps": 99}, CFG)
    assert name == "move"
    assert args["direction"] == "up"
    assert args["steps"] == CFG.move_max_steps  # clamped


def test_move_coerces_string_steps():
    _, args = validate_tool_call("move", {"direction": "left", "steps": "3"}, CFG)
    assert args["steps"] == 3


def test_press_rejects_unknown_button():
    with pytest.raises(ToolValidationError):
        validate_tool_call("press", {"button": "x"}, CFG)


def test_move_rejects_non_direction():
    with pytest.raises(ToolValidationError):
        validate_tool_call("move", {"direction": "a", "steps": 2}, CFG)


def test_unknown_tool_raises():
    with pytest.raises(ToolValidationError):
        validate_tool_call("teleport", {}, CFG)