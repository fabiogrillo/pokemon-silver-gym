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


def test_navigate_to_coerces_string_ints_without_state():
    name, args = validate_tool_call("navigate_to", {"x": "3", "y": "4"}, CFG)
    assert name == "navigate_to"
    assert args == {"x": 3, "y": 4}


def test_navigate_to_rejects_non_integer():
    with pytest.raises(ToolValidationError):
        validate_tool_call("navigate_to", {"x": "abc", "y": 4}, CFG)


def test_navigate_to_accepts_walkable_target_with_state():
    state = {"map_bank": 10, "map_number": 7}  # assets/collision/gym.json; (1, 1) is walkable
    name, args = validate_tool_call("navigate_to", {"x": 1, "y": 1}, CFG, state)
    assert name == "navigate_to"
    assert args == {"x": 1, "y": 1}


def test_navigate_to_rejects_wall_target_with_state():
    state = {"map_bank": 10, "map_number": 7}  # gym.json row 0 is all walls
    with pytest.raises(ToolValidationError):
        validate_tool_call("navigate_to", {"x": 0, "y": 0}, CFG, state)


def test_navigate_to_rejects_out_of_map_target_with_state():
    state = {"map_bank": 10, "map_number": 7}  # gym.json is 10x16
    with pytest.raises(ToolValidationError):
        validate_tool_call("navigate_to", {"x": 999, "y": 999}, CFG, state)