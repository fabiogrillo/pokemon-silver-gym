BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
DIRECTIONS = ("up", "down", "left", "right")


class ToolValidationError(Exception):
    pass


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_MOVE = _tool(
    "move", "Walk several tiles in one direction in the overworld. Stops early if the map "
    "changes or a battle starts.",
    {"direction": {"type": "string", "enum": list(DIRECTIONS)},
     "steps": {"type": "integer", "minimum": 1, "maximum": 10}},
    ["direction", "steps"],
)
_PRESS = _tool(
    "press", "Press a single button once (use in battle, menus, dialogue).",
    {"button": {"type": "string", "enum": list(BUTTONS)}}, ["button"],
)
_GET_STATE = _tool("get_state", "Return the current game state as text.", {}, [])
_WAIT = _tool(
    "wait_frames", "Advance the game without input (skip animations/dialogue).",
    {"n": {"type": "integer", "minimum": 1, "maximum": 240}}, ["n"],
)

OVERWORLD_TOOLS = [_MOVE, _PRESS, _GET_STATE, _WAIT]
BATTLE_TOOLS = [_PRESS, _GET_STATE, _WAIT]


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolValidationError(f"not an integer: {value!r}")


def validate_tool_call(name: str, args: dict, cfg) -> tuple[str, dict]:
    args = dict(args or {})
    if name == "move":
        direction = str(args.get("direction", "")).lower()
        if direction not in DIRECTIONS:
            raise ToolValidationError(f"bad direction: {direction!r}")
        steps = max(1, min(_coerce_int(args.get("steps", 1)), cfg.move_max_steps))
        return "move", {"direction": direction, "steps": steps}
    if name == "press":
        button = str(args.get("button", "")).lower()
        if button not in BUTTONS:
            raise ToolValidationError(f"bad button: {button!r}")
        return "press", {"button": button}
    if name == "get_state":
        return "get_state", {}
    if name == "wait_frames":
        n = max(1, min(_coerce_int(args.get("n", 24)), 240))
        return "wait_frames", {"n": n}
    raise ToolValidationError(f"unknown tool: {name!r}")