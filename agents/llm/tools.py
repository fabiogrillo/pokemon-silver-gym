from . import pathfind
from .perception import format_state_text
from env.actions import ACTIONS

BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
DIRECTIONS = ("up", "down", "left", "right")

# PyBoyWrapper.step() indexes ACTIONS by an integer (it serves the RL discrete action
# space). Map our validated button/direction names back to that index.
_BTN_INDEX = {name: i for i, name in enumerate(ACTIONS)}


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
_NAVIGATE = _tool(
    "navigate_to",
    "Walk to a specific (x, y) tile on the CURRENT map, pathing around walls automatically "
    "(A*). Coordinates are TRUE map tile coordinates: x grows EAST, y grows SOUTH. Prefer this "
    "over `move` when you know the destination tile. Stops early if the map changes, a battle "
    "starts, or the path is blocked (desync guard).",
    {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"],
)

OVERWORLD_TOOLS = [_NAVIGATE, _MOVE, _PRESS, _GET_STATE, _WAIT]
BATTLE_TOOLS = [_PRESS, _GET_STATE, _WAIT]


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolValidationError(f"not an integer: {value!r}")


def _map_key(state):
    return (state["map_bank"], state["map_number"])


def validate_tool_call(name: str, args: dict, cfg, state: dict | None = None) -> tuple[str, dict]:
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
    if name == "navigate_to":
        x = _coerce_int(args.get("x"))
        y = _coerce_int(args.get("y"))
        if state is not None:
            grid = pathfind.load_grids().get(_map_key(state))
            if grid is None or not grid.is_walkable(x, y):
                raise ToolValidationError(
                    f"target ({x}, {y}) is out of the current map bounds or not walkable")
        return "navigate_to", {"x": x, "y": y}
    raise ToolValidationError(f"unknown tool: {name!r}")


def _true_xy(state):
    # RAM local_x/local_y are swapped relative to their names (see
    # agents/rl/map_layout.ram_to_image_px's docstring); TRUE (x, y) is (local_y, local_x).
    return state["local_y"], state["local_x"]


def _execute_navigate(args, wrapper, ram_reader, cfg, state0):
    goal = (args["x"], args["y"])
    directions = pathfind.plan(state0["map_bank"], state0["map_number"], _true_xy(state0), goal)
    if directions is None:
        return {"ok": False, "note": f"no path to ({goal[0]}, {goal[1]})", "stopped_early": True}

    start_map = _map_key(state0)
    cur_xy = _true_xy(state0)
    non_moves = 0
    reason = None
    for direction in directions:
        wrapper.step(_BTN_INDEX[direction], n=cfg.frames_per_press)
        s = ram_reader.read_all()
        new_xy = _true_xy(s)
        non_moves = 0 if new_xy != cur_xy else non_moves + 1
        cur_xy = new_xy
        if _map_key(s) != start_map:
            reason = "map_change"
            break
        if s["battle_type"] > 0:
            reason = "battle"
            break
        if non_moves >= 3:
            reason = "blocked"
            break

    if reason == "blocked":
        return {"ok": False, "note": f"path blocked at ({cur_xy[0]}, {cur_xy[1]})",
                "stopped_early": True}
    return {"ok": True, "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]})",
            "stopped_early": reason is not None}


def execute_tool(name, args, wrapper, ram_reader, cfg) -> dict:
    state0 = ram_reader.read_all()
    name, args = validate_tool_call(name, args, cfg, state0)
    if name == "get_state":
        return {"ok": True, "note": format_state_text(state0), "stopped_early": False}
    if name == "wait_frames":
        wrapper.pyboy.tick(count=args["n"])
        return {"ok": True, "note": f"waited {args['n']} frames", "stopped_early": False}
    if name == "press":
        wrapper.step(_BTN_INDEX[args["button"]], n=cfg.frames_per_press, settle=cfg.settle_frames)
        return {"ok": True, "note": f"pressed {args['button']}", "stopped_early": False}
    if name == "move":
        start_map = _map_key(state0)
        taken, stopped = 0, False
        for _ in range(args["steps"]):
            wrapper.step(_BTN_INDEX[args["direction"]], n=cfg.frames_per_press)
            taken += 1
            s = ram_reader.read_all()
            if _map_key(s) != start_map or s["battle_type"] > 0:
                stopped = True
                break
        return {"ok": True, "note": f"moved {args['direction']} x{taken}",
                "stopped_early": stopped}
    if name == "navigate_to":
        return _execute_navigate(args, wrapper, ram_reader, cfg, state0)
    return {"ok": False, "note": f"unhandled tool {name}", "stopped_early": False}