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


# Bound on total presses per navigate_to call. Generous relative to any in-bounds map (the
# largest collision grid is 20x18): a couple of extra steps of slack cover the occasional
# overshoot-then-correct pair described below, without letting a truly stuck call spin forever.
_MAX_STEPS = 40
# How many times to retry a single planned direction before concluding its target tile is
# actually blocked (matches the previous "3 consecutive non-moves" desync guard).
_STEP_RETRIES = 3


def _execute_navigate(args, wrapper, ram_reader, cfg, state0):
    """Walk the A* path one tile at a time, RE-PLANNING from the actual post-press position before
    every single step (not just after a hard block).

    Two independent bugs made the previous "plan once, blindly press the whole direction list"
    design unreliable:

    1. Dynamic obstacles: the static collision grid (assets/collision/*.json) only encodes
       terrain, not NPCs. A* can plan through a tile an NPC happens to be standing on; the engine
       then correctly refuses to move there. Fix: track tiles that fail to move into
       (`blocked_tiles`) and route around them instead of aborting.
    2. Press/walk-cycle misalignment: cfg.frames_per_press (24, tuned for dialogue advancement)
       is not an exact multiple of the overworld's internal per-tile walk-cycle length (16
       frames), so a single fixed-length press can advance 0, 1, *or 2* tiles depending on
       animation phase -- confirmed empirically by holding "right" for repeated 24-frame presses
       from a fixed save state and watching local_y jump by 1 or 2 tiles at random. A stale
       multi-step plan silently drifts off course when this happens (this is what the "target
       (12,11) -> navigated to (13,11)" free-form-run symptom was). Fix: re-derive the plan from
       wherever we actually land after *every* press, so an overshoot just becomes the new start
       position for the next A* call instead of corrupting the rest of a stale route.
    """
    goal = (args["x"], args["y"])
    start_map = _map_key(state0)
    grids = pathfind.load_grids()
    cur_xy = _true_xy(state0)
    blocked_tiles: set = set()

    for _ in range(_MAX_STEPS):
        if cur_xy == goal:
            return {"ok": True, "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]})",
                    "stopped_early": False}

        directions = pathfind.plan(state0["map_bank"], state0["map_number"], cur_xy, goal,
                                    grids=grids, blocked=frozenset(blocked_tiles))
        if not directions:  # None (unreachable) -- goal already handled by the check above
            return {"ok": False, "note": f"no path to ({goal[0]}, {goal[1]})", "stopped_early": True}

        direction = directions[0]
        dx, dy = pathfind.DIR_DELTA[direction]
        target = (cur_xy[0] + dx, cur_xy[1] + dy)
        moved = False
        for _attempt in range(_STEP_RETRIES):
            wrapper.step(_BTN_INDEX[direction], n=cfg.frames_per_press)
            s = ram_reader.read_all()
            new_xy = _true_xy(s)
            if _map_key(s) != start_map:
                return {"ok": True, "note": f"navigated to ({new_xy[0]}, {new_xy[1]})",
                        "stopped_early": True}
            if s["battle_type"] > 0:
                return {"ok": True, "note": f"navigated to ({new_xy[0]}, {new_xy[1]})",
                        "stopped_early": True}
            if new_xy != cur_xy:
                cur_xy = new_xy
                moved = True
                break
        if not moved:
            # Genuinely can't step onto `target` right now (e.g. an NPC occupying a tile the
            # static grid calls walkable) -- mark it and re-plan a detour next iteration.
            blocked_tiles.add(target)

    return {"ok": False, "note": f"path blocked at ({cur_xy[0]}, {cur_xy[1]})",
            "stopped_early": True}


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