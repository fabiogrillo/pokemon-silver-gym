import io

from . import pathfind
from . import sprites
from .perception import format_state_text
from env.actions import ACTIONS

BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
DIRECTIONS = ("up", "down", "left", "right")

# PyBoyWrapper.step() indexes ACTIONS by an integer (it serves the RL discrete action
# space). Map our validated button/direction names back to that index.
_BTN_INDEX = {name: i for i, name in enumerate(ACTIONS)}


def probe_walkable(wrapper, reader, n=24, presses=3, confine_to_home_map=False):
    """Return the directions the player can actually step in, by save->try->restore.

    For each direction we reload a snapshot, press it `presses` times (the first press may only
    turn the character), and see if the local position changed. This is the navigation signal the
    vision-only model lacks: it cleanly distinguishes "I'm boxed in by a trainer/dialogue" (no
    direction walkable -> press 'a' to engage) from "I just need to walk on" (some direction open).
    Pure read-only w.r.t. the real episode: the final reload restores the pre-probe state exactly.

    When `confine_to_home_map` is True (gym-slice harness only), a direction that changes the map
    is treated as NOT walkable — this steers the gym-only agent away from the exit. For the
    corridor task (the default) map-changing directions ARE genuine exits and are reported as
    walkable, since leaving the current map is exactly how the agent makes progress.

    Lives here (not agents/llm/agent.py) so the navigate_to executor's probe-verified fallback
    (see _execute_navigate's bug 5) can reuse the exact same live-emulator signal the ReAct loop
    itself uses to advise the model, without an agent.py<->tools.py import cycle (agent.py already
    imports from this module). agent.py re-exports/imports it from here for its per-step probe.
    """
    snap = wrapper.save_state_bytes()
    s = reader.read_all()
    before = (s["local_x"], s["local_y"])
    home_map = (s["map_bank"], s["map_number"])
    open_dirs = []
    for d in DIRECTIONS:
        wrapper.pyboy.load_state(io.BytesIO(snap))
        for _ in range(presses):
            wrapper.step(_BTN_INDEX[d], n=n)
        s2 = reader.read_all()
        moved = (s2["local_x"], s2["local_y"]) != before
        same_map = (s2["map_bank"], s2["map_number"]) == home_map
        if moved and (same_map or not confine_to_home_map):
            open_dirs.append(d)
    wrapper.pyboy.load_state(io.BytesIO(snap))
    wrapper.pyboy.tick(1)  # refresh the rendered frame after the final restore
    return open_dirs


def _greedy_direction(open_dirs, current, goal):
    """Pick the open direction that best advances `current` toward `goal`, for the probe-verified
    fallback _execute_navigate falls back to when pathfind.plan() can't find a route (see bug 5).

    "Best" = smallest post-step Manhattan distance to `goal`. Since a single step only ever
    changes one axis by 1, any direction that walks along the correct axis (toward, not away from,
    the goal) reduces the total Manhattan distance by exactly the same amount (1) — so whenever
    both the correct x-direction and the correct y-direction are open, this is a tie by
    construction, broken toward the axis with the larger remaining delta (closing the bigger gap
    first). Directions that don't reduce distance at all are still ranked (least-bad first) so the
    fallback always has an answer as long as ANY direction is open — see the docstring on why: the
    caller needs to make forward progress even when no direction is actively "toward" the goal
    (e.g. sidestepping around an obstacle).

    Iterates in the canonical DIRECTIONS order (not `open_dirs`'s order) so ties are resolved the
    same way regardless of what order the probe happened to return them in. Returns None if
    `open_dirs` is empty (every direction is currently sealed).
    """
    open_set = set(open_dirs)
    best_direction, best_key = None, None
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    for direction in DIRECTIONS:
        if direction not in open_set:
            continue
        ddx, ddy = pathfind.DIR_DELTA[direction]
        new_xy = (current[0] + ddx, current[1] + ddy)
        dist = abs(new_xy[0] - goal[0]) + abs(new_xy[1] - goal[1])
        axis_delta = abs(dx) if ddx != 0 else abs(dy)
        key = (dist, -axis_delta)  # lower dist wins; among ties, larger axis_delta wins
        if best_key is None or key < best_key:
            best_direction, best_key = direction, key
    return best_direction


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
# How many extra outward presses to try once we've arrived at a border-tile target -- GSC map
# connections trigger on stepping PAST the edge, one tile further than any in-grid A* target can
# reach (see pathfind.border_exit_direction). 3 matches _STEP_RETRIES's slack for the walk-cycle
# overshoot described above.
_BORDER_EXIT_RETRIES = 3
# How many 'a' presses to try, once a direction has exhausted _STEP_RETRIES with zero movement,
# before concluding the target tile is genuinely blocked by terrain/a stationary obstacle -- see
# _clear_blocking_interaction's docstring ("bug 4").
_INTERACTION_CLEAR_RETRIES = 8
# How many consecutive times the probe-verified greedy fallback (see bug 5 below) may fire in a
# row -- i.e. pathfind.plan() failing repeatedly with no successful re-plan in between -- before
# concluding the position is genuinely stuck (rather than just one roaming NPC that will likely
# have moved on by the model's next tool call) and returning a non-terminal, self-describing
# observation instead of spinning for the rest of _MAX_STEPS.
_GREEDY_FALLBACK_LIMIT = 3


def _clear_blocking_interaction(wrapper, ram_reader, cfg, start_map, cur_xy):
    """Press 'a' up to `_INTERACTION_CLEAR_RETRIES` times to clear a scripted NPC dialogue that has
    seized directional input, then report what happened.

    Route maps have stationary NPCs (e.g. Route 29's rival, "I've seen you a couple times...")
    that trigger a forced-facing dialogue -- not a battle yet, not a map change -- the instant the
    player walks within their sight line. While that dialogue box is open, GSC ignores directional
    input entirely, so a live player standing at the trigger tile can fail to move in literally
    *every* direction (confirmed directly against the emulator: from Route 29 true (10, 7), the
    ledge-hop direction 'left' -- and, after marking it blocked, 'up'/'down'/'right' too -- all
    fail 3/3 retries with zero position change and battle_type staying 0 throughout). The old
    executor had no notion of this state: it only reacted to a single occupied *tile* (bug 1's
    NPC-collision fix), so it walled off every neighbor into `blocked_tiles` one at a time and
    finished by reporting a permanent, misleading "no path"/"path blocked" -- reproducing the
    "no path to (0, 7)" symptom this task set out to fix. The real fix mirrors what a human player
    (and this codebase's own SYSTEM_PROMPT, which already tells the LLM "press 'a' until battle
    starts or dialogue clears") would do: press 'a' to advance the dialogue, which either clears
    (ordinary NPC greeting) or leads into a battle (confirmed both live: mashing 'a' from the
    Route 29 trigger above ends in battle_type=1 within a handful of presses).

    Returns one of:
    - ("battle"|"map_change", new_xy): the dialogue resolved into something the walk loop's own
      early-stop convention already handles; caller should return that observation immediately.
    - ("moved", new_xy): a scripted cutscene shuffled the player without a battle/map change (also
      observed live -- an NPC "notices you" script can auto-step the player a tile); caller should
      accept this as progress and re-plan from `new_xy` instead of marking a tile blocked.
    - ("none", cur_xy): nothing happened after every retry -- an ordinary terrain/NPC-tile block,
      not a dialogue. Caller falls back to the pre-existing blocked_tiles behavior.
    """
    for _ in range(_INTERACTION_CLEAR_RETRIES):
        wrapper.step(_BTN_INDEX["a"], n=cfg.frames_per_press, settle=cfg.settle_frames)
        s = ram_reader.read_all()
        new_xy = _true_xy(s)
        if _map_key(s) != start_map:
            return "map_change", new_xy
        if s["battle_type"] > 0:
            return "battle", new_xy
        if new_xy != cur_xy:
            return "moved", new_xy
    return "none", cur_xy


def _finish_at_goal(wrapper, ram_reader, cfg, grids, start_map, cur_xy, last_direction):
    """Arrived at the A* target `cur_xy` (same map as the call started on).

    If `cur_xy` is an interior tile, the leg is simply done. If it's on the current map's border,
    GSC won't trigger the neighboring-map connection until the player steps past the edge -- so
    press the outward direction (pathfind.border_exit_direction) a few more times, reusing the
    same map-change/battle early-stop checks as the main walk loop. This turns a leg that targets
    an exit tile into an actual transition instead of parking on the border forever.
    """
    grid = grids.get(start_map)
    exit_direction = pathfind.border_exit_direction(grid, cur_xy, last_direction) if grid else None
    if exit_direction is None:
        return {"ok": True, "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]})", "stopped_early": False}

    for _ in range(_BORDER_EXIT_RETRIES):
        wrapper.step(_BTN_INDEX[exit_direction], n=cfg.frames_per_press)
        s = ram_reader.read_all()
        new_xy = _true_xy(s)
        if _map_key(s) != start_map:
            return {"ok": True,
                    "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]}) and crossed into the next "
                            f"map at ({new_xy[0]}, {new_xy[1]})",
                    "stopped_early": True}
        if s["battle_type"] > 0:
            return {"ok": True, "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]})",
                    "stopped_early": True}
        cur_xy = new_xy  # rare walk-cycle overshoot could still move us without crossing yet

    return {"ok": True, "note": f"navigated to ({cur_xy[0]}, {cur_xy[1]})", "stopped_early": False}


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
    3. Landing mid-hop over a ledge: `assets/collision/*.json` ledge tiles (and the non-walkable
       "buffer" tile immediately past each one -- see pathfind.ledge_recovery_direction's
       docstring) are intentionally 0 in the static grid, since pathfind.py's A* only ever crosses
       a ledge as a single atomic hop edge straight from the foothold to the far landing tile. But
       the *live* player can genuinely end up standing exactly on either of those two tiles
       mid-hop if a press's frame budget doesn't cover the whole jump animation in one go (same
       root cause as bug 2 above) -- confirmed directly against the emulator
       (saves/crossing.state, walking towards a nearby interior tile overshoots onto Route 30's
       ledge tile (4, 2) and, separately, onto its buffer tile (4, 3)). Once that happens,
       `pathfind.plan()`'s very first check (`grid.is_walkable(start)`) fails for literally any
       goal, and the walk loop below would otherwise report "no path" forever with no way to move
       off of it (this reproduces the "no path to (0, 7)" x334 symptom this task set out to fix).
       Fix: before planning, check `pathfind.ledge_recovery_direction`; if `cur_xy` is mid-hop,
       skip planning and press the recovered direction directly -- pressing it again is exactly
       how the real game gets you off of either tile too.
    4. Stuck in a scripted NPC dialogue: a stationary NPC's "spotted you" trigger seizes
       directional input in every direction at once (not just the tile it's blocking) until its
       dialogue is advanced with 'a' -- see _clear_blocking_interaction's docstring for the full
       live-emulator trace. Fix: once a direction has exhausted _STEP_RETRIES with zero movement,
       try _clear_blocking_interaction before concluding the target is genuinely blocked.
    5. plan() failing from the LIVE position: `blocked_tiles` (bug 1) only accumulates tiles this
       call has already tried and failed to step into -- it can't see a roaming NPC standing
       somewhere else on the route *before* A* ever tries to cross it, so `pathfind.plan()` can
       return None (no path found on the static grid + blocked_tiles) purely because a currently-
       occupied tile happens to sit on the only route the static grid knows about. The old executor
       treated `plan() is None` as terminal and reported a permanent "no path", even though the
       live emulator (unlike the stale grid) can always be asked directly whether a neighboring
       tile is currently steppable -- this reproduced 957x and 334x "no path" stalls that a human
       player would have simply walked around. Fix: when plan() fails (and we're not mid-ledge-hop,
       point 3 above), fall back to `probe_walkable` -- the same save/try/restore emulator probe
       the ReAct loop itself uses to advise the model -- and `_greedy_direction` picks whichever
       open direction most reduces Manhattan distance to `goal`. Taking that one step and letting
       the loop re-plan from the new tile self-heals the moment the NPC has moved off the blocking
       tile (typically within a step or two). `_GREEDY_FALLBACK_LIMIT` consecutive fallbacks with
       no successful re-plan in between (a genuinely sealed position, not a transient NPC) give up
       with a non-terminal, self-describing observation instead of spinning for `_MAX_STEPS`.
    """
    goal = (args["x"], args["y"])
    start_map = _map_key(state0)
    grids = pathfind.load_grids()
    grid = grids.get(start_map)
    cur_xy = _true_xy(state0)
    blocked_tiles: set = set()
    last_direction = None  # direction of the last successful step; feeds border_exit_direction's
                            # corner tie-break once we arrive at `goal` (see _finish_at_goal).
    greedy_fallbacks = 0    # consecutive plan() failures resolved via the probe-verified fallback
                            # (docstring point 5); reset to 0 the moment plan() succeeds again.

    for _ in range(_MAX_STEPS):
        if cur_xy == goal:
            return _finish_at_goal(wrapper, ram_reader, cfg, grids, start_map, cur_xy,
                                    last_direction)

        recovery_direction = pathfind.ledge_recovery_direction(grid, cur_xy) if grid else None
        if recovery_direction is not None:
            # Mid-hop over a ledge (see docstring point 3 above) -- not a plannable A* node.
            # `target` is left None since there's no single adjacent tile to mark blocked if this
            # doesn't move us (see the `if not moved` handling below).
            direction = recovery_direction
            target = None
        else:
            # Live sprites (roaming NPCs, the scripted rival, camping trainers) re-read from the
            # game's own object table before EVERY plan, so A* routes around people it has never
            # bumped into -- blocked_tiles then only has to catch what the table can't (see
            # agents/llm/sprites.py). The goal tile itself stays plannable: if a person is
            # standing ON the goal, walking up to them and engaging is the right move, and
            # astar() refusing `goal_xy in blocked` would report a misleading permanent block.
            npc_tiles = sprites.live_npc_tiles(wrapper.pyboy) - {goal}
            directions = pathfind.plan(state0["map_bank"], state0["map_number"], cur_xy, goal,
                                        grids=grids,
                                        blocked=frozenset(blocked_tiles) | npc_tiles)
            if directions:  # goal-already-reached case is handled by the check above
                greedy_fallbacks = 0
                direction = directions[0]
                dx, dy = pathfind.DIR_DELTA[direction]
                target = (cur_xy[0] + dx, cur_xy[1] + dy)
            else:
                # plan() found no route from the live position (docstring point 5) -- expensive
                # (4 save/restores), so only run the probe here, never on the happy path above.
                greedy_fallbacks += 1
                if greedy_fallbacks > _GREEDY_FALLBACK_LIMIT:
                    # LLM-5: name the likely cause AND the unblocking action. Trainers/NPCs camping
                    # the only corridor tile can hold this state for many turns; walking around them
                    # is impossible, so the model must ENGAGE (press 'a' facing them -> dialogue or
                    # battle clears the tile) instead of re-calling navigate_to forever (LLM-4 runs
                    # showed 200+ identical retries at one tile).
                    return {"ok": False,
                            "note": f"path blocked around ({cur_xy[0]}, {cur_xy[1]}) — a person is "
                                    f"probably standing in the way. Face them and press 'a' to "
                                    f"talk/battle (this clears the path), then call navigate_to "
                                    f"again",
                            "stopped_early": True}
                open_dirs = probe_walkable(wrapper, ram_reader, n=cfg.frames_per_press)
                direction = _greedy_direction(open_dirs, cur_xy, goal)
                if direction is None:
                    # Every direction is sealed right now -- the same all-directions-blocked
                    # signature _clear_blocking_interaction's docstring documents for a scripted
                    # dialogue (bug 4), just caught before a step was ever attempted this time.
                    outcome, new_xy = _clear_blocking_interaction(wrapper, ram_reader, cfg,
                                                                    start_map, cur_xy)
                    if outcome in ("battle", "map_change"):
                        return {"ok": True, "note": f"navigated to ({new_xy[0]}, {new_xy[1]})",
                                "stopped_early": True}
                    if outcome == "moved":
                        cur_xy = new_xy  # re-plan/re-probe from wherever the cutscene left us
                    continue
                target = None

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
                last_direction = direction
                break
        if not moved:
            # Genuinely can't step onto `target` right now -- either an NPC occupying a tile the
            # static grid calls walkable, or (see docstring point 4) a scripted dialogue that has
            # seized input in every direction. Try clearing the latter before giving up.
            outcome, new_xy = _clear_blocking_interaction(wrapper, ram_reader, cfg, start_map,
                                                            cur_xy)
            if outcome in ("battle", "map_change"):
                return {"ok": True, "note": f"navigated to ({new_xy[0]}, {new_xy[1]})",
                        "stopped_early": True}
            if outcome == "moved":
                cur_xy = new_xy  # a cutscene shuffled us; re-plan from here next iteration
            elif target is not None:
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