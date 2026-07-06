import io
import time
from env.actions import ACTIONS
from .config import SYSTEM_PROMPT
from .perception import build_user_content, waypoint_ordinal
from .tools import OVERWORLD_TOOLS, BATTLE_TOOLS, execute_tool, validate_tool_call, ToolValidationError
from .memory import ShortTermMemory
from .llm_client import OllamaClient

_DIRS = ("up", "down", "left", "right")


def probe_walkable(wrapper, reader, n=24, presses=3):
    """Return the directions the player can actually step in, by save->try->restore.

    For each direction we reload a snapshot, press it `presses` times (the first press may only
    turn the character), and see if the local position changed. This is the navigation signal the
    vision-only model lacks: it cleanly distinguishes "I'm boxed in by a trainer/dialogue" (no
    direction walkable -> press 'a' to engage) from "I just need to walk on" (some direction open).
    Pure read-only w.r.t. the real episode: the final reload restores the pre-probe state exactly.
    """
    snap = wrapper.save_state_bytes()
    s = reader.read_all()
    before = (s["local_x"], s["local_y"])
    home_map = (s["map_bank"], s["map_number"])
    open_dirs = []
    for d in _DIRS:
        wrapper.pyboy.load_state(io.BytesIO(snap))
        for _ in range(presses):
            wrapper.step(ACTIONS.index(d), n=n)
        s2 = reader.read_all()
        moved = (s2["local_x"], s2["local_y"]) != before
        # only count it walkable if we stayed on this map — a direction that changes the map is
        # an EXIT, which we never want to offer (the goal is up, not out of the gym).
        same_map = (s2["map_bank"], s2["map_number"]) == home_map
        if moved and same_map:
            open_dirs.append(d)
    wrapper.pyboy.load_state(io.BytesIO(snap))
    wrapper.pyboy.tick(1)  # refresh the rendered frame after the final restore
    return open_dirs


class ReActAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OllamaClient(cfg)

    def run(self, wrapper, reader, on_step=None) -> dict:
        cfg = self.cfg
        mem = ShortTermMemory(cfg.stuck_window * 2, cfg.stuck_window, cfg.stuck_radius)
        tiles, tokens, battles_won = set(), 0, 0
        max_wp = 0
        prev_battle = 0
        stopped = "max_steps"
        prev_xy, last_move, blocked = None, None, set()
        home_map = None       # the gym map; the agent is confined to it (mirrors RL CONFINE_TO_GYM)
        gym_anchor = None     # last snapshot CONFIRMED safely inside the gym, for robust undo

        for step in range(cfg.max_steps):
            state = reader.read_all()
            cur_xy = (state["map_bank"], state["map_number"], state["local_x"], state["local_y"])
            tiles.add(cur_xy)
            max_wp = max(max_wp, waypoint_ordinal(state["map_bank"], state["map_number"]))
            if home_map is None:
                home_map = (state["map_bank"], state["map_number"])
            # Clean pre-action snapshot, taken BEFORE the probe so it never carries a warp the
            # probe may have armed while testing directions near a door.
            pre_snap = wrapper.save_state_bytes()

            # anti-fixation: if the previous action was a move that did NOT change our position,
            # that direction is blocked in practice — remember it so we stop suggesting it. Any
            # real position change clears the blocklist (we are at a fresh tile).
            if prev_xy is not None and last_move is not None:
                if cur_xy == prev_xy:
                    blocked.add(last_move)
                else:
                    blocked.clear()

            # battle win edge: leaving a battle with enemy hp 0
            if prev_battle > 0 and state["battle_type"] == 0:
                battles_won += 1  # refined in Task 11
            prev_battle = state["battle_type"]

            if state["zephyr"]:
                stopped = "badge"
                break
            if tokens > cfg.token_budget:
                stopped = "token_budget"
                break

            tools = OVERWORLD_TOOLS if state["battle_type"] == 0 else BATTLE_TOOLS
            frame = wrapper.pyboy.screen.ndarray.copy()  # capture before probing restores state
            note = mem.render_note()
            if state["battle_type"] == 0:
                open_dirs = probe_walkable(wrapper, reader, n=cfg.frames_per_press)
                if open_dirs:
                    note += f"\nWalkable directions from here: {', '.join(open_dirs)}."
                    if blocked:
                        note += (f" You already tried to move {'/'.join(sorted(blocked))} and did NOT "
                                 f"move — those are blocked walls, do NOT pick them again.")
                    # prefer 'up' (Falkner is at the top); skip directions known-blocked this tile.
                    rec = next((d for d in ("up", "right", "left", "down")
                                if d in open_dirs and d not in blocked), None)
                    if rec:
                        note += f" RECOMMENDED next action: move {rec} (head toward Falkner at the top)."
                else:
                    note += ("\nWalkable directions from here: NONE — a trainer or sign is blocking "
                             "you; press 'a' to interact/battle.")
            content = build_user_content(state, frame, note, cfg.send_image)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}]

            out = self.client.chat(messages, tools)
            tokens += out["tokens"]
            name, args = out["tool_name"], out["args"]

            if name is None:
                # qwen3-vl sometimes returns an empty response (no content, no tool_calls).
                # Don't waste the turn: press 'a' — it advances battles/dialogue (where empty
                # responses cluster) and is harmless in the overworld. `out["tool_name"]` stays
                # None so callers can still measure the empty-response rate.
                execute_tool("press", {"button": "a"}, wrapper, reader, cfg)
                obs = {"ok": False, "note": "no tool call → fallback press a", "stopped_early": False}
            else:
                try:
                    name, args = validate_tool_call(name, args, cfg, state)
                    obs = execute_tool(name, args, wrapper, reader, cfg)
                except ToolValidationError as e:
                    obs = {"ok": False, "note": f"invalid tool: {e}", "stopped_early": False}
                    wrapper.pyboy.tick(count=cfg.frames_per_press)  # no-op advance

            # gym confinement (mirrors RL CONFINE_TO_GYM): if the action left the gym map, undo it.
            # The restored "no movement" also lets the anti-fixation guardrail mark that exit blocked.
            def _in_gym():
                s = reader.read_all()
                return (s["map_bank"], s["map_number"]) == home_map

            after = reader.read_all()
            if (after["map_bank"], after["map_number"]) != home_map:
                # Restore robustly: some door warps leave the immediate pre_snap "armed" so a plain
                # reload re-fires it. Try this step's clean snapshot, then the last confirmed-safe
                # anchor, each a few times, until we are verifiably back inside the gym.
                restored = False
                for snap in (pre_snap, gym_anchor):
                    if snap is None:
                        continue
                    for _ in range(4):
                        wrapper.pyboy.load_state(io.BytesIO(snap))
                        wrapper.pyboy.tick(1)
                        if _in_gym():
                            restored = True
                            break
                    if restored:
                        break
                obs = {**obs, "note": obs["note"] + " | cannot leave the gym (move undone)"}
            elif _in_gym():
                # the action kept us safely inside — remember this as a clean fallback anchor
                gym_anchor = pre_snap

            mem.record(state, out["thought"], name or "none", args, obs["note"])
            prev_xy = cur_xy
            last_move = args.get("direction") if name == "move" else None
            if on_step:
                on_step(step, state, out, obs, frame)

        return {"badge": reader.read_all()["zephyr"], "steps": step + 1, "tokens": tokens,
                "battles_won": battles_won, "tiles": len(tiles), "stopped": stopped,
                "max_waypoint": max_wp}