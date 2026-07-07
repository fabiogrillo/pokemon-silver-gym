import io
import time
from .config import SYSTEM_PROMPT
from .perception import build_user_content, waypoint_ordinal
from .tools import (OVERWORLD_TOOLS, BATTLE_TOOLS, execute_tool, validate_tool_call,
                     ToolValidationError, probe_walkable)
from .memory import ShortTermMemory
from .llm_client import OllamaClient
from .legs import LegTracker


class ReActAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OllamaClient(cfg)
        self.legs = LegTracker()  # LLM-3: harness-owned corridor-leg goals (see cfg.leg_mode)

    def run(self, wrapper, reader, on_step=None) -> dict:
        cfg = self.cfg
        mem = ShortTermMemory(cfg.stuck_window * 2, cfg.stuck_window, cfg.stuck_radius)
        tiles, tokens, battles_won = set(), 0, 0
        max_wp = 0
        prev_battle = 0
        stopped = "max_steps"
        prev_xy, last_move, blocked = None, None, set()
        home_map = None       # boot map; only enforced as a confinement boundary when
                              # cfg.confine_to_home_map is True (gym-slice harness, mirrors RL
                              # CONFINE_TO_GYM). False (default) for the corridor task: map
                              # transitions are a normal, successful outcome and are never undone.
        gym_anchor = None     # last snapshot CONFIRMED safely inside home_map, for robust undo
                              # (gym-slice harness only)

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
                open_dirs = probe_walkable(wrapper, reader, n=cfg.frames_per_press,
                                            confine_to_home_map=cfg.confine_to_home_map)
                if open_dirs:
                    note += f"\nWalkable directions from here: {', '.join(open_dirs)}."
                    if blocked:
                        note += (f" You already tried to move {'/'.join(sorted(blocked))} and did NOT "
                                 f"move — those are blocked walls, do NOT pick them again.")
                    if cfg.confine_to_home_map:
                        # gym-slice harness only: prefer 'up' (Falkner is at the top of the gym).
                        rec = next((d for d in ("up", "right", "left", "down")
                                    if d in open_dirs and d not in blocked), None)
                        if rec:
                            note += f" RECOMMENDED next action: move {rec} (head toward Falkner at the top)."
                else:
                    note += ("\nWalkable directions from here: NONE — a trainer or sign is blocking "
                             "you; press 'a' to interact/battle.")
                if cfg.leg_mode:
                    # LLM-3: the harness (not the model) picks the current strategic target — see
                    # agents/llm/legs.py's module docstring for why (LLM-2's model perseverated on
                    # one mid-town coordinate). local_x/local_y are RAM-swapped; un-swap to TRUE
                    # (x, y) before handing off, mirroring perception.format_state_text.
                    true_x, true_y = state["local_y"], state["local_x"]
                    note += "\n" + self.legs.goal_note(
                        state["map_bank"], state["map_number"], true_x, true_y)
            content = build_user_content(state, frame, note, cfg.send_image)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}]

            out = self.client.chat(messages, tools)
            tokens += out["tokens"]
            name, args = out["tool_name"], out["args"]

            if name is None:
                # qwen3-vl sometimes returns an empty response (no content, no tool_calls) even
                # after llm_client's retry. In battle, pressing 'a' is still correct (advances
                # dialogue/attack menus). In the overworld it is NOT safe: facing an NPC, 'a'
                # self-feeds their textbox (open/close) forever with no way out. Use a harmless
                # no-op wait there instead. `out["tool_name"]` stays None so callers can still
                # measure the empty-response rate.
                if state["battle_type"] == 0:
                    execute_tool("wait_frames", {"n": 24}, wrapper, reader, cfg)
                    obs = {"ok": False, "note": "no tool call → fallback wait_frames",
                           "stopped_early": False}
                else:
                    execute_tool("press", {"button": "a"}, wrapper, reader, cfg)
                    obs = {"ok": False, "note": "no tool call → fallback press a",
                           "stopped_early": False}
            else:
                try:
                    name, args = validate_tool_call(name, args, cfg, state)
                    obs = execute_tool(name, args, wrapper, reader, cfg)
                except ToolValidationError as e:
                    obs = {"ok": False, "note": f"invalid tool: {e}", "stopped_early": False}
                    wrapper.pyboy.tick(count=cfg.frames_per_press)  # no-op advance

            # gym confinement (mirrors RL CONFINE_TO_GYM): gym-slice harness ONLY. When disabled
            # (default, corridor task) a map change is a normal, successful outcome — it is never
            # undone, and the probe (see confine_to_home_map above) already reports exit
            # directions as walkable.
            if cfg.confine_to_home_map:
                def _in_gym():
                    s = reader.read_all()
                    return (s["map_bank"], s["map_number"]) == home_map

                after = reader.read_all()
                if (after["map_bank"], after["map_number"]) != home_map:
                    # Restore robustly: some door warps leave the immediate pre_snap "armed" so a
                    # plain reload re-fires it. Try this step's clean snapshot, then the last
                    # confirmed-safe anchor, each a few times, until we are verifiably back inside
                    # the gym.
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

        visited_maps = {(b, n) for b, n, _x, _y in tiles}
        return {"badge": reader.read_all()["zephyr"], "steps": step + 1, "tokens": tokens,
                "battles_won": battles_won, "tiles": len(tiles), "stopped": stopped,
                "max_waypoint": max_wp, "legs_completed": self.legs.completed_count(visited_maps)}