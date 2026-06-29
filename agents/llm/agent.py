import time
from .config import SYSTEM_PROMPT
from .perception import build_user_content
from .tools import OVERWORLD_TOOLS, BATTLE_TOOLS, execute_tool, validate_tool_call, ToolValidationError
from .memory import ShortTermMemory
from .llm_client import OllamaClient


class ReActAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OllamaClient(cfg)

    def run(self, wrapper, reader, on_step=None) -> dict:
        cfg = self.cfg
        mem = ShortTermMemory(cfg.stuck_window * 2, cfg.stuck_window, cfg.stuck_radius)
        tiles, tokens, battles_won = set(), 0, 0
        prev_battle = 0
        stopped = "max_steps"

        for step in range(cfg.max_steps):
            state = reader.read_all()
            tiles.add((state["map_bank"], state["map_number"], state["local_x"], state["local_y"]))

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
            frame = wrapper.pyboy.screen.ndarray
            content = build_user_content(state, frame, mem.render_note(), cfg.send_image)
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}]

            out = self.client.chat(messages, tools)
            tokens += out["tokens"]
            name, args = out["tool_name"], out["args"]

            if name is None:
                obs = {"ok": False, "note": "no tool call", "stopped_early": False}
            else:
                try:
                    name, args = validate_tool_call(name, args, cfg)
                    obs = execute_tool(name, args, wrapper, reader, cfg)
                except ToolValidationError as e:
                    obs = {"ok": False, "note": f"invalid tool: {e}", "stopped_early": False}
                    wrapper.pyboy.tick(count=cfg.frames_per_press)  # no-op advance

            mem.record(state, out["thought"], name or "none", args, obs["note"])
            if on_step:
                on_step(step, state, out, obs)

        return {"badge": reader.read_all()["zephyr"], "steps": step + 1, "tokens": tokens,
                "battles_won": battles_won, "tiles": len(tiles), "stopped": stopped}