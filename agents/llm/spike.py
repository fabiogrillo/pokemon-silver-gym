"""Phase 0 feasibility spike: does qwen3-vl:8b return a tool call from a screenshot?

Run: python -m agents.llm.spike
Requires: `ollama serve` running with qwen3-vl:8b pulled, plus the ROM + save state.
"""
from openai import OpenAI
from agents.llm.config import LLMConfig, SYSTEM_PROMPT
from agents.llm.perception import build_user_content
from agents.llm.tools import OVERWORLD_TOOLS
from env.pyboy_wrapper import PyBoyWrapper
from env.ram_reader import RAMReader


def main():
    cfg = LLMConfig()
    wrapper = PyBoyWrapper(cfg.rom_path, cfg.state_path, headless=True)
    reader = RAMReader(wrapper.pyboy)
    state = reader.read_all()
    frame = wrapper.pyboy.screen.ndarray

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    content = build_user_content(state, frame, "", cfg.send_image)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
        tools=OVERWORLD_TOOLS,
        temperature=cfg.temperature,
        timeout=cfg.request_timeout,
    )
    msg = resp.choices[0].message
    print("THOUGHT:", (msg.content or "").strip()[:200])
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print("TOOL CALL:", tc.function.name, tc.function.arguments)
        print("RESULT: PASS — model returned a tool call.")
    else:
        print("RESULT: FAIL — no tool call. Consider Plan B (text + ASCII map, send_image=False).")
    wrapper.pyboy.stop(save=False)


if __name__ == "__main__":
    main()