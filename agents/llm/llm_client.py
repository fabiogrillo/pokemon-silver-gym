import json
import re
from openai import OpenAI

# Some models (qwen3-vl on Ollama) sometimes emit the tool call as TEXT in the message
# content (Hermes-style `<tool_call>{...}</tool_call>`) instead of the structured
# `tool_calls` field — especially when also asked to produce reasoning text. Parse that
# text channel as a fallback so no turn is wasted.
_TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call_from_content(content: str):
    """Extract (name, args) from a text-format tool call, or (None, {}) if absent."""
    if not content:
        return None, {}
    m = _TOOLCALL_TAG_RE.search(content)
    if m:
        candidate = m.group(1)
    else:  # fall back to the first {...} span that might be a tool call
        start, end = content.find("{"), content.rfind("}")
        candidate = content[start:end + 1] if 0 <= start < end else None
    if not candidate:
        return None, {}
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None, {}
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    return name, (args if isinstance(args, dict) else {})


class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def chat(self, messages: list, tools: list) -> dict:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            tools=tools,
            temperature=self.cfg.temperature,
            timeout=self.cfg.request_timeout,
        )
        msg = resp.choices[0].message
        tokens = getattr(resp, "usage", None)
        tokens = tokens.total_tokens if tokens else 0
        thought = (msg.content or "").strip()
        if not msg.tool_calls:
            # Fallback: the model may have emitted the tool call as text in `content`.
            name, args = parse_tool_call_from_content(thought)
            return {"thought": thought, "tool_name": name, "args": args, "tokens": tokens}
        call = msg.tool_calls[0].function
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return {"thought": thought, "tool_name": call.name, "args": args, "tokens": tokens}