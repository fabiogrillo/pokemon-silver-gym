import json
from openai import OpenAI


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
            return {"thought": thought, "tool_name": None, "args": {}, "tokens": tokens}
        call = msg.tool_calls[0].function
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return {"thought": thought, "tool_name": call.name, "args": args, "tokens": tokens}