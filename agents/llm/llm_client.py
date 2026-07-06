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


def _is_empty_reply(msg) -> bool:
    """True when a chat completion message ended the turn with no tool call and no content —
    the always-thinking qwen3-vl:8b pathology where the model emits reasoning only. `msg` is
    duck-typed (needs `.tool_calls` and `.content`) so this can be unit-tested without the
    OpenAI SDK or a network call."""
    content = (getattr(msg, "content", None) or "").strip()
    return not getattr(msg, "tool_calls", None) and not content


def _retry_messages(messages: list) -> list:
    """Messages for the single empty-reply retry: append a short text-only nudge asking for a
    tool call. Returns a NEW list — the caller's original `messages` (which may carry an image
    in its last user turn) is left untouched, and the appended message is plain text so the
    screenshot is never resent on retry."""
    return messages + [{"role": "user", "content": "Reply with exactly one tool call now."}]


def _usage_tokens(resp) -> int:
    usage = getattr(resp, "usage", None)
    return usage.total_tokens if usage else 0


class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def _complete(self, messages: list, tools: list):
        return self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            tools=tools,
            temperature=self.cfg.temperature,
            timeout=self.cfg.request_timeout,
        )

    def chat(self, messages: list, tools: list) -> dict:
        resp = self._complete(messages, tools)
        msg = resp.choices[0].message
        tokens = _usage_tokens(resp)

        if _is_empty_reply(msg):
            # qwen3-vl:8b always thinks and intermittently ends a turn with reasoning only
            # (no content, no tool_calls). Don't burn the whole step on that — retry ONCE with
            # a short text-only nudge instead of resending the image.
            retry_resp = self._complete(_retry_messages(messages), tools)
            msg = retry_resp.choices[0].message
            tokens += _usage_tokens(retry_resp)

        # Thinking models (qwen3-vl) put their reasoning in a separate `reasoning` field
        # (Ollama's OpenAI-compat extension) and often leave `content` empty when they emit
        # a structured tool call. Fall back to it so the trace keeps the model's real
        # thoughts — the montage captions are verbatim extracts of this field.
        reasoning = (getattr(msg, "reasoning", None) or "").strip()
        content = (msg.content or "").strip()
        thought = content or reasoning
        if not msg.tool_calls:
            # Fallback: the model may have emitted the tool call as text in `content`
            # (never in `reasoning` — don't mine that channel for spurious JSON spans).
            name, args = parse_tool_call_from_content(content)
            return {"thought": thought, "tool_name": name, "args": args, "tokens": tokens}
        call = msg.tool_calls[0].function
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return {"thought": thought, "tool_name": call.name, "args": args, "tokens": tokens}