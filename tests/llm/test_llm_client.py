from types import SimpleNamespace

from agents.llm.llm_client import parse_tool_call_from_content, _is_empty_reply, _retry_messages


def test_parses_hermes_tool_call_tag():
    content = '<tool_call>{"name": "move", "arguments": {"direction": "right", "steps": 2}}</tool_call>'
    name, args = parse_tool_call_from_content(content)
    assert name == "move"
    assert args == {"direction": "right", "steps": 2}


def test_parses_bare_json_object():
    content = '{"name": "press", "arguments": {"button": "a"}}'
    name, args = parse_tool_call_from_content(content)
    assert name == "press"
    assert args == {"button": "a"}


def test_arguments_as_json_string():
    content = '<tool_call>{"name": "move", "arguments": "{\\"direction\\": \\"up\\", \\"steps\\": 1}"}</tool_call>'
    name, args = parse_tool_call_from_content(content)
    assert name == "move"
    assert args == {"direction": "up", "steps": 1}


def test_plain_prose_returns_none():
    name, args = parse_tool_call_from_content("I think I should move right toward Cherrygrove.")
    assert name is None
    assert args == {}


def test_empty_content_returns_none():
    name, args = parse_tool_call_from_content("")
    assert name is None
    assert args == {}


def test_is_empty_reply_true_when_no_tool_calls_and_no_content():
    msg = SimpleNamespace(tool_calls=None, content="")
    assert _is_empty_reply(msg) is True


def test_is_empty_reply_true_when_content_is_whitespace_only():
    msg = SimpleNamespace(tool_calls=None, content="   \n  ")
    assert _is_empty_reply(msg) is True


def test_is_empty_reply_false_when_tool_calls_present():
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(name="press"))],
                           content="")
    assert _is_empty_reply(msg) is False


def test_is_empty_reply_false_when_content_present_even_without_tool_calls():
    # the model may have emitted the tool call as text (Hermes-style) — that's not "empty",
    # it's handled by the parse_tool_call_from_content fallback instead.
    msg = SimpleNamespace(tool_calls=None, content="<tool_call>{\"name\": \"press\"}</tool_call>")
    assert _is_empty_reply(msg) is False


def test_retry_messages_appends_text_only_nudge():
    original = [{"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": "x"}}]}]
    retried = _retry_messages(original)
    assert retried[-1] == {"role": "user", "content": "Reply with exactly one tool call now."}
    assert isinstance(retried[-1]["content"], str)  # no image in the appended message


def test_retry_messages_does_not_mutate_original_list():
    original = [{"role": "user", "content": "hi"}]
    retried = _retry_messages(original)
    assert len(original) == 1  # original list untouched
    assert len(retried) == 2
