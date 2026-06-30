from agents.llm.llm_client import parse_tool_call_from_content


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
