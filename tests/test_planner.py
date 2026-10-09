from types import SimpleNamespace

import pytest

from tileset_gen.planner import MODEL, PlannerError, WireSpec, plan, system_prompt

GOOD = {
    "name": "meadow",
    "theme": "a sunny meadow",
    "palette": [{"name": "grass", "hex": "#5aa83c"}, {"name": "water", "hex": "#3a78c8"}],
    "terrains": [
        {"name": "water", "generator": "water", "colors": ["water"], "variants": 1, "params": []},
        {
            "name": "grass",
            "generator": "grass",
            "colors": ["grass"],
            "variants": 2,
            "params": [{"name": "density", "value": 0.4}],
        },
    ],
    "transitions": [{"lower": "water", "upper": "grass", "edge_style": "shore", "roughness": 0.5}],
    "objects": [
        {
            "name": "oak",
            "generator": "tree",
            "colors": [],
            "width": 1,
            "height": 1,
            "params": [{"name": "clumps", "value": 4}],
            "place_on": ["grass"],
            "density": 0.5,
        }
    ],
    "animations": [
        {
            "name": "waves",
            "terrain": "water",
            "generator": None,
            "colors": [],
            "frames": 4,
            "frame_ms": 200,
            "params": [],
        }
    ],
}


def bad_spec() -> dict:
    spec = {**GOOD, "terrains": [dict(t) for t in GOOD["terrains"]]}
    spec["terrains"][1]["generator"] = "lawn"
    return spec


class FakeBlock:
    def __init__(self, data: dict):
        self.data = data

    def to_dict(self):
        return {"type": "text", "text": "{...}", "parsed_output": self.data}


class FakeMessages:
    def __init__(self, outputs, stop_reason="end_turn"):
        self.outputs = list(outputs)
        self.stop_reason = stop_reason
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        data = self.outputs.pop(0)
        return SimpleNamespace(
            stop_reason=self.stop_reason, parsed_output=WireSpec.model_validate(data), content=[FakeBlock(data)]
        )


def fake_client(*outputs, stop_reason="end_turn"):
    return SimpleNamespace(messages=FakeMessages(outputs, stop_reason))


def test_plan_returns_valid_spec():
    client = fake_client(GOOD)
    spec = plan("a sunny meadow", seed=9, tile_size=32, client=client)
    assert spec.seed == 9 and spec.tile_size == 32
    assert spec.terrains[1].params == {"density": 0.4}
    assert spec.objects[0].params == {"clumps": 4}
    call = client.messages.calls[0]
    assert call["model"] == MODEL == "claude-haiku-5-5"
    assert call["output_format"] is WireSpec


def test_plan_retries_once_with_errors():
    client = fake_client(bad_spec(), GOOD)
    spec = plan("meadow", client=client)
    assert spec.name == "meadow"
    calls = client.messages.calls
    assert len(calls) == 2
    retry_msgs = calls[1]["messages"]
    assert retry_msgs[1]["role"] == "assistant"
    assert "parsed_output" not in retry_msgs[1]["content"][0]
    assert "lawn" in retry_msgs[2]["content"]


def test_plan_gives_up_after_retry():
    client = fake_client(bad_spec(), bad_spec())
    with pytest.raises(PlannerError, match="lawn"):
        plan("meadow", client=client)
    assert len(client.messages.calls) == 2


def test_plan_handles_refusal():
    with pytest.raises(PlannerError, match="declined"):
        plan("meadow", client=fake_client(GOOD, stop_reason="refusal"))


def test_system_prompt_lists_generators():
    prompt = system_prompt()
    for name in ("grass", "water", "tree", "house", "torch"):
        assert f"- {name}" in prompt
