from types import SimpleNamespace

import pytest

from tileset_gen.planner import MODEL, PlannerError, WireSpec, plan, system_prompt

GOOD = {
    "name": "meadow",
    "theme": "a sunny meadow",
    "palette": [{"name": "grass", "hex": "#5aa83c"}, {"name": "water", "hex": "#3a78c8"}],
    "terrains": [
        {
            "name": "water",
            "generator": "water",
            "colors": ["water"],
            "variants": 1,
            "decor": 2,
            "patches": 0.3,
            "params": [],
        },
        {
            "name": "grass",
            "generator": "grass",
            "colors": ["grass"],
            "variants": 2,
            "decor": 3,
            "patches": 0.3,
            "params": ["density=0.4"],
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
            "params": ["clumps=4"],
            "place_on": ["grass"],
            "density": 0.5,
        }
    ],
    "animations": [
        {
            "name": "waves",
            "terrain": "water",
            "generator": "",
            "colors": [],
            "frames": 4,
            "frame_ms": 200,
            "params": [],
        }
    ],
    "characters": [
        {
            "name": "hero",
            "generator": "humanoid",
            "colors": ["#f0c49a", "#6a4028", "water"],
            "params": ["hair=spiky", "cape=true"],
            "place_on": ["grass"],
            "count": 1,
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
    assert spec.characters[0].params == {"hair": "spiky", "cape": True}
    call = client.messages.calls[0]
    assert call["model"] == MODEL == "claude-haiku-5-5"
    assert call["output_format"] is WireSpec


def test_plan_in_a_retro_style():
    client = fake_client(GOOD)
    spec = plan("a sunny meadow", style="nes", client=client)
    assert spec.style == "nes" and spec.max_colors == 16
    assert all(c.max_colors == 3 for c in spec.characters)
    assert "NES hardware palette" in client.messages.calls[0]["system"]
    assert "NES" not in system_prompt()


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
    for name in ("grass", "water", "tree", "house", "torch", "humanoid", "slime"):
        assert f"- {name}" in prompt


def test_refine_round_trips_and_keeps_build_settings():
    from tileset_gen.planner import refine, spec_diff
    from tileset_gen.spec import TilesetSpec

    old = plan("meadow", seed=5, tile_size=32, client=fake_client(GOOD))
    changed = {**GOOD, "objects": GOOD["objects"] + [{**GOOD["objects"][0], "name": "birch", "density": 0.7}]}
    changed["terrains"] = [GOOD["terrains"][0], {**GOOD["terrains"][1], "params": ["density=0.9"]}]
    client = fake_client(changed)
    new = refine(old, "add birches and lusher grass", client=client)
    assert (new.seed, new.tile_size, new.max_colors) == (5, 32, old.max_colors)
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "add birches" in sent and '"name": "oak"' in sent and '"seed"' not in sent
    diff = spec_diff(old, new)
    assert "+ object birch" in diff
    assert any(line.startswith("~ terrain grass") and "0.9" in line for line in diff)
    assert spec_diff(old, old) == []
    assert isinstance(new, TilesetSpec)


def test_wire_params_round_trip():
    from tileset_gen.planner import param_str, parse_params

    params = {"density": 0.4, "clumps": 3, "cape": True, "snow": False, "outfit": "robe"}
    assert parse_params([param_str(k, v) for k, v in params.items()]) == params
    assert parse_params(["tiers = 3.0", "style=stone"]) == {"tiers": 3, "style": "stone"}


def test_wire_schema_has_no_unions():
    # Unions (number|bool|string values, nullable fields) blow up the compiled output grammar, and the
    # API rejects grammars past its size limit.
    import json

    assert "anyOf" not in json.dumps(WireSpec.model_json_schema())


def test_refine_keeps_character_settings_the_wire_omits():
    from tileset_gen.planner import refine

    old = plan("meadow", client=fake_client(GOOD))
    old.characters[0].max_colors, old.characters[0].frame_ms = 3, 200
    new = refine(old, "nothing", client=fake_client(GOOD))
    assert (new.characters[0].max_colors, new.characters[0].frame_ms) == (3, 200)
