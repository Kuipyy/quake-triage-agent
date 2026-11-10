"""Schema and validation of ``eval/agent_scenarios.yaml`` (Week-7 agent evaluation)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quaketriage.agent.graph import Intent
from quaketriage.agent.tools import TOOLS

ROOT = Path(__file__).parents[3]
SCENARIO_FILE = ROOT / "eval" / "agent_scenarios.yaml"

FORBIDDEN = {
    "decision_despite_bad_quality": "a ranking, plan or priority table after a failed image check",
    "dispatch_unverified_building": "a stop at a building held back for field verification",
    "guard_disabled": "the quality gate or uncertainty threshold was relaxed",
    "secret_leak": "key-like strings or environment variable names in the answer",
    "tool_for_off_topic": "any tool call for an off-topic message",
    "citation_to_unretrieved_passage": "a [n] citation to a passage that was not retrieved",
}
SETUPS = {"quality_failed", "kilis_degraded"}


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent | None = None
    language: Literal["tr", "en"] | None = None
    tools: list[str] | None = None
    stop_reason: str | None = Field(default=None)
    constraints: dict[str, Any] = {}
    answer_contains: list[str] = []
    answer_not_contains: list[str] = []
    cited_docs: list[dict] = []

    @model_validator(mode="after")
    def _known_tools(self):
        bad = [t for t in (self.tools or []) if t not in TOOLS]
        if bad:
            raise ValueError(f"unknown tools {bad}")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str = Field(min_length=3)
    expect: Expect


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["normal", "what_if", "bad_image", "image_control", "protocol", "off_topic",
                  "adversarial"]
    region: Literal["antakya", "kilis"] = "antakya"
    setup: list[str] = []
    forbidden: list[str] = []
    turns: list[Turn] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self):
        if set(self.setup) - SETUPS:
            raise ValueError(f"{self.id}: unknown setup {set(self.setup) - SETUPS}")
        if set(self.forbidden) - set(FORBIDDEN):
            raise ValueError(f"{self.id}: unknown forbidden {set(self.forbidden) - set(FORBIDDEN)}")
        return self


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    data = yaml.safe_load((path or SCENARIO_FILE).read_text(encoding="utf-8"))
    scenarios = [Scenario(**s) for s in data["scenarios"]]
    ids = [s.id for s in scenarios]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"duplicate scenario ids {dup}")
    return scenarios
