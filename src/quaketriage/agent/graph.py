"""LangGraph agent: router -> constraints -> step planner -> quality gate -> step loop -> compose.

* The router (LLM, with a keyword fallback) only picks intent, language, region and what-if
  parameters; its output is validated with pydantic. It never selects buildings.
* Guard rules live in code: a failed image-quality check stops the run before any decision step
  (``quality_gate``), the tools themselves refuse to rank/plan on bad imagery and hold back
  high-uncertainty buildings, and ``run_step`` stops on any ``verification_required`` result.
* What-if: ``update_constraints`` marks steps stale; only stale steps that had run are re-run.
* Every turn records LLM calls, cache hits and latency (Week-7 metrics) in ``run_log``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from quaketriage.agent import prompts
from quaketriage.agent.llm import LLMError
from quaketriage.agent.tools import (
    PIPELINE,
    ToolContext,
    ToolError,
    UpdateConstraintsInput,
    check_support,
)
from quaketriage.config import get

ROOT = Path(__file__).parents[3]
Intent = Literal["full_assessment", "prioritize", "route_plan", "what_if", "status_report",
                 "image_quality", "protocol_question", "off_topic"]
DECISION_STEPS = {"prioritize_sites", "plan_routes", "write_sitrep"}
CHAINS = {  # steps needed to reach a goal, in order (already-done steps are skipped)
    "prioritize": ["detect_damage", "score_buildings", "estimate_occupancy", "prioritize_sites"],
    "route_plan": ["detect_damage", "score_buildings", "estimate_occupancy", "prioritize_sites",
                   "plan_routes"],
    "full_assessment": ["detect_damage", "score_buildings", "estimate_occupancy",
                        "prioritize_sites", "plan_routes", "lookup_protocol", "write_sitrep"],
    "status_report": ["write_sitrep"],
    "image_quality": ["check_image_quality"],
    "protocol_question": ["lookup_protocol"],
}


class LLMLike(Protocol):
    calls: int
    cache_hits: int

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class RouterDecision(BaseModel):
    intent: Intent
    language: Literal["tr", "en"]
    region: Literal["antakya", "kilis"] | None = None
    constraint_changes: dict[str, Any] = {}
    protocol_query: str | None = None


class AgentState(TypedDict, total=False):
    message: str
    language: str
    region: str
    intent: str
    constraint_changes: dict
    protocol_query: str | None
    pending: list[str]
    rerun: list[str]
    results: dict[str, dict]
    constraints: dict
    trace: list[dict]
    stop_reason: str | None
    notes: list[str]
    answer: str
    router_fallback: bool


@dataclass
class AgentResult:
    answer: str
    intent: str
    language: str
    region: str
    stop_reason: str | None
    tools_called: list[str]
    results: dict[str, dict]
    metrics: dict = field(default_factory=dict)


# ------------------------------------------------------------------ heuristics (router fallback)
_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TR_WORDS = {"ve", "bir", "için", "kaç", "ekip", "bina", "rota", "hasar", "nedir", "mı", "mi", "ne",
             "nasıl", "yap", "göster", "bana", "saat", "kapalı", "öncelik", "rapor", "değerlendir"}
_DOMAIN = ("bina", "building", "hasar", "damage", "rota", "route", "ekip", "team", "öncelik",
           "priorit", "rapor", "report", "sitrep", "deprem", "earthquake", "usar", "insarag",
           "afad", "tamp", "protokol", "protocol", "görüntü", "image", "kalite", "quality",
           "kurtarma", "rescue", "yol", "road", "saat", "hour")


def detect_language(text: str) -> str:
    words = set(re.findall(r"\w+", text.lower()))
    if _TR_CHARS & set(text) or len(words & _TR_WORDS) >= 2:
        return "tr"
    return "en"


def heuristic_route(message: str) -> RouterDecision:
    """Keyword router used when the LLM is unavailable or answers with invalid JSON."""
    m = message.lower()
    lang = detect_language(message)
    changes: dict[str, Any] = {}
    if n := re.search(r"(\d+)\s*(?:ekip|team)", m):
        changes["num_teams"] = int(n.group(1))
    if h := re.search(r"(\d+(?:[.,]\d+)?)\s*(?:saat|hours?|hrs?)\b", m):
        changes["mission_hours"] = float(h.group(1).replace(",", "."))
    if not any(k in m for k in _DOMAIN):
        intent: Intent = "off_topic"
    elif changes:
        intent = "what_if"
    elif any(k in m for k in ("protokol", "protocol", "insarag", "tamp", "nasıl", "how do",
                              "what does")) and not any(k in m for k in ("rota", "route")):
        intent = "protocol_question"
    elif any(k in m for k in ("kalite", "quality", "bulut", "cloud")):
        intent = "image_quality"
    elif any(k in m for k in ("rota", "route", "plan")):
        intent = "route_plan"
    elif any(k in m for k in ("öncelik", "priorit", "sırala", "rank")):
        intent = "prioritize"
    elif any(k in m for k in ("rapor", "report", "sitrep")):
        intent = "status_report"
    else:
        intent = "full_assessment"
    region = "kilis" if "kilis" in m else ("antakya" if "antakya" in m else None)
    return RouterDecision(intent=intent, language=lang, region=region, constraint_changes=changes,
                          protocol_query=message if intent == "protocol_question" else None)


def parse_router_json(text: str) -> RouterDecision:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in router reply")
    return RouterDecision(**json.loads(text[a:b + 1]))


# ------------------------------------------------------------------ messages
_MSG = {
    "tr": {
        "off_topic": "Bu soru deprem hasar değerlendirmesi ve kurtarma planlamasıyla ilgili değil; "
                     "yardımcı olamıyorum. Hasar değerlendirmesi, öncelik, rota, rapor veya AFAD/"
                     "INSARAG protokolleri hakkında sorabilirsiniz.",
        "verification": "⚠ Karar verilmedi: saha doğrulaması gerekli.",
        "tool_error": "Bir adım çalışmadı",
        "updated": "Kısıtlar güncellendi",
        "rerun": "yeniden çalışan adımlar", "kept": "değişmeyen adımlar",
        "nothing": "Henüz çalışmış bir adım yok; değerlendirmeyi başlatın.",
        "no_hits": "Kılavuzlarda bu soruya yanıt veren bir bölüm bulunamadı.",
        "sources": "Kaynaklar", "simulated": "SİMÜLE DEDEKTÖR — demo; operasyonel karar için değil",
        "kilis": "Kilis kesiti bir boru hattı demosudur: gerçek Siamese modeli çalıştı, ancak bu "
                 "kesit için bina tablosu yok; öncelik ve rota Antakya demosunda üretilir.",
        "top": "Öncelikli binalar", "plan": "Rota planı", "held": "doğrulama bekleyen bina",
    },
    "en": {
        "off_topic": "That question is not about earthquake damage assessment or rescue planning, "
                     "so I can't help with it. You can ask about damage assessment, priorities, "
                     "routes, the situation report or AFAD/INSARAG protocols.",
        "verification": "⚠ No decision taken: field verification is required.",
        "tool_error": "A step failed",
        "updated": "Constraints updated",
        "rerun": "steps re-run", "kept": "steps unchanged",
        "nothing": "No step has run yet; start an assessment first.",
        "no_hits": "No passage in the guidelines answers this question.",
        "sources": "Sources",
        "simulated": "SIMULATED detector — demo, not for operational decisions",
        "kilis": "The Kilis crop is a pipeline demo: the real Siamese model ran, but there is no "
                 "building table for this crop; priorities and routes are produced for the "
                 "Antakya demo.",
        "top": "Priority buildings", "plan": "Route plan",
        "held": "buildings awaiting verification",
    },
}


# ------------------------------------------------------------------ the agent
class Agent:
    """Holds the persistent ``ToolContext`` (conversation state) and the compiled graph."""

    def __init__(self, llm: LLMLike, ctx: ToolContext | None = None, region: str = "antakya",
                 run_log: str | Path | None = None, regions: dict | None = None):
        self.llm, self.ctx, self.region = llm, ctx or ToolContext(), region
        self.regions = regions if regions is not None else get("regions")
        self.run_log = Path(run_log) if run_log else ROOT / get("agent.run_log")
        self.history: list[dict] = []
        self.last_support: list[dict] = []
        self._node_seconds: dict[str, float] = {}
        self.graph = self._build()

    # -- helpers
    def _timed(self, name: str, fn):
        def wrapper(state):
            t = time.time()
            out = fn(state)
            self._node_seconds[name] = round(self._node_seconds.get(name, 0.0) + time.time() - t, 3)
            return out
        return wrapper

    def _reset_region(self, region: str) -> None:
        old = self.ctx
        self.ctx = ToolContext(constraints=old.constraints, base_graph=old.base_graph,
                               rag_index=old.rag_index, embed_query=old.embed_query,
                               siamese_runner=old.siamese_runner, lang=old.lang)
        self.region = region

    # -- nodes
    def _router(self, state: AgentState) -> dict:
        ctx, notes = self.ctx, list(state.get("notes", []))
        prompt = prompts.router_prompt(state["message"], self.region, sorted(ctx.done),
                                       ctx.constraints.model_dump(mode="json"))
        decision, fallback = None, False
        try:
            for attempt in range(2):
                reply = self.llm.complete(prompt if attempt == 0 else
                                          prompt + "\nYour last reply was not valid JSON for the "
                                          "schema. Reply with the JSON object only.",
                                          system=prompts.router_system())
                try:
                    decision = parse_router_json(reply)
                    break
                except (ValueError, ValidationError):
                    continue
        except LLMError as exc:
            notes.append(f"router LLM unavailable ({exc}); keyword router used")
        if decision is None:
            decision, fallback = heuristic_route(state["message"]), True
        lang = decision.language
        if fallback is False and lang != detect_language(state["message"]) and (
                _TR_CHARS & set(state["message"])):
            lang = "tr"  # Turkish letters are unambiguous; correct a wrong model guess
        changes = {}
        for k, v in decision.constraint_changes.items():
            changes[k] = v
        try:
            changes = UpdateConstraintsInput(**changes).model_dump(exclude_none=True)
        except ValidationError as exc:
            notes.append("ignored invalid constraint changes: " + "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()))
            changes = {}
        region = decision.region or self.region
        if region not in self.regions:
            notes.append(f"unknown region {region!r}; keeping {self.region}")
            region = self.region
        if region != self.region:
            self._reset_region(region)
        self.ctx.lang = lang  # user-facing warnings (also the guard's) follow the user's language
        return {"intent": decision.intent, "language": lang, "region": region,
                "constraint_changes": changes, "protocol_query": decision.protocol_query,
                "notes": notes, "router_fallback": fallback, "results": {}, "stop_reason": None,
                "rerun": [], "pending": []}

    def _apply_constraints(self, state: AgentState) -> dict:
        changes = state.get("constraint_changes") or {}
        if not changes:
            return {}
        before = set(self.ctx.done)
        notes = list(state.get("notes", []))
        try:
            out = self.ctx.call("update_constraints", changes)
        except ToolError as exc:
            notes.append(str(exc))
            return {"notes": notes, "stop_reason": "tool_error"}
        results = {**state.get("results", {}), "update_constraints": out}
        return {"results": results, "rerun": [s for s in out["stale_steps"] if s in before],
                "notes": notes}

    def _plan_steps(self, state: AgentState) -> dict:
        if state.get("stop_reason"):
            return {}
        intent, ctx = state["intent"], self.ctx
        spec = self.regions[state["region"]]
        if spec["detector"] == "siamese":  # Kilis: image pipeline only, no building table
            steps = ["check_image_quality", "detect_damage"]
            steps = [s for s in steps if s not in ctx.done] if intent != "what_if" else []
            if intent in ("protocol_question",):
                steps = ["lookup_protocol"]
            return {"pending": steps}
        if intent == "what_if":
            pending = [s for s in PIPELINE if s in state.get("rerun", [])]
        elif intent == "status_report" and "detect_damage" not in ctx.done:
            pending = [s for s in CHAINS["full_assessment"] if s not in ctx.done]
        elif intent == "image_quality":
            pending = []  # Antakya has no imagery in this demo; answered in compose
        elif intent == "off_topic":
            pending = []
        else:
            chain = CHAINS[intent]
            pending = [s for s in chain if s not in ctx.done or s == "lookup_protocol"
                       and intent == "protocol_question"]
            if intent == "full_assessment" and ctx.protocol_hits:
                pending = [s for s in pending if s != "lookup_protocol"]
        return {"pending": pending}

    def _quality_gate(self, state: AgentState) -> dict:
        """Code-level guard: never reach a decision step with a failed image check."""
        if state.get("stop_reason"):
            return {}
        ctx = self.ctx
        spec = self.regions[state["region"]]
        pending = state.get("pending", [])
        if spec["detector"] == "siamese" and "check_image_quality" in pending:
            try:
                out = ctx.call("check_image_quality", {"pre_path": ROOT / spec["pre"],
                                                       "post_path": ROOT / spec["post"]})
            except ToolError as exc:
                return {"stop_reason": "tool_error",
                        "notes": [*state.get("notes", []), str(exc)], "pending": []}
            results = {**state.get("results", {}), "check_image_quality": out}
            pending = [s for s in pending if s != "check_image_quality"]
            if out["status"] != "ok":
                return {"results": results, "stop_reason": "verification", "pending": []}
            return {"results": results, "pending": pending}
        if ctx.quality_block() and (DECISION_STEPS & set(pending) or
                                    state["intent"] in ("full_assessment", "prioritize",
                                                        "route_plan", "status_report", "what_if")):
            return {"stop_reason": "verification", "pending": []}
        return {}

    def _step_args(self, step: str, state: AgentState) -> dict:
        spec = self.regions[state["region"]]
        if step == "detect_damage" and spec["detector"] == "siamese":
            return {"source": "siamese", "pre_path": str(ROOT / spec["pre"]),
                    "post_path": str(ROOT / spec["post"])}
        if step == "lookup_protocol":
            return {"query": state.get("protocol_query") or get("agent.default_protocol_query")}
        if step == "write_sitrep":
            return {"language": state["language"]}
        return {}

    def _run_step(self, state: AgentState) -> dict:
        pending = list(state.get("pending", []))
        if not pending:
            return {}
        step, rest = pending[0], pending[1:]
        results, notes = dict(state.get("results", {})), list(state.get("notes", []))
        try:
            out = self.ctx.call(step, self._step_args(step, state))
        except ToolError as exc:
            if step == "lookup_protocol" and state["intent"] != "protocol_question":
                notes.append(f"protocol lookup skipped: {exc}")  # report without citations
                return {"pending": rest, "notes": notes}
            notes.append(f"{step}: {exc}")
            return {"pending": [], "stop_reason": "tool_error", "notes": notes}
        results[step] = out
        if out["status"] != "ok":
            return {"results": results, "pending": [], "stop_reason": "verification"}
        return {"results": results, "pending": rest}

    # -- compose
    def _compose(self, state: AgentState) -> dict:
        lang = state.get("language", "en")
        t = _MSG[lang]
        ctx, results, intent = self.ctx, state.get("results", {}), state.get("intent")
        parts: list[str] = []
        if intent == "off_topic":
            return {"answer": t["off_topic"]}
        stop = state.get("stop_reason")
        if "update_constraints" in results:
            u = results["update_constraints"]
            parts.append(f"{t['updated']}: {u['changed'] or '—'}; {t['rerun']}: "
                         f"{[s for s in state.get('rerun', [])] or '—'}; {t['kept']}: "
                         f"{[s for s in u['unchanged_steps'] if s in ctx.done] or '—'}.")
        if stop == "verification":
            reasons = [ctx.quality_block()] if ctx.quality_block() else []
            for r in results.values():
                reasons += r.get("warnings", [])
            bullets = "\n".join(f"- {x}" for x in dict.fromkeys(reasons))
            parts.append(t["verification"] + "\n" + bullets)
        elif stop == "tool_error":
            parts.append(f"{t['tool_error']}: " + "; ".join(state.get("notes", [])[-2:]))
        elif intent == "protocol_question":
            parts.append(self._protocol_answer(state, results.get("lookup_protocol"), lang))
        elif state["region"] and self.regions[state["region"]]["detector"] == "siamese":
            d = results.get("detect_damage")
            if d:
                unc = d["uncertainty"]["mean"] if d["uncertainty"] else "n/a"
                parts.append(t["kilis"] + f"\n- mean damage probability {d['mean_prob']}, "
                             f"uncertainty (TTA std) {unc}"
                             f", tiles {d['detail'].get('n_tiles')}, rejected "
                             f"{d['detail'].get('n_rejected_tiles')}.")
        elif intent == "image_quality":
            q = results.get("check_image_quality")
            parts.append(("Görüntü kalitesi: " if lang == "tr" else "Image quality: ")
                         + (f"ok={q['ok']}, reasons={q['reasons']}" if q else
                            ("Antakya demosunda görüntü yok (simüle dedektör)." if lang == "tr"
                             else "no imagery in the Antakya demo (simulated detector).")))
        elif "write_sitrep" in results:
            parts.append(results["write_sitrep"]["markdown"])
        else:
            parts.append(self._summary(results, lang) or t["nothing"])
        for n in state.get("notes", []):
            if n.startswith(("ignored", "unknown region")):
                parts.append(f"({n})")
        if ctx.simulated and "write_sitrep" not in results:
            parts.append(f"_{t['simulated']}_")
        return {"answer": "\n\n".join(parts)}

    def _summary(self, results: dict, lang: str) -> str:
        t = _MSG[lang]
        lines = []
        pr = results.get("prioritize_sites")
        if pr and pr["top"]:
            lines.append(f"**{t['top']}**")
            for i, r in enumerate(pr["top"][:5], 1):
                bid = r["id"] if r["id"] is not None else r["row"]
                lines.append(f"{i}. #{bid} — {r['priority']:.3f} ({r['damage_prob']:.2f}, "
                             f"~{r['estimated_persons']:.0f})")
            if pr["n_verification_queue"]:
                lines.append(f"- {pr['n_verification_queue']} {t['held']}")
        pl = results.get("plan_routes")
        if pl and pl["teams"]:
            lines.append(f"**{t['plan']}**: {pl['n_visited']} / {len(pl['teams'])} teams, "
                         f"priority {pl['priority_collected']:.2f}, ~{pl['persons_reached']:.0f}")
        return "\n".join(lines)

    def _protocol_answer(self, state: AgentState, found: dict | None, lang: str) -> str:
        t = _MSG[lang]
        hits = (found or {}).get("hits", [])
        if not hits:
            return t["no_hits"]
        try:
            text = self.llm.complete(prompts.protocol_prompt(state["message"], hits),
                                     system=prompts.protocol_system(lang)).strip()
        except LLMError:
            text = " ".join(h["text"][:200] for h in hits[:2])
        refs = {int(r) for r in re.findall(r"\[(\d+)\]", text)}
        valid = {h["ref"] for h in hits}
        for bad in refs - valid:  # a citation to a passage that was never retrieved is removed
            text = text.replace(f"[{bad}]", "")
        self.last_support = check_support(text, hits)
        used = sorted(refs & valid) or [h["ref"] for h in hits[:2]]
        src = "; ".join(f"[{h['ref']}] {h['doc']}, p. {h['pages']}" for h in hits
                        if h["ref"] in used)
        return f"{text}\n\n{t['sources']}: {src}"

    # -- graph
    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in (("router", self._router), ("apply_constraints", self._apply_constraints),
                         ("plan_steps", self._plan_steps), ("quality_gate", self._quality_gate),
                         ("run_step", self._run_step), ("compose", self._compose)):
            g.add_node(name, self._timed(name, fn))
        g.add_edge(START, "router")
        g.add_conditional_edges("router", lambda s: "compose" if s["intent"] == "off_topic"
                                else "apply_constraints", ["compose", "apply_constraints"])
        g.add_edge("apply_constraints", "plan_steps")
        g.add_edge("plan_steps", "quality_gate")
        g.add_conditional_edges("quality_gate", self._after_gate, ["run_step", "compose"])
        g.add_conditional_edges("run_step", self._after_step, ["run_step", "compose"])
        g.add_edge("compose", END)
        return g.compile()

    @staticmethod
    def _after_gate(state: AgentState) -> str:
        return "compose" if state.get("stop_reason") or not state.get("pending") else "run_step"

    @staticmethod
    def _after_step(state: AgentState) -> str:
        return "compose" if state.get("stop_reason") or not state.get("pending") else "run_step"

    # -- public API
    def ask(self, message: str) -> AgentResult:
        t0, calls0, hits0 = time.time(), self.llm.calls, self.llm.cache_hits
        used0 = len(getattr(self.llm, "used", []))
        cached0 = len(getattr(self.llm, "cached_from", []))
        trace0 = len(self.ctx.trace)
        self._node_seconds = {}
        self.last_support = []
        state = self.graph.invoke({"message": message, "region": self.region, "notes": []},
                                  {"recursion_limit": 60})
        tools = [e["tool"] for e in self.ctx.trace[trace0:]]
        metrics = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "intent": state["intent"],
                   "language": state["language"], "region": state["region"],
                   "stop_reason": state.get("stop_reason"),
                   "router_fallback": state.get("router_fallback", False),
                   "llm_calls": self.llm.calls - calls0,
                   "cache_hits": self.llm.cache_hits - hits0,
                   "providers": list(getattr(self.llm, "used", []))[used0:],
                   "cached_providers": list(getattr(self.llm, "cached_from", []))[cached0:],
                   "latency_s": round(time.time() - t0, 3), "node_seconds": self._node_seconds,
                   "tools_called": tools,
                   "citation_support": {
                       "checked": len(self.last_support),
                       "supported": sum(x["supported"] for x in self.last_support)}}
        self.history.append(metrics)
        try:
            self.run_log.parent.mkdir(parents=True, exist_ok=True)
            with self.run_log.open("a") as fh:
                fh.write(json.dumps(metrics) + "\n")
        except OSError:
            pass  # metrics stay in memory; logging must never break a turn
        return AgentResult(answer=state["answer"], intent=state["intent"],
                           language=state["language"], region=state["region"],
                           stop_reason=state.get("stop_reason"), tools_called=tools,
                           results=state.get("results", {}), metrics=metrics)

    def state_snapshot(self) -> dict:
        """Region, constraints, completed steps and tool trace (what the UI shows)."""
        return {"region": self.region, "constraints": self.ctx.constraints.model_dump(mode="json"),
                "done": sorted(self.ctx.done), "trace": list(self.ctx.trace)}
