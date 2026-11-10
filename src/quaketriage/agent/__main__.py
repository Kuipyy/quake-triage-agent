"""CLI: ``uv run python -m quaketriage.agent "question"`` (or ``-i`` for a conversation).

Uses the provider/model from ``.env`` (Gemini, Groq fallback on rate limits). Prints the answer and
a one-line footer (tools, LLM calls, provider); never prints keys or environment values.
"""
from __future__ import annotations

import argparse
import sys

from quaketriage.agent.graph import Agent, AgentResult
from quaketriage.agent.llm import LLMError, make_llm


def _footer(r: AgentResult) -> str:
    m = r.metrics
    prov = ",".join(m.get("providers", [])) or ("cache" if m.get("cache_hits") else "none")
    return (f"[{r.intent} | {r.language} | region={r.region} | tools: "
            f"{', '.join(r.tools_called) or '-'} | LLM calls {m['llm_calls']} "
            f"(cache hits {m['cache_hits']}, provider {prov}) | {m['latency_s']:.1f}s]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m quaketriage.agent", description=__doc__)
    ap.add_argument("question", nargs="*", help="question (omit with -i)")
    ap.add_argument("-i", "--interactive", action="store_true", help="keep the conversation going")
    ap.add_argument("--trace", action="store_true", help="print the tool trace after each answer")
    args = ap.parse_args(argv)
    if not args.question and not args.interactive:
        ap.error("give a question or use -i")
    try:
        agent = Agent(make_llm())
    except LLMError as exc:
        print(f"LLM not configured: {exc}", file=sys.stderr)
        return 2

    def turn(text: str) -> None:
        r = agent.ask(text)
        print(r.answer)
        print("\n" + _footer(r))
        if args.trace:
            for e in agent.ctx.trace[-len(r.tools_called):] if r.tools_called else []:
                print(f"  - {e['tool']} {e.get('status', 'error')} {e.get('seconds', '')}s")

    try:
        if args.question:
            turn(" ".join(args.question))
        if args.interactive:
            while (line := input("\n> ").strip()) not in ("", "exit", "quit"):
                turn(line)
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
