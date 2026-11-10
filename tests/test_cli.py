import pytest

from quaketriage.agent import __main__ as cli
from quaketriage.agent.graph import Agent
from tests.test_graph import FakeLLM, _route


def _patch(monkeypatch, tmp_path, llm):
    monkeypatch.setattr(cli, "make_llm", lambda: llm)
    monkeypatch.setattr(cli, "Agent", lambda m: Agent(m, run_log=tmp_path / "runs.jsonl"))


def test_cli_answers_and_prints_footer_without_secrets(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-secret-value-123")
    _patch(monkeypatch, tmp_path, FakeLLM(_route("off_topic", "tr")))
    assert cli.main(["Bana", "bir", "kek", "tarifi", "ver"]) == 0
    out = capsys.readouterr().out
    assert "yardımcı olamıyorum" in out and "off_topic" in out and "LLM calls 1" in out
    assert "sk-test-secret-value-123" not in out


def test_cli_requires_a_question_or_interactive_flag():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_reports_llm_configuration_error(monkeypatch, capsys):
    from quaketriage.agent.llm import LLMError

    def boom():
        raise LLMError("no LLM backend configured")
    monkeypatch.setattr(cli, "make_llm", boom)
    assert cli.main(["hello"]) == 2
    assert "LLM not configured" in capsys.readouterr().err
