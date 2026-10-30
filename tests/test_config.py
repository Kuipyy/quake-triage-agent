"""Config loader tests."""
import pytest

from quaketriage.config import ConfigError, get, load_config


def _write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_default_config_loads():
    cfg = load_config()
    assert cfg["geo"]["crs"].startswith("EPSG:")
    assert get("topsis.weights.damage_prob") > 0


def test_env_placeholder_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_TEST_PROVIDER", "groq")
    path = _write(
        tmp_path, 'llm:\n  provider: ${QT_TEST_PROVIDER}\n  names: ["a-${QT_TEST_PROVIDER}"]\n'
    )
    cfg = load_config(path)
    assert cfg["llm"]["provider"] == "groq"
    assert cfg["llm"]["names"] == ["a-groq"]


def test_default_used_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_TEST_UNSET", raising=False)
    path = _write(tmp_path, "x: ${QT_TEST_UNSET:-fallback}\n")
    assert load_config(path)["x"] == "fallback"


def test_unset_is_empty_unless_strict(tmp_path, monkeypatch):
    monkeypatch.delenv("QT_TEST_UNSET", raising=False)
    path = _write(tmp_path, "x: ${QT_TEST_UNSET}\n")
    assert load_config(path)["x"] == ""
    with pytest.raises(ConfigError):
        load_config(path, strict=True)


def test_returned_dict_is_a_copy(tmp_path):
    path = _write(tmp_path, "a:\n  b: 1\n")
    load_config(path)["a"]["b"] = 99
    assert load_config(path)["a"]["b"] == 1


def test_get_missing_key(tmp_path):
    path = _write(tmp_path, "a:\n  b: 1\n")
    assert get("a.b", path=path) == 1
    assert get("a.c", default=7, path=path) == 7
    with pytest.raises(ConfigError):
        get("a.c", path=path)
