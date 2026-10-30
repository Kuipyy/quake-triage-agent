"""Provenance helper tests."""
import json

from quaketriage.provenance import config_hash, file_sha256, provenance, write_json


def test_file_sha256_matches_known_value(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_provenance_fields(tmp_path):
    f = tmp_path / "in.bin"
    f.write_bytes(b"x")
    info = provenance(cfg={"a": 1}, inputs=[f], seed=42)
    assert info["seed"] == 42
    assert len(info["config_hash"]) == 12
    assert info["inputs_sha256"][str(f)] == file_sha256(f)
    assert info["created_utc"].endswith("+00:00")
    assert "git_sha" in info and "git_dirty" in info


def test_write_json_adds_provenance(tmp_path):
    out = write_json(tmp_path / "sub" / "m.json", {"f1": 0.5}, info={"seed": 1})
    data = json.loads(out.read_text())
    assert data == {"f1": 0.5, "provenance": {"seed": 1}}
