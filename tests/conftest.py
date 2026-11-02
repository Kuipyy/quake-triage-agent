"""Shared pytest config.

CI installs only the core dependencies (no PyTorch, to stay fast). Test modules that need
torch / segmentation-models-pytorch are skipped at collection time when torch is missing;
they always run locally (`uv sync --all-extras`).
"""
import importlib.util

TORCH_TESTS = ["test_dataset.py", "test_infer.py", "test_siamese.py", "test_uncertainty.py"]

collect_ignore = [] if importlib.util.find_spec("torch") else list(TORCH_TESTS)


def pytest_report_header(config):
    if collect_ignore:
        return f"torch not installed — skipping: {', '.join(collect_ignore)}"
    return None
