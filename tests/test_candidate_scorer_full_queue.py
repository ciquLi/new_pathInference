"""Scope extension and timing must not change the frozen pilot method."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def launcher():
    path = Path(__file__).resolve().parents[1] / "research/candidate_scorer_v1/full_february.py"
    spec = importlib.util.spec_from_file_location("full_february_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extended_prepare_keeps_config_and_opens_only_registered_sources():
    module = launcher()
    original = module.experiment.config()
    expanded = module.expanded_prepare()
    assert expanded.__globals__["cfg"] if "cfg" in expanded.__globals__ else True
    assert "32590" in expanded.__globals__["FULL_SOURCES"]
    assert len(expanded.__globals__["FULL_SOURCES"]) == 10
    assert module.experiment.config() == original
    assert original["sources"] == ["19529", "328840", "16552"]


def test_training_clock_preserves_return_and_parameters(monkeypatch):
    module = launcher()
    passed = []
    fake = SimpleNamespace(train=lambda *args, **kwargs: passed.append((args, kwargs)) or "fitted")
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    original = module.experiment.training_matrix
    timing = {}
    try:
        module.instrument_training(timing)
        assert fake.train("unchanged-data", num_boost_round=123) == "fitted"
        assert passed == [(("unchanged-data",), {"num_boost_round": 123})]
        assert timing["lightgbm_fit_wall_seconds"] >= 0
        assert timing["lightgbm_fit_cpu_seconds"] >= 0
    finally:
        module.experiment.training_matrix = original
