from pathlib import Path

from runtime.config import load_runtime_config


def test_all_four_modes_have_explicit_fp32_performance():
    cfg = load_runtime_config(Path(__file__).resolve().parents[1] / "adaptive_runtime_config.yaml")
    expected = {"metadata_only", "action_history_only", "shock_metadata", "shock_action_history"}
    assert expected.issubset(cfg["models"])
    for mode in expected:
        p = cfg["models"][mode]["performance"]
        assert p["estimator_amp"] is False
        assert p["transition_amp"] is False
        assert 1 <= int(p["candidate_chunk_size"]) <= 27
