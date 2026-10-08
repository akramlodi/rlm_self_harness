import pytest

from shrlm.experiment.calibrate_promotion_tau import upper_quantile


def test_upper_quantile_uses_conservative_nearest_rank() -> None:
    assert upper_quantile([0.01, 0.02, 0.03], 0.95) == 0.03
    assert upper_quantile([0.01, 0.02, 0.03, 0.04], 0.50) == 0.02


def test_upper_quantile_rejects_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="no deltas"):
        upper_quantile([], 0.95)
    with pytest.raises(ValueError, match="quantile"):
        upper_quantile([0.1], 0.0)
