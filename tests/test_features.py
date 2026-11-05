import numpy as np
import pytest

from dce_mri.features import compute_kinetic_channels, joint_percentile_scale


def test_joint_scaling_preserves_enhancement_above_precontrast_range():
    pre = np.linspace(0, 100, 1_000, dtype=np.float32).reshape(10, 10, 10)
    post = pre + 50
    pre_scaled, posts_scaled = joint_percentile_scale(pre, [post])

    assert pre_scaled.min() == pytest.approx(0.0)
    assert pre_scaled.max() == pytest.approx(1.0)
    assert posts_scaled[0].max() > 1.0


def test_channels_have_expected_shape_dtype_and_finite_values():
    pre = np.zeros((4, 5, 6), dtype=np.float32)
    channels = compute_kinetic_channels(pre, [pre + 1, pre + 0.5])

    assert channels.shape == (4, 4, 5, 6)
    assert channels.dtype == np.float32
    assert np.isfinite(channels).all()


def test_positive_washout_is_peak_minus_last_phase():
    pre = np.linspace(0, 100, 27, dtype=np.float32).reshape(3, 3, 3)
    early = pre + 20.0
    late = pre + 10.0
    channels = compute_kinetic_channels(pre, [early, late])

    assert (channels[3] > 0).all()


def test_missing_postcontrast_phases_produce_zero_kinetic_channels():
    pre = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    channels = compute_kinetic_channels(pre, [])

    np.testing.assert_array_equal(channels[1:], 0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same shape"):
        joint_percentile_scale(np.zeros((2, 2, 2)), [np.zeros((3, 2, 2))])
