"""Numerically stable kinetic-channel construction for registered DCE-MRI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def joint_percentile_scale(
    pre: FloatArray,
    posts: Sequence[FloatArray],
    percentiles: tuple[float, float] = (0.1, 99.9),
    post_upper: float = 10.0,
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
    """Scale every phase with bounds estimated from the pre-contrast volume.

    Applying one transform to the temporal series preserves enhancement deltas.
    The pre-contrast image is clipped to [0, 1], while post-contrast images may
    exceed 1 to retain strongly enhancing tissue.
    """
    pre_array = np.asarray(pre, dtype=np.float32)
    post_arrays = [np.asarray(post, dtype=np.float32) for post in posts]
    if any(post.shape != pre_array.shape for post in post_arrays):
        raise ValueError("All registered DCE phases must have the same shape")

    low, high = np.percentile(pre_array, percentiles)
    scale = max(float(high - low), 1e-5)
    pre_scaled = np.clip((pre_array - low) / scale, 0.0, 1.0)
    posts_scaled = [
        np.clip((post - low) / scale, 0.0, post_upper) for post in post_arrays
    ]
    return pre_scaled.astype(np.float32), [p.astype(np.float32) for p in posts_scaled]


def compute_kinetic_channels(
    pre: FloatArray,
    posts: Sequence[FloatArray],
    percentiles: tuple[float, float] = (0.1, 99.9),
) -> NDArray[np.float32]:
    """Return baseline, relative enhancement, subtraction, and washout channels."""
    pre_scaled, posts_scaled = joint_percentile_scale(pre, posts, percentiles)
    zeros = np.zeros_like(pre_scaled, dtype=np.float32)

    if posts_scaled:
        first = posts_scaled[0]
        safe_pre = np.maximum(pre_scaled, 0.01)
        relative_enhancement = np.clip(
            (first - pre_scaled) / safe_pre, 0.0, 10.0
        ).astype(np.float32)
        subtraction = np.clip(first - pre_scaled, 0.0, 5.0).astype(np.float32)
    else:
        relative_enhancement = zeros.copy()
        subtraction = zeros.copy()

    if len(posts_scaled) > 1:
        temporal = np.stack(posts_scaled, axis=-1)
        peak = temporal.max(axis=-1)
        peak_index = temporal.argmax(axis=-1)
        last_index = len(posts_scaled) - 1
        elapsed = np.maximum(last_index - peak_index, 1)
        washout = np.clip((peak - posts_scaled[-1]) / elapsed, 0.0, 5.0)
        washout = washout.astype(np.float32)
    else:
        washout = zeros.copy()

    return np.stack(
        [pre_scaled, relative_enhancement, subtraction, washout], axis=0
    ).astype(np.float32)
