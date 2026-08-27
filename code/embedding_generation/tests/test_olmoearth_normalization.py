"""Tests for the preprocessing expected by the OlmoEarth encoder."""

import numpy as np

from embed import (
    OLMOEARTH_S2_MEANS,
    OLMOEARTH_S2_STDS,
    normalize_olmoearth_s2,
)


def test_training_range_maps_to_unit_interval():
    means = OLMOEARTH_S2_MEANS[:, None]
    stds = OLMOEARTH_S2_STDS[:, None]
    chips = np.stack((means - 2 * stds, means, means + 2 * stds), axis=-1)

    got = normalize_olmoearth_s2(chips[None].astype("float32"))

    expected = np.broadcast_to([0.0, 0.5, 1.0], (12, 3))
    np.testing.assert_allclose(got[0, :, 0, :], expected, rtol=1e-6)


def test_nan_imputation_occurs_before_normalization():
    chips = np.broadcast_to(
        OLMOEARTH_S2_MEANS[None, :, None, None],
        (2, 12, 2, 2),
    ).copy()
    chips[0, 0] = np.array([[100.0, np.nan], [300.0, np.nan]])
    chips[1, 1] = np.nan
    original = chips.copy()

    got = normalize_olmoearth_s2(chips)

    raw_mean_normalized = (
        200.0 - (OLMOEARTH_S2_MEANS[0] - 2 * OLMOEARTH_S2_STDS[0])
    ) / (4 * OLMOEARTH_S2_STDS[0])
    np.testing.assert_allclose(got[0, 0, 0, 1], raw_mean_normalized, rtol=1e-6)
    np.testing.assert_allclose(got[1, 1], 0.5, rtol=1e-6)
    np.testing.assert_equal(chips, original)
