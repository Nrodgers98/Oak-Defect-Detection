import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import predict_image as prediction


def test_tessellate_image_matches_requested_examples():
    image = np.zeros((2000, 5030, 3), dtype=np.uint8)

    crops, boxes = prediction.tessellate_image(image, 3000, 1000)

    assert boxes == [
        (0, 1000, 0, 3000),
        (0, 1000, 3000, 5030),
        (1000, 2000, 0, 3000),
        (1000, 2000, 3000, 5030),
    ]
    assert [crop.shape[:2] for crop in crops] == [
        (1000, 3000),
        (1000, 2030),
        (1000, 3000),
        (1000, 2030),
    ]

    _, wide_boxes = prediction.tessellate_image(image, 3000, 9999)
    assert wide_boxes == [(0, 2000, 0, 3000), (0, 2000, 3000, 5030)]


def test_tessellate_image_covers_non_divisible_image_once():
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    crops, boxes = prediction.tessellate_image(image, 4, 3)
    coverage = np.zeros(image.shape[:2], dtype=np.uint8)

    for crop, (ymin, ymax, xmin, xmax) in zip(crops, boxes):
        assert crop.shape[:2] == (ymax - ymin, xmax - xmin)
        coverage[ymin:ymax, xmin:xmax] += 1

    assert coverage.shape == image.shape[:2]
    assert np.all(coverage == 1)


@pytest.mark.parametrize("max_x,max_y", [(0, 100), (100, 0), (-1, 100), (100, -1)])
def test_tessellate_image_rejects_non_positive_limits(max_x, max_y):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        prediction.tessellate_image(image, max_x, max_y)


def test_predict_image_tiled_stitches_native_probability_maps(monkeypatch):
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    calls = []

    def fake_predict_full_image(model, crop, device="cuda", mean=None, std=None):
        calls.append(crop.shape[:2])
        height, width = crop.shape[:2]
        probabilities = np.zeros((2, height, width), dtype=np.float32)
        probabilities[1] = 1.0
        return np.ones((height, width), dtype=np.uint8), probabilities

    monkeypatch.setattr(prediction, "predict_full_image", fake_predict_full_image)
    mask, probabilities = prediction.predict_image_tiled(
        model=object(),
        image=image,
        max_crop_width=4,
        max_crop_height=3,
    )

    assert calls == [(3, 4), (3, 3), (2, 4), (2, 3)]
    assert mask.shape == (5, 7)
    assert probabilities.shape == (2, 5, 7)
    assert np.all(mask == 1)
    assert np.all(probabilities[1] == 1.0)


def test_apply_confidence_threshold_suppresses_only_low_confidence_winners():
    pred_mask = np.array([[1, 1, 2, 2, 0]], dtype=np.uint8)
    probabilities = np.array([
        [[0.10, 0.20, 0.10, 0.20, 1.00]],
        [[0.25, 0.30, 0.60, 0.10, 0.00]],
        [[0.65, 0.50, 0.30, 0.70, 0.00]],
    ], dtype=np.float32)

    thresholded = prediction.apply_confidence_threshold(
        pred_mask, probabilities, {1: 30, 2: 50}
    )

    np.testing.assert_array_equal(thresholded, [[0, 1, 0, 2, 0]])


def test_apply_confidence_threshold_validates_inputs():
    pred_mask = np.zeros((2, 2), dtype=np.uint8)
    probabilities = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        prediction.apply_confidence_threshold(pred_mask, probabilities, {2: 10})
    with pytest.raises(ValueError):
        prediction.apply_confidence_threshold(pred_mask, probabilities, {1: 101})
    with pytest.raises(ValueError):
        prediction.apply_confidence_threshold(
            pred_mask, np.zeros((2, 3, 2), dtype=np.float32), {1: 10}
        )
