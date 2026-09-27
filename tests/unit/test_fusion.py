"""Fusion: feature vector order, each method, thresholds, disagreement gate."""
from __future__ import annotations

import math

import pytest

from phishlens import fusion
from phishlens.fusion import FusionConfig, FusionConfigError, clipped_logit


def test_feature_order_constant_matches_spec():
    assert fusion.FEATURE_ORDER == [
        "logit_p_url_x_a_url",
        "logit_p_text_x_a_text",
        "logit_p_image_x_a_image",
        "a_url",
        "a_text",
        "a_image",
    ]


def test_build_feature_vector_order_and_values():
    probs = {"url": 0.9, "text": None, "image": 0.2}
    features = fusion.build_feature_vector(probs)
    assert len(features) == 6
    assert features[3:] == [1.0, 0.0, 1.0]
    assert features[0] == pytest.approx(clipped_logit(0.9))
    assert features[1] == 0.0
    assert features[2] == pytest.approx(clipped_logit(0.2))


def test_build_feature_vector_all_missing():
    features = fusion.build_feature_vector({"url": None, "text": None, "image": None})
    assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_clipped_logit_extremes_are_finite():
    assert math.isfinite(clipped_logit(0.0))
    assert math.isfinite(clipped_logit(1.0))


# ---- mean ---------------------------------------------------------------
def test_mean_method_averages_available_branches():
    cfg = FusionConfig(method="mean").validate()
    result = fusion.fuse_probabilities({"url": 0.8, "text": 0.4, "image": None}, cfg)
    assert result.score == pytest.approx(0.6)
    assert sorted(result.branches_used) == ["text", "url"]


def test_mean_method_single_branch():
    cfg = FusionConfig(method="mean").validate()
    result = fusion.fuse_probabilities({"url": None, "text": None, "image": 0.7}, cfg)
    assert result.score == pytest.approx(0.7)


def test_mean_method_no_branches_gives_no_score():
    cfg = FusionConfig(method="mean").validate()
    result = fusion.fuse_probabilities({"url": None, "text": None, "image": None}, cfg)
    assert result.score is None
    assert result.branches_used == []


# ---- max ------------------------------------------------------------------
def test_max_method_picks_highest():
    cfg = FusionConfig(method="max").validate()
    result = fusion.fuse_probabilities({"url": 0.1, "text": 0.9, "image": 0.5}, cfg)
    assert result.score == pytest.approx(0.9)


# ---- best_single ------------------------------------------------------------
def test_best_single_uses_named_branch():
    cfg = FusionConfig(method="best_single", best_single_branch="text").validate()
    result = fusion.fuse_probabilities({"url": 0.9, "text": 0.3, "image": 0.9}, cfg)
    assert result.score == pytest.approx(0.3)
    assert result.branches_used == ["text"]


def test_best_single_falls_back_to_mean_when_branch_missing():
    cfg = FusionConfig(method="best_single", best_single_branch="image").validate()
    result = fusion.fuse_probabilities({"url": 0.4, "text": 0.6, "image": None}, cfg)
    assert result.score == pytest.approx(0.5)
    assert result.note is not None


def test_best_single_requires_branch_configured():
    cfg = FusionConfig(method="best_single")
    with pytest.raises(FusionConfigError):
        cfg.validate()


# ---- learned ----------------------------------------------------------------
def test_learned_method_uses_coefficients_and_intercept():
    cfg = FusionConfig(
        method="learned",
        coefficients=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        intercept=0.0,
    ).validate()
    result = fusion.fuse_probabilities({"url": 0.9, "text": None, "image": None}, cfg)
    expected = fusion.sigmoid(clipped_logit(0.9))
    assert result.score == pytest.approx(expected)


def test_learned_method_missing_coefficients_raises():
    cfg = FusionConfig(method="learned")
    with pytest.raises(FusionConfigError):
        cfg.validate()


# ---- config validation ------------------------------------------------------
def test_unknown_method_rejected():
    cfg = FusionConfig(method="bogus")
    with pytest.raises(FusionConfigError):
        cfg.validate()


def test_bad_feature_order_rejected():
    cfg = FusionConfig(feature_order=["wrong"])
    with pytest.raises(FusionConfigError):
        cfg.validate()


def test_bad_thresholds_rejected():
    cfg = FusionConfig(thresholds={"low": 0.8, "high": 0.2})
    with pytest.raises(FusionConfigError):
        cfg.validate()


# ---- default provisional config ------------------------------------------
def test_default_config_is_provisional_mean_with_spec_thresholds():
    cfg = FusionConfig().validate()
    assert cfg.method == "mean"
    assert cfg.thresholds == {"low": 0.30, "high": 0.70}
    assert cfg.provisional is True
    assert cfg.fitted is False


# ---- fit --------------------------------------------------------------------
def test_fit_writes_learned_config(tmp_path):
    probs = [
        [0.9, 0.8, 0.7],
        [0.85, 0.9, 0.6],
        [0.1, 0.2, 0.15],
        [0.05, 0.1, 0.2],
        [0.95, None, None],
        [0.02, None, None],
    ]
    availability = [[p is not None for p in row] for row in probs]
    probs_filled = [[p if p is not None else float("nan") for p in row] for row in probs]
    labels = [1, 1, 0, 0, 1, 0]
    out_path = tmp_path / "fusion.json"
    cfg = fusion.fit(
        probs_filled, availability, labels, output_path=out_path, fitted_on="unit test data"
    )
    assert cfg.method == "learned"
    assert len(cfg.coefficients) == 6
    assert cfg.fitted is True
    assert out_path.exists()
    reloaded = fusion.load_config(out_path)
    assert reloaded.method == "learned"
    assert reloaded.coefficients == cfg.coefficients


def test_fit_requires_both_classes():
    probs = [[0.9, 0.8, 0.7], [0.1, 0.2, 0.15]]
    with pytest.raises(ValueError):
        fusion.fit(probs, None, [1, 1], write=False)
