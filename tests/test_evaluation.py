"""Unit tests for src/evaluation.py metric functions."""

import numpy as np
import pytest

from src.evaluation import gini_coefficient, ks_statistic, roc_auc


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    return y_true, y_prob


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    return y_true, y_prob


class TestGiniCoefficient:
    def test_returns_float(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = gini_coefficient(y_true, y_prob)
        assert isinstance(result, float)

    def test_range_zero_to_one(self, random_predictions):
        y_true, y_prob = random_predictions
        result = gini_coefficient(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_perfect_model_near_one(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = gini_coefficient(y_true, y_prob)
        assert result > 0.95

    def test_random_model_near_zero(self, random_predictions):
        y_true, y_prob = random_predictions
        result = gini_coefficient(y_true, y_prob)
        # Random model Gini should be close to 0 (not necessarily below 0.1
        # with only 200 samples, but definitely below 0.4)
        assert result < 0.4

    def test_equals_two_auc_minus_one(self, random_predictions):
        y_true, y_prob = random_predictions
        auc = roc_auc(y_true, y_prob)
        gini = gini_coefficient(y_true, y_prob)
        assert abs(gini - (2 * auc - 1)) < 1e-9


class TestRocAuc:
    def test_returns_float(self, random_predictions):
        y_true, y_prob = random_predictions
        assert isinstance(roc_auc(y_true, y_prob), float)

    def test_range(self, random_predictions):
        y_true, y_prob = random_predictions
        result = roc_auc(y_true, y_prob)
        assert 0.0 <= result <= 1.0


class TestKsStatistic:
    def test_returns_float(self, random_predictions):
        y_true, y_prob = random_predictions
        assert isinstance(ks_statistic(y_true, y_prob), float)

    def test_range_zero_to_one(self, random_predictions):
        y_true, y_prob = random_predictions
        result = ks_statistic(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_perfect_model_ks(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = ks_statistic(y_true, y_prob)
        assert result > 0.8
