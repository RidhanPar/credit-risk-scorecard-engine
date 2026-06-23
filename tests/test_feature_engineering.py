"""Unit tests for src/feature_engineering.py (logic not requiring trained models)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import IV_THRESHOLD, get_woe_contributions


class TestGetWoeContributions:
    @pytest.fixture
    def sample_woe_and_coefs(self):
        woe = pd.Series({
            "checking_status": -0.5,
            "duration": 0.3,
            "credit_history": -0.2,
            "age": 0.1,
            "savings_status": 0.4,
        })
        coefs = {
            "checking_status": 1.2,
            "duration": -0.8,
            "credit_history": 0.9,
            "age": -0.3,
            "savings_status": 0.6,
        }
        return woe, coefs

    def test_returns_series(self, sample_woe_and_coefs):
        woe, coefs = sample_woe_and_coefs
        result = get_woe_contributions(woe, coefs)
        assert isinstance(result, pd.Series)

    def test_contribution_calculation(self, sample_woe_and_coefs):
        woe, coefs = sample_woe_and_coefs
        result = get_woe_contributions(woe, coefs)
        # checking_status: -0.5 * 1.2 = -0.6
        assert abs(result["checking_status"] - (-0.5 * 1.2)) < 1e-9

    def test_sorted_by_absolute_value(self, sample_woe_and_coefs):
        woe, coefs = sample_woe_and_coefs
        result = get_woe_contributions(woe, coefs)
        abs_vals = result.abs().values
        assert all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1))

    def test_missing_feature_in_woe_ignored(self):
        woe = pd.Series({"feature_a": 0.5, "feature_b": -0.3})
        coefs = {"feature_a": 1.0, "feature_c": 2.0}  # feature_c not in woe
        result = get_woe_contributions(woe, coefs)
        assert "feature_c" not in result.index
        assert "feature_a" in result.index


class TestIvThreshold:
    def test_threshold_value(self):
        assert IV_THRESHOLD == 0.02

    def test_iv_threshold_type(self):
        assert isinstance(IV_THRESHOLD, float)


class TestMonitoringPsi:
    """Tests for PSI calculation — no model required."""

    def test_identical_distributions_psi_near_zero(self):
        from src.monitoring import calculate_psi
        scores = np.random.default_rng(0).integers(300, 851, 500)
        psi = calculate_psi(scores, scores.copy())
        assert psi < 0.05

    def test_psi_returns_nonnegative_float(self):
        from src.monitoring import calculate_psi
        rng = np.random.default_rng(1)
        expected = rng.integers(300, 851, 300)
        actual = rng.integers(400, 851, 300)
        psi = calculate_psi(expected, actual)
        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_interpret_psi_stable(self):
        from src.monitoring import interpret_psi
        assert interpret_psi(0.05) == "Stable"

    def test_interpret_psi_warning(self):
        from src.monitoring import interpret_psi
        assert "Slight" in interpret_psi(0.15)

    def test_interpret_psi_alert(self):
        from src.monitoring import interpret_psi
        assert "Major" in interpret_psi(0.30)
