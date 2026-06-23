"""Unit tests for src/scorecard.py."""

import numpy as np
import pytest

from src.scorecard import (
    SCORE_MAX,
    SCORE_MIN,
    _compute_scaling_factors,
    get_score_tier,
    log_odds_to_score,
)


class TestScalingFactors:
    def test_factor_is_pdo_over_ln2(self):
        factor, _ = _compute_scaling_factors(pdo=20, base_score=600, base_odds=1.0)
        expected = 20 / np.log(2)
        assert abs(factor - expected) < 1e-8

    def test_offset_at_unit_odds(self):
        # When base_odds=1, offset should equal base_score
        _, offset = _compute_scaling_factors(pdo=20, base_score=600, base_odds=1.0)
        assert abs(offset - 600) < 1e-6

    def test_doubling_odds_reduces_score_by_pdo(self):
        pdo = 20
        factor, offset = _compute_scaling_factors(pdo=pdo, base_score=600, base_odds=1.0)
        score_base = log_odds_to_score(np.log(1.0), factor, offset)
        score_double = log_odds_to_score(np.log(2.0), factor, offset)
        assert abs((score_base - score_double) - pdo) <= 1  # rounding tolerance


class TestLogOddsToScore:
    @pytest.fixture
    def factors(self):
        return _compute_scaling_factors(pdo=20, base_score=600, base_odds=1.0)

    def test_returns_int(self, factors):
        factor, offset = factors
        result = log_odds_to_score(0.0, factor, offset)
        assert isinstance(result, int)

    def test_score_in_range(self, factors):
        factor, offset = factors
        for log_odds in np.linspace(-10, 10, 50):
            score = log_odds_to_score(log_odds, factor, offset)
            assert SCORE_MIN <= score <= SCORE_MAX

    def test_clamped_at_extremes(self, factors):
        factor, offset = factors
        assert log_odds_to_score(-100, factor, offset) == SCORE_MAX
        assert log_odds_to_score(100, factor, offset) == SCORE_MIN

    def test_base_odds_gives_base_score(self, factors):
        factor, offset = factors
        score = log_odds_to_score(np.log(1.0), factor, offset)
        assert score == 600


class TestGetScoreTier:
    def test_low_risk(self):
        assert get_score_tier(750) == "Low Risk"
        assert get_score_tier(700) == "Low Risk"

    def test_medium_risk(self):
        assert get_score_tier(699) == "Medium Risk"
        assert get_score_tier(600) == "Medium Risk"

    def test_high_risk(self):
        assert get_score_tier(599) == "High Risk"
        assert get_score_tier(500) == "High Risk"

    def test_declined(self):
        assert get_score_tier(499) == "Declined"
        assert get_score_tier(300) == "Declined"

    def test_boundary_exhaustive(self):
        tiers = {get_score_tier(s) for s in range(300, 851)}
        assert tiers == {"Low Risk", "Medium Risk", "High Risk", "Declined"}
