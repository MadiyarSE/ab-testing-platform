import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import holm_correction


def test_holm_all_significant():
    """If all p-values are very small, all hypotheses should be rejected."""
    pvalues = [0.001, 0.002, 0.003]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [True, True, True], f"Expected all rejected, got {rejected}"


def test_holm_none_significant():
    """If all p-values are large, no hypotheses should be rejected."""
    pvalues = [0.5, 0.6, 0.9]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [False, False, False], f"Expected none rejected, got {rejected}"


def test_holm_stops_at_first_failure():
    """Once a p-value fails to meet its threshold, all larger p-values 
    (even if potentially significant on their own) are also not rejected."""
    # m=3, thresholds: 0.05/3≈0.0167, 0.05/2=0.025, 0.05/1=0.05
    # sorted: 0.01 (passes 0.0167), 0.03 (fails 0.025) -> stop
    pvalues = [0.03, 0.01, 0.04]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [False, True, False], f"Expected [False, True, False], got {rejected}"


def test_holm_matches_our_topup_experiment():
    """Regression test on a real scenario: primary metric is significant, 
    guardrail metrics are not."""
    metric_names = ["avg_topup_amount", "fraud_rate", "chargeback_rate"]
    pvalues = [0.000000, 0.101980, 0.948383]

    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [True, False, False], (
        f"Expected only avg_topup_amount significant, got {dict(zip(metric_names, rejected))}"
    )


def test_holm_single_pvalue_behaves_like_plain_alpha():
    """When m=1, the Holm threshold coincides with standard alpha."""
    rejected_significant = holm_correction([0.03], alpha=0.05)
    rejected_not_significant = holm_correction([0.07], alpha=0.05)

    assert rejected_significant == [True]
    assert rejected_not_significant == [False]