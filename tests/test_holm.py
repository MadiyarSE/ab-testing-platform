import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import holm_correction


def test_holm_all_significant():
    """Если все p-value очень маленькие, все гипотезы должны быть отклонены."""
    pvalues = [0.001, 0.002, 0.003]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [True, True, True], f"Expected all rejected, got {rejected}"


def test_holm_none_significant():
    """Если все p-value большие, ни одна гипотеза не должна быть отклонена."""
    pvalues = [0.5, 0.6, 0.9]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [False, False, False], f"Expected none rejected, got {rejected}"


def test_holm_stops_at_first_failure():
    """Как только p-value не проходит свой порог, все более крупные p-value
    (даже потенциально значимые сами по себе) тоже не отклоняются."""
    # m=3, пороги: 0.05/3≈0.0167, 0.05/2=0.025, 0.05/1=0.05
    # отсортировано: 0.01 (проходит 0.0167), 0.03 (не проходит 0.025) -> обрыв
    pvalues = [0.03, 0.01, 0.04]
    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [False, True, False], f"Expected [False, True, False], got {rejected}"


def test_holm_matches_our_topup_experiment():
    """Регрессионный тест на реальном сценарии: primary metric значим,
    guardrail-метрики — нет."""
    metric_names = ["avg_topup_amount", "fraud_rate", "chargeback_rate"]
    pvalues = [0.000000, 0.101980, 0.948383]

    rejected = holm_correction(pvalues, alpha=0.05)

    assert rejected == [True, False, False], (
        f"Expected only avg_topup_amount significant, got {dict(zip(metric_names, rejected))}"
    )


def test_holm_single_pvalue_behaves_like_plain_alpha():
    """При m=1 порог Holm совпадает с обычным alpha."""
    rejected_significant = holm_correction([0.03], alpha=0.05)
    rejected_not_significant = holm_correction([0.07], alpha=0.05)

    assert rejected_significant == [True]
    assert rejected_not_significant == [False]
