import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import calculate_pooled_p, get_proportions_ztest_pvalue


def test_calculate_pooled_p_known_values():
    """Проверяем pooled p на простом примере: (10+20)/(100+100) = 0.15."""
    pooled_p = calculate_pooled_p(count_a=10, nobs_a=100, count_b=20, nobs_b=100)

    assert abs(pooled_p - 0.15) < 1e-9, f"Expected pooled_p=0.15, got {pooled_p}"


def test_ztest_no_effect_gives_high_pvalue():
    """Если пропорции в обеих группах равны, p-value должен быть большим (не значимо)."""
    pvalue = get_proportions_ztest_pvalue(
        count_a=100, nobs_a=1000, count_b=100, nobs_b=1000
    )

    assert pvalue > 0.5, f"Expected high pvalue for equal proportions, got {pvalue}"


def test_ztest_detects_strong_effect():
    """Z-test должен обнаружить явную, сильную разницу в пропорциях."""
    # control: 5% conversion, pilot: 15% conversion, большая выборка
    pvalue = get_proportions_ztest_pvalue(
        count_a=50, nobs_a=1000, count_b=150, nobs_b=1000
    )

    assert pvalue < 0.05, f"Expected significant pvalue for strong effect, got {pvalue}"


def test_ztest_symmetric_regardless_of_group_order():
    """P-value не должен зависеть от того, какую группу подать первой."""
    pvalue_ab = get_proportions_ztest_pvalue(count_a=30, nobs_a=500, count_b=60, nobs_b=500)
    pvalue_ba = get_proportions_ztest_pvalue(count_a=60, nobs_a=500, count_b=30, nobs_b=500)

    assert abs(pvalue_ab - pvalue_ba) < 1e-9, (
        f"P-value should be symmetric: ab={pvalue_ab}, ba={pvalue_ba}"
    )


def test_ztest_matches_manual_calculation():
    """Сверяем результат функции с ручным расчётом z-статистики."""
    count_a, nobs_a = 40, 400   # p_a = 0.10
    count_b, nobs_b = 60, 400   # p_b = 0.15

    p = (count_a + count_b) / (nobs_a + nobs_b)
    se = np.sqrt(p * (1 - p) * (1 / nobs_a + 1 / nobs_b))
    z = (count_b / nobs_b - count_a / nobs_a) / se
    from scipy import stats
    expected_pvalue = 2 * (1 - stats.norm.cdf(abs(z)))

    pvalue = get_proportions_ztest_pvalue(count_a, nobs_a, count_b, nobs_b)

    assert abs(pvalue - expected_pvalue) < 1e-9, (
        f"Expected pvalue={expected_pvalue}, got {pvalue}"
    )
