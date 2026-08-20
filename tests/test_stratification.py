import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import get_ttest_strat_pvalue


def test_stratified_no_effect():
    """Если данные из одного распределения в обеих группах — pvalue не значим."""
    np.random.seed(42)
    n = 300

    # 2 страты, поровну представлены в обеих группах
    metric_a = np.random.normal(100, 15, n)
    strat_a = np.random.choice([0, 1], size=n)
    a = np.column_stack([metric_a, strat_a])

    metric_b = np.random.normal(100, 15, n)
    strat_b = np.random.choice([0, 1], size=n)
    b = np.column_stack([metric_b, strat_b])

    pvalue, delta = get_ttest_strat_pvalue(a, b)

    assert pvalue > 0.05, f"Expected non-significant pvalue, got {pvalue}"


def test_stratified_detects_clear_effect():
    """Явный, сильный эффект между группами должен быть обнаружен."""
    np.random.seed(7)
    n = 300

    metric_a = np.random.normal(100, 15, n)
    strat_a = np.random.choice([0, 1], size=n)
    a = np.column_stack([metric_a, strat_a])

    metric_b = np.random.normal(120, 15, n)  # +20 эффект
    strat_b = np.random.choice([0, 1], size=n)
    b = np.column_stack([metric_b, strat_b])

    pvalue, delta = get_ttest_strat_pvalue(a, b)

    assert pvalue < 0.05, f"Expected significant pvalue, got {pvalue}"
    assert delta > 10, f"Expected delta close to +20, got {delta}"


def test_stratified_reduces_variance_vs_naive():
    """При сильном различии страт по среднему, стратификация должна снижать
    дисперсию оценки по сравнению с наивным сравнением средних."""
    np.random.seed(123)
    n_per_strat = 200

    # strat=0: низкие значения, strat=1: высокие значения (сильное расхождение страт)
    def make_group(shift=0):
        metric_low = np.random.normal(50 + shift, 5, n_per_strat)
        metric_high = np.random.normal(150 + shift, 5, n_per_strat)
        strat = np.array([0] * n_per_strat + [1] * n_per_strat)
        metric = np.concatenate([metric_low, metric_high])
        return np.column_stack([metric, strat])

    a = make_group(shift=0)
    b = make_group(shift=5)

    pvalue, delta = get_ttest_strat_pvalue(a, b)

    assert pvalue < 0.05, f"Expected to detect the +5 shift, got pvalue={pvalue}"
    assert 3 < delta < 7, f"Expected delta close to +5, got {delta}"
