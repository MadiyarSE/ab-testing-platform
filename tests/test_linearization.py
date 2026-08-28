import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import check_linearization, get_ttest_pvalue


def test_check_linearization_no_effect():
    """If data comes from the same distribution in both groups, p-value should be non-significant."""
    np.random.seed(42)
    a = [np.random.normal(10, 3, size=np.random.randint(2, 5)) for _ in range(50)]
    b = [np.random.normal(10, 3, size=np.random.randint(2, 5)) for _ in range(50)]

    pvalue, delta = check_linearization(a, b)

    assert pvalue > 0.05, f"Expected non-significant pvalue, got {pvalue}"


def test_check_linearization_clear_effect():
    """If pilot is clearly and significantly larger than control, p-value should be small."""
    a = [np.array([10, 10]), np.array([10, 10]), np.array([10, 10])]
    b = [np.array([50, 50]), np.array([50, 50]), np.array([50, 50])]

    pvalue, delta = check_linearization(a, b)

    assert pvalue < 0.05, f"Expected significant pvalue for clear effect, got {pvalue}"
    assert delta > 0, f"Expected positive delta (b > a), got {delta}"


def test_get_ttest_pvalue_basic():
    """Simple check that standard t-test works and returns a number between 0 and 1."""
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]

    pvalue = get_ttest_pvalue(a, b)

    assert 0 <= pvalue <= 1, f"pvalue should be between 0 and 1, got {pvalue}"
    assert pvalue == 1.0, "Identical samples should return pvalue=1.0"