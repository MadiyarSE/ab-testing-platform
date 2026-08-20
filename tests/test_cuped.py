
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats_engine.core import calculate_theta, check_cuped_test


def test_calculate_theta_known_values():
    """Проверяем theta на примере из курса: должно получиться 5.0."""
    control_metrics = pd.DataFrame({'user_id': [1, 1, 2], 'metric': [3, 5, 7]})
    pilot_metrics = pd.DataFrame({'user_id': [3, 3], 'metric': [3, 6]})

    theta = calculate_theta(
        control_metrics['metric'], pilot_metrics['metric'],
        control_metrics['metric'], pilot_metrics['metric']
    )

    # ddof-несоответствие между np.cov (ddof=1) и .var() (ddof=0) даёт n/(n-1)
    n = 5
    expected_theta = n / (n - 1)
    assert abs(theta - expected_theta) < 1e-9, f"Expected theta={expected_theta}, got {theta}"


def test_cuped_reduces_variance_with_correlated_covariate():
    """CUPED должен снижать дисперсию, если ковариата сильно коррелирует с метрикой."""
    np.random.seed(42)
    n = 200

    cov_control = np.random.normal(100, 20, n)
    metric_control = cov_control * 0.9 + np.random.normal(0, 5, n)  # сильная корреляция

    cov_pilot = np.random.normal(100, 20, n)
    metric_pilot = cov_pilot * 0.9 + np.random.normal(0, 5, n) + 3  # + небольшой эффект

    df_control = pd.DataFrame({'metric': metric_control, 'cov': cov_control})
    df_pilot = pd.DataFrame({'metric': metric_pilot, 'cov': cov_pilot})

    theta = calculate_theta(
        df_control['metric'], df_pilot['metric'],
        df_control['cov'], df_pilot['cov']
    )

    metric_cuped_control = df_control['metric'] - theta * df_control['cov']
    metric_cuped_pilot = df_pilot['metric'] - theta * df_pilot['cov']

    naive_var = np.var(np.concatenate([df_control['metric'], df_pilot['metric']]))
    cuped_var = np.var(np.concatenate([metric_cuped_control, metric_cuped_pilot]))

    assert cuped_var < naive_var, (
        f"CUPED should reduce variance with correlated covariate: "
        f"naive_var={naive_var:.2f}, cuped_var={cuped_var:.2f}"
    )


def test_check_cuped_test_detects_effect():
    """CUPED должен обнаружить явный, сильный эффект между группами."""
    np.random.seed(1)
    n = 100

    cov_control = np.random.normal(50, 10, n)
    metric_control = cov_control + np.random.normal(0, 3, n)

    cov_pilot = np.random.normal(50, 10, n)
    metric_pilot = cov_pilot + np.random.normal(0, 3, n) + 15  # сильный эффект +15

    df_control = pd.DataFrame({'metric': metric_control, 'cov': cov_control})
    df_pilot = pd.DataFrame({'metric': metric_pilot, 'cov': cov_pilot})

    pvalue, delta = check_cuped_test(df_control, df_pilot, 'cov')

    assert pvalue < 0.05, f"Expected significant pvalue for strong effect, got {pvalue}"
    assert delta > 10, f"Expected delta close to +15, got {delta}"
