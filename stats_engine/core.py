"""Core statistical methods for A/B test evaluation."""

import numpy as np
import pandas as pd
from scipy import stats


def get_ttest_pvalue(metrics_a_group, metrics_b_group):
    """Applies Student's t-test and returns the p-value."""
    return stats.ttest_ind(metrics_a_group, metrics_b_group).pvalue


def check_linearization(a, b):
    """Checks the hypothesis using linearization.

    a: List[List], list of user metric arrays for the control group
    b: List[List], list of user metric arrays for the pilot group
    """
    a_x = np.array([np.sum(row) for row in a])
    a_y = np.array([len(row) for row in a])
    b_x = np.array([np.sum(row) for row in b])
    b_y = np.array([len(row) for row in b])
    coef = np.sum(a_x) / np.sum(a_y) if np.sum(a_y) > 0 else 0
    a_lin = a_x - coef * a_y
    b_lin = b_x - coef * b_y
    _, pvalue = stats.ttest_ind(a_lin, b_lin)
    delta = np.mean(b_lin) - np.mean(a_lin)
    return pvalue, delta


def calculate_theta(y_control, y_pilot, x_control, x_pilot):
    """Calculates theta (CUPED coefficient) using data from both groups."""
    y = np.hstack([y_control, y_pilot])
    x = np.hstack([x_control, x_pilot])
    covariance = np.cov(x, y)[0, 1]
    variance = x.var()
    return covariance / variance


def check_cuped_test(df_control, df_pilot, covariate_column):
    """Tests the hypothesis of mean equality using CUPED."""
    theta = calculate_theta(
        df_control['metric'], df_pilot['metric'],
        df_control[covariate_column], df_pilot[covariate_column]
    )
    metric_cuped_control = df_control['metric'] - theta * df_control[covariate_column]
    metric_cuped_pilot = df_pilot['metric'] - theta * df_pilot[covariate_column]
    _, pvalue = stats.ttest_ind(metric_cuped_control, metric_cuped_pilot)
    delta = metric_cuped_pilot.mean() - metric_cuped_control.mean()
    return pvalue, delta


def get_ttest_strat_pvalue(metrics_strat_a_group, metrics_strat_b_group):
    """Applies post-stratification and returns the p-value.

    Stratum weights are calculated using data from both groups.
    """
    weights = pd.Series(
        np.vstack([metrics_strat_a_group, metrics_strat_b_group])[:, 1]
    ).value_counts(normalize=True).to_dict()

    df_a = pd.DataFrame({'metric': metrics_strat_a_group[:, 0], 'strat': metrics_strat_a_group[:, 1]})
    df_b = pd.DataFrame({'metric': metrics_strat_b_group[:, 0], 'strat': metrics_strat_b_group[:, 1]})

    strat_a_mean = df_a.groupby('strat')['metric'].mean()
    strat_a_var = df_a.groupby('strat')['metric'].var()
    strat_b_mean = df_b.groupby('strat')['metric'].mean()
    strat_b_var = df_b.groupby('strat')['metric'].var()

    def weighted(stat_series):
        merged = pd.merge(stat_series, pd.Series(weights, name='weight'),
                           how='inner', left_index=True, right_index=True)
        merged['weight'] = merged['weight'] / merged['weight'].sum()
        return (merged['weight'] * merged['metric']).sum()

    mean_strat_a = weighted(strat_a_mean)
    mean_strat_b = weighted(strat_b_mean)
    var_strat_a = weighted(strat_a_var)
    var_strat_b = weighted(strat_b_var)

    delta_mean_strat = mean_strat_b - mean_strat_a
    std_mean_strat = (var_strat_b / len(metrics_strat_b_group) + var_strat_a / len(metrics_strat_a_group)) ** 0.5
    t = delta_mean_strat / std_mean_strat
    pvalue = (1 - stats.norm.cdf(np.abs(t))) * 2
    return pvalue, delta_mean_strat


def calculate_pooled_p(count_a, nobs_a, count_b, nobs_b):
    return (count_a + count_b) / (nobs_a + nobs_b)

def get_proportions_ztest_pvalue(count_a, nobs_a, count_b, nobs_b):
    p = calculate_pooled_p(count_a, nobs_a, count_b, nobs_b)
    p_a = count_a / nobs_a
    p_b = count_b / nobs_b
    se = np.sqrt(p * (1 - p) * (1/nobs_a + 1/nobs_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value