"""
Runs the existing stats_engine (linearization, CUPED, stratification)
on synthetic data from the FX Fee Reduction experiment.
"""

import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

DB_PATH = "ab_platform.db"


# ---------------------------------------------------------------------------
# Functions from your notebook (reused as-is)
# ---------------------------------------------------------------------------

def check_linearization(a, b):
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
    y = np.hstack([y_control, y_pilot])
    x = np.hstack([x_control, x_pilot])
    covariance = np.cov(x, y)[0, 1]
    variance = x.var()
    return covariance / variance


def check_cuped_test(df_control, df_pilot, covariate_column):
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


# ---------------------------------------------------------------------------
# 1. Read data from SQLite
# ---------------------------------------------------------------------------

conn = sqlite3.connect(DB_PATH)
users = pd.read_sql("SELECT * FROM users", conn)
tx_exp = pd.read_sql("SELECT * FROM transactions_experiment", conn)
tx_hist = pd.read_sql("SELECT * FROM transactions_history", conn)
conn.close()

print(f"users: {len(users)}, tx_exp: {len(tx_exp)}, tx_hist: {len(tx_hist)}")

# ---------------------------------------------------------------------------
# 2. Naive t-test (baseline — shows incorrect calibration/unit of randomization)
# ---------------------------------------------------------------------------

merged_naive = tx_exp.merge(users[['user_id', 'pilot']], on='user_id')
naive_a = merged_naive[merged_naive['pilot'] == 0]['revenue']
naive_b = merged_naive[merged_naive['pilot'] == 1]['revenue']
_, naive_pvalue = stats.ttest_ind(naive_a, naive_b)
print(f"\n=== Naive t-test (per transaction) ===")
print(f"pvalue = {naive_pvalue:.6f}")

# ---------------------------------------------------------------------------
# 3. Linearization
# ---------------------------------------------------------------------------

user_revenue_lists = tx_exp.groupby('user_id')['revenue'].apply(list)
df_user_lists = users[['user_id', 'pilot']].merge(
    user_revenue_lists.rename('revenue_list'), on='user_id', how='left'
)
df_user_lists['revenue_list'] = df_user_lists['revenue_list'].apply(
    lambda x: x if isinstance(x, list) else []
)

a_lists = df_user_lists[df_user_lists['pilot'] == 0]['revenue_list'].values
b_lists = df_user_lists[df_user_lists['pilot'] == 1]['revenue_list'].values

lin_pvalue, lin_delta = check_linearization(a_lists, b_lists)
print(f"\n=== Linearization ===")
print(f"pvalue = {lin_pvalue:.6f}, delta = {lin_delta:.4f}")

# ---------------------------------------------------------------------------
# 4. CUPED
# ---------------------------------------------------------------------------

user_metric = tx_exp.groupby('user_id')['revenue'].sum().rename('metric').reset_index()
user_cov = tx_hist.groupby('user_id')['revenue'].sum().rename('cov').reset_index()

df_cuped = users[['user_id', 'pilot']].merge(user_metric, on='user_id', how='left')
df_cuped = df_cuped.merge(user_cov, on='user_id', how='left')
df_cuped[['metric', 'cov']] = df_cuped[['metric', 'cov']].fillna(0)

df_control_cuped = df_cuped[df_cuped['pilot'] == 0]
df_pilot_cuped = df_cuped[df_cuped['pilot'] == 1]

cuped_pvalue, cuped_delta = check_cuped_test(df_control_cuped, df_pilot_cuped, 'cov')
print(f"\n=== CUPED ===")
print(f"pvalue = {cuped_pvalue:.6f}, delta = {cuped_delta:.4f}")

# ---------------------------------------------------------------------------
# 5. Stratification by country
# ---------------------------------------------------------------------------

country_map = {c: i for i, c in enumerate(users['country'].unique())}
df_strat = df_cuped.merge(users[['user_id', 'country']], on='user_id')
df_strat['strat'] = df_strat['country'].map(country_map)

a_strat = df_strat[df_strat['pilot'] == 0][['metric', 'strat']].values
b_strat = df_strat[df_strat['pilot'] == 1][['metric', 'strat']].values

strat_pvalue, strat_delta = get_ttest_strat_pvalue(a_strat, b_strat)
print(f"\n=== Stratification (by country) ===")
print(f"pvalue = {strat_pvalue:.6f}, delta = {strat_delta:.4f}")

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("\n=== SUMMARY ===")
print(f"{'Method':<20} {'p-value':<15} {'delta':<10}")
print(f"{'Naive (broken)':<20} {naive_pvalue:<15.6f} {'N/A':<10}")
print(f"{'Linearization':<20} {lin_pvalue:<15.6f} {lin_delta:<10.4f}")
print(f"{'CUPED':<20} {cuped_pvalue:<15.6f} {cuped_delta:<10.4f}")
print(f"{'Stratification':<20} {strat_pvalue:<15.6f} {strat_delta:<10.4f}")