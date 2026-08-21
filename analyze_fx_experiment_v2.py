"""
Анализ эксперимента FX Fee Reduction, используя готовую dbt-таблицу
mart_ab_test_ready вместо ручной агрегации в Python.
"""

import sqlite3
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_engine.core import calculate_theta, check_cuped_test, get_ttest_strat_pvalue

DB_PATH = "ab_platform.db"

# ---------------------------------------------------------------------------
# Читаем ГОТОВУЮ таблицу, построенную dbt (никакого groupby/merge здесь!)
# ---------------------------------------------------------------------------

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM mart_ab_test_ready", conn)
conn.close()

print(f"Total users: {len(df)}")
print(df.head())

df_control = df[df['pilot'] == 0]
df_pilot = df[df['pilot'] == 1]

# ---------------------------------------------------------------------------
# CUPED (используя ковариату из dbt-модели)
# ---------------------------------------------------------------------------

cuped_pvalue, cuped_delta = check_cuped_test(df_control, df_pilot, 'cov')
print(f"\n=== CUPED ===")
print(f"pvalue = {cuped_pvalue:.6f}, delta = {cuped_delta:.4f}")

# ---------------------------------------------------------------------------
# Stratification (по country)
# ---------------------------------------------------------------------------

country_map = {c: i for i, c in enumerate(df['country'].unique())}
df['strat'] = df['country'].map(country_map)

a_strat = df[df['pilot'] == 0][['metric', 'strat']].values
b_strat = df[df['pilot'] == 1][['metric', 'strat']].values

strat_pvalue, strat_delta = get_ttest_strat_pvalue(a_strat, b_strat)
print(f"\n=== Stratification (by country) ===")
print(f"pvalue = {strat_pvalue:.6f}, delta = {strat_delta:.4f}")

# ---------------------------------------------------------------------------
# Naive t-test (для сравнения)
# ---------------------------------------------------------------------------

_, naive_pvalue = stats.ttest_ind(df_control['metric'], df_pilot['metric'])
print(f"\n=== Naive t-test (on pre-aggregated metric) ===")
print(f"pvalue = {naive_pvalue:.6f}")
