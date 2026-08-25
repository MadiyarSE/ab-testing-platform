"""
Симуляция ежедневного мониторинга эксперимента FX Fee Reduction.

Демонстрирует "peeking problem": если проверять p-value каждый день
и останавливаться при первой значимости, реальная ошибка I рода
оказывается намного выше заявленного alpha.

Поддерживает два режима:
- apply_effect=True  -- реальный FX-эксперимент (эффект есть, наш основной кейс)
- apply_effect=False -- AA-тест (эффекта нет), демонстрирует, как p-value
  может случайно "нырять" ниже 0.05 даже при отсутствии реального эффекта

Не является частью основного production-пайплайна (generate -> dbt ->
analyze) -- это отдельная образовательная симуляция.
"""

import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_engine.core import calculate_theta, check_cuped_test

# ---------------------------------------------------------------------------
# Параметры симуляции (переиспользуем логику из основного генератора)
# ---------------------------------------------------------------------------

COUNTRY_PARAMS = {
    "UK": {"weight": 0.35, "mean_amount": 180, "sigma_amount": 0.45, "base_freq": 2.8},
    "DE": {"weight": 0.30, "mean_amount": 140, "sigma_amount": 0.50, "base_freq": 2.2},
    "PL": {"weight": 0.20, "mean_amount": 90, "sigma_amount": 0.55, "base_freq": 1.8},
    "ES": {"weight": 0.15, "mean_amount": 110, "sigma_amount": 0.50, "base_freq": 2.0},
}

BASE_FEE = 0.005
PILOT_FEE = 0.002
FREQUENCY_LIFT = 1.15


def _sample_country(rng):
    countries = list(COUNTRY_PARAMS.keys())
    weights = [COUNTRY_PARAMS[c]["weight"] for c in countries]
    return rng.choice(countries, p=weights)


def _daily_user_revenue(rng, country, is_pilot, apply_effect=True):
    """Генерирует revenue ОДНОГО пользователя за ОДИН день.

    :param apply_effect: если False -- pilot ведёт себя ТОЧНО так же, как
        control (чистый AA-тест, никакого реального эффекта).
    """
    params = COUNTRY_PARAMS[country]

    if apply_effect and is_pilot:
        fee = PILOT_FEE
        freq_multiplier = FREQUENCY_LIFT
    else:
        fee = BASE_FEE
        freq_multiplier = 1.0

    lam = params["base_freq"] * freq_multiplier / 7  # дневная интенсивность
    n_tx = rng.poisson(lam=max(lam, 0.001))

    if n_tx == 0:
        return 0.0

    amounts = rng.lognormal(
        mean=np.log(params["mean_amount"]), sigma=params["sigma_amount"], size=n_tx
    )
    return float(np.sum(amounts * fee))


def simulate_daily_peeking(n_users_per_group=1428, n_days=7, seed=42, apply_effect=True):
    """Симулирует накопление данных день за днём и считает p-value
    на накопленных данных после каждого дня (naive t-test и CUPED).

    :param apply_effect: True -- реальный FX-эксперимент с эффектом.
        False -- AA-тест (для демонстрации peeking problem).
    :return: pd.DataFrame с колонками ['day', 'naive_pvalue', 'cuped_pvalue',
        'naive_delta', 'cuped_delta', 'n_users_so_far']
    """
    rng = np.random.default_rng(seed)

    users = []
    for i in range(n_users_per_group * 2):
        country = _sample_country(rng)
        pilot = 1 if i % 2 == 1 else 0
        users.append({"user_id": i, "country": country, "pilot": pilot})
    users_df = pd.DataFrame(users)

    # историческая ковариата для CUPED -- ВСЕГДА без эффекта (это прошлое,
    # эксперимент ещё не начался, apply_effect сюда не передаём)
    history_revenue = {}
    for _, row in users_df.iterrows():
        history_revenue[row["user_id"]] = sum(
            _daily_user_revenue(rng, row["country"], is_pilot=False, apply_effect=False)
            for _ in range(28)
        )
    users_df["cov"] = users_df["user_id"].map(history_revenue)

    cumulative_revenue = {uid: 0.0 for uid in users_df["user_id"]}
    results = []

    for day in range(1, n_days + 1):
        for _, row in users_df.iterrows():
            daily_rev = _daily_user_revenue(
                rng, row["country"], bool(row["pilot"]), apply_effect=apply_effect
            )
            cumulative_revenue[row["user_id"]] += daily_rev

        users_df["metric"] = users_df["user_id"].map(cumulative_revenue)

        df_control = users_df[users_df["pilot"] == 0]
        df_pilot = users_df[users_df["pilot"] == 1]

        _, naive_pvalue = stats.ttest_ind(df_control["metric"], df_pilot["metric"])
        naive_delta = df_pilot["metric"].mean() - df_control["metric"].mean()

        cuped_pvalue, cuped_delta = check_cuped_test(df_control, df_pilot, "cov")

        results.append({
            "day": day,
            "n_users_so_far": len(users_df),
            "naive_pvalue": naive_pvalue,
            "naive_delta": naive_delta,
            "cuped_pvalue": cuped_pvalue,
            "cuped_delta": cuped_delta,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Scenario 1: Real FX experiment (effect exists) ===")
    result_effect = simulate_daily_peeking(n_users_per_group=1428, n_days=7, apply_effect=True)
    print(result_effect)
    result_effect.to_csv("peeking_with_effect.csv", index=False)

    print("\n=== Scenario 2: AA-test (no effect, peeking demonstration) ===")
    result_aa = simulate_daily_peeking(n_users_per_group=1428, n_days=7, seed=123, apply_effect=False)
    print(result_aa)
    result_aa.to_csv("peeking_aa_test.csv", index=False)

    print("\nSaved both scenarios to CSV.")
