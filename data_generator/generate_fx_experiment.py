"""
Генератор синтетических данных для эксперимента "FX Fee Reduction".

Гипотеза: снижение комиссии за обмен валют с 0.5% до 0.2% увеличивает
частоту обменов, но вопрос — компенсирует ли рост частоты потерю
маржи с каждой транзакции (revenue = amount * fee).

Структура вывода (SQLite):
    - users: user_id, country, pilot, signup_date
    - transactions_experiment: transaction_id, user_id, date, amount, fee, revenue
    - transactions_history: то же самое, но за 4 недели ДО эксперимента (для CUPED)
"""

import hashlib
import sqlite3
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Split-система (двойное хеширование) — переиспользуем логику из курса
# ---------------------------------------------------------------------------

def get_hash_modulo(value: str, modulo: int, salt: str) -> int:
    """Вычисляет остаток от деления: hash(value + salt) % modulo."""
    hash_value = int(hashlib.md5(str.encode(value + salt)).hexdigest(), 16)
    return hash_value % modulo


def assign_group(user_id: str, salt: str) -> str:
    """Второе хеширование: относит пользователя в control/pilot (50/50)."""
    bucket = get_hash_modulo(user_id, 2, salt)
    return "pilot" if bucket == 1 else "control"


# ---------------------------------------------------------------------------
# Параметры стран (для стратификации) — разные "базовые" паттерны трат
# ---------------------------------------------------------------------------

COUNTRY_PARAMS = {
    "UK": {"weight": 0.35, "mean_amount": 180, "sigma_amount": 0.45, "base_freq": 2.8},
    "DE": {"weight": 0.30, "mean_amount": 140, "sigma_amount": 0.50, "base_freq": 2.2},
    "PL": {"weight": 0.20, "mean_amount": 90, "sigma_amount": 0.55, "base_freq": 1.8},
    "ES": {"weight": 0.15, "mean_amount": 110, "sigma_amount": 0.50, "base_freq": 2.0},
}

BASE_FEE = 0.005          # 0.5% — стандартная комиссия (control)
PILOT_FEE = 0.002         # 0.2% — сниженная комиссия (pilot)
FREQUENCY_LIFT = 1.15     # +15% к частоте обменов у pilot (эластичность спроса)

EXPERIMENT_SALT = "fx_fee_reduction_v1"


def _sample_country(rng: np.random.Generator) -> str:
    countries = list(COUNTRY_PARAMS.keys())
    weights = [COUNTRY_PARAMS[c]["weight"] for c in countries]
    return rng.choice(countries, p=weights)


def _generate_user_transactions(
    rng: np.random.Generator,
    user_id: str,
    country: str,
    is_pilot: bool,
    period_start: datetime,
    period_days: int,
) -> list[dict]:
    """Генерирует список FX-транзакций одного пользователя за период."""
    params = COUNTRY_PARAMS[country]
    fee = PILOT_FEE if is_pilot else BASE_FEE
    freq_multiplier = FREQUENCY_LIFT if is_pilot else 1.0

    # ожидаемое число транзакций за период (масштабируем недельную частоту)
    lam = params["base_freq"] * freq_multiplier * (period_days / 7)
    n_transactions = rng.poisson(lam=max(lam, 0.01))

    rows = []
    for _ in range(n_transactions):
        amount = rng.lognormal(
            mean=np.log(params["mean_amount"]), sigma=params["sigma_amount"]
        )
        amount = round(float(amount), 2)
        revenue = round(amount * fee, 4)
        offset_days = rng.uniform(0, period_days)
        tx_date = period_start + timedelta(days=float(offset_days))
        rows.append(
            {
                "transaction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "date": tx_date,
                "amount": amount,
                "fee": fee,
                "revenue": revenue,
            }
        )
    return rows


def generate_fx_experiment(
    n_users: int = 10000,
    experiment_start: datetime = datetime(2026, 1, 12),
    experiment_days: int = 7,
    history_days: int = 28,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Генерирует полный синтетический датасет для эксперимента FX fee reduction.

    :return: словарь с тремя DataFrame: 'users', 'transactions_experiment',
        'transactions_history'.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Пользователи + распределение по группам через двойное хеширование ---
    users_rows = []
    for i in range(n_users):
        user_id = f"u{i:06d}"
        country = _sample_country(rng)
        group = assign_group(user_id, EXPERIMENT_SALT)
        signup_date = experiment_start - timedelta(days=int(rng.integers(30, 720)))
        users_rows.append(
            {
                "user_id": user_id,
                "country": country,
                "pilot": 1 if group == "pilot" else 0,
                "signup_date": signup_date,
            }
        )
    df_users = pd.DataFrame(users_rows)

    # --- 2. Транзакции во время эксперимента ---
    history_start = experiment_start - timedelta(days=history_days)

    experiment_rows = []
    history_rows = []
    for row in users_rows:
        is_pilot = bool(row["pilot"])
        experiment_rows.extend(
            _generate_user_transactions(
                rng, row["user_id"], row["country"], is_pilot,
                experiment_start, experiment_days,
            )
        )
        # история ВСЕГДА генерируется с БАЗОВЫМ fee и БЕЗ лифта частоты —
        # до эксперимента разницы между будущими control/pilot быть не может
        history_rows.extend(
            _generate_user_transactions(
                rng, row["user_id"], row["country"], is_pilot=False,
                period_start=history_start, period_days=history_days,
            )
        )

    df_experiment = pd.DataFrame(experiment_rows)
    df_history = pd.DataFrame(history_rows)

    return {
        "users": df_users,
        "transactions_experiment": df_experiment,
        "transactions_history": df_history,
    }


def save_to_sqlite(tables: dict[str, pd.DataFrame], db_path: str = "ab_platform.db") -> None:
    """Сохраняет сгенерированные таблицы в SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    tables = generate_fx_experiment(n_users=10000)

    print("=== users ===")
    print(tables["users"].head())
    print(f"\nTotal users: {len(tables['users'])}")
    print(tables["users"]["pilot"].value_counts())
    print("\nBy country:")
    print(tables["users"]["country"].value_counts(normalize=True).round(3))

    print("\n=== transactions_experiment ===")
    print(tables["transactions_experiment"].head())
    print(f"Total experiment transactions: {len(tables['transactions_experiment'])}")

    print("\n=== transactions_history ===")
    print(f"Total history transactions: {len(tables['transactions_history'])}")

    save_to_sqlite(tables, db_path="ab_platform.db")
    print("\nSaved to ab_platform.db")