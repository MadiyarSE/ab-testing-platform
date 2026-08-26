"""
Synthetic data generator for the "Top-up Limit Increase" experiment.

Hypothesis: increasing the instant top-up limit from EUR 500 to EUR 1000
increases the average top-up amount per user, but may also increase
fraud rate and chargeback rate (guardrail metrics).
"""

import hashlib
import sqlite3
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Split system (double hashing) -- reused pattern from the FX experiment
# ---------------------------------------------------------------------------

def get_hash_modulo(value, modulo, salt):
    """Computes hash(value + salt) % modulo."""
    hash_value = int(hashlib.md5(str.encode(value + salt)).hexdigest(), 16)
    return hash_value % modulo


def assign_group(user_id, salt):
    """Second hashing step: assigns a user to control/pilot (50/50)."""
    bucket = get_hash_modulo(user_id, 2, salt)
    return "pilot" if bucket == 1 else "control"


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

COUNTRY_PARAMS = {
    "UK": {"weight": 0.35, "base_topup_freq": 1.5},
    "DE": {"weight": 0.30, "base_topup_freq": 1.2},
    "PL": {"weight": 0.20, "base_topup_freq": 1.0},
    "ES": {"weight": 0.15, "base_topup_freq": 1.1},
}

BASE_LIMIT = 500
PILOT_LIMIT = 1000
EXPERIMENT_SALT = "topup_limit_v1"

BASE_FRAUD_RATE = 0.01           # 1% baseline fraud rate
PILOT_FRAUD_LIFT = 1.3           # +30% relative fraud rate increase in pilot
BASE_CHARGEBACK_RATE = 0.005
PILOT_CHARGEBACK_LIFT = 1.2


def _generate_user_topups(rng, country, is_pilot):
    """Generates all top-up transactions for ONE user over one week.

    :return: pd.DataFrame with columns ['amount', 'is_fraud', 'is_chargeback'],
        one row per top-up transaction (can be empty if the user made no
        top-ups this week).
    """
    params = COUNTRY_PARAMS[country]
    n_topups = rng.poisson(lam=params["base_topup_freq"])

    limit = PILOT_LIMIT if is_pilot else BASE_LIMIT
    amounts = rng.uniform(low=50, high=limit, size=n_topups)

    fraud_prob = BASE_FRAUD_RATE * (PILOT_FRAUD_LIFT if is_pilot else 1.0)
    is_fraud = rng.random(size=n_topups) < fraud_prob

    chargeback_prob = BASE_CHARGEBACK_RATE * (PILOT_CHARGEBACK_LIFT if is_pilot else 1.0)
    is_chargeback = rng.random(size=n_topups) < chargeback_prob

    df = pd.DataFrame({
        "amount": amounts,
        "is_fraud": is_fraud,
        "is_chargeback": is_chargeback,
    })
    return df


def _sample_country(rng):
    """Randomly picks a country, weighted by COUNTRY_PARAMS[*]['weight']."""
    countries = list(COUNTRY_PARAMS.keys())
    weights = [COUNTRY_PARAMS[c]["weight"] for c in countries]
    return rng.choice(countries, p=weights)


def generate_topup_experiment(n_users=10000, seed=42, experiment_start=datetime(2026, 1, 12)):
    """Generates the full synthetic dataset for the top-up limit experiment.

    :return: dict with two DataFrames: 'users' and 'topup_transactions'.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Users, assigned to control/pilot via double hashing ---
    users_rows = []
    for i in range(n_users):
        user_id = f"u{i:06d}"
        country = _sample_country(rng)
        group = assign_group(user_id, EXPERIMENT_SALT)
        signup_date = experiment_start - timedelta(days=int(rng.integers(30, 720)))
        users_rows.append({
            "user_id": user_id,
            "country": country,
            "pilot": 1 if group == "pilot" else 0,
            "sign_up_date": signup_date,
        })
    df_users = pd.DataFrame(users_rows)

    # --- 2. Top-up transactions for each user ---
    all_topups = []
    for row in users_rows:
        user_topups = _generate_user_topups(rng, row["country"], bool(row["pilot"]))
        user_topups["user_id"] = row["user_id"]
        all_topups.append(user_topups)

    df_topups = pd.concat(all_topups, ignore_index=True)
    df_topups["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df_topups))]

    return {"users": df_users, "topup_transactions": df_topups}


def save_to_sqlite(tables, db_path="ab_platform.db"):
    """Saves the generated tables to SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    tables = generate_topup_experiment(n_users=10000)

    print("=== users ===")
    print(tables["users"]["pilot"].value_counts())

    print("\n=== topup_transactions ===")
    print(tables["topup_transactions"].head())
    print(f"\nTotal topup transactions: {len(tables['topup_transactions'])}")
    print(f"Overall fraud rate: {tables['topup_transactions']['is_fraud'].mean():.4f}")
    print(f"Overall chargeback rate: {tables['topup_transactions']['is_chargeback'].mean():.4f}")

    # breakdown by group -- sanity check that pilot > control, as designed
    merged = tables["topup_transactions"].merge(tables["users"], on="user_id")
    print("\n=== Fraud/chargeback rate by group ===")
    print(merged.groupby("pilot")[["is_fraud", "is_chargeback"]].mean())

    save_to_sqlite(tables, db_path="ab_platform.db")
    print("\nSaved to ab_platform.db (tables: users, topup_transactions)")
