"""
Day-by-day incremental generator for the "Top-up Limit Increase" experiment.

Unlike generate_topup_experiment.py (one-shot full snapshot), this script
simulates ONE additional day each time it runs:
  - Day 1: creates the fixed user population (control/pilot assignment).
  - Every day: generates that day's transactions for existing users and
    appends them (does not overwrite previous days).
"""

import sqlite3
import uuid

import numpy as np
import pandas as pd

from generate_topup_experiment import (
    COUNTRY_PARAMS, EXPERIMENT_SALT, BASE_LIMIT, PILOT_LIMIT,
    BASE_FRAUD_RATE, PILOT_FRAUD_LIFT, BASE_CHARGEBACK_RATE, PILOT_CHARGEBACK_LIFT,
    assign_group, _sample_country,
)

DB_PATH = "ab_platform.db"
SEED = 42
N_USERS = 10000


def get_current_day(conn):
    """Reads how many days of the experiment have run so far (0 if not started)."""
    try:
        row = conn.execute(
            "SELECT current_day FROM experiment_state WHERE experiment_name = 'topup'"
        ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def set_current_day(conn, day):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS experiment_state "
        "(experiment_name TEXT PRIMARY KEY, current_day INTEGER)"
    )
    conn.execute(
        "INSERT INTO experiment_state (experiment_name, current_day) VALUES ('topup', ?) "
        "ON CONFLICT(experiment_name) DO UPDATE SET current_day = ?",
        (day, day),
    )
    conn.commit()


def create_users(rng, n_users):
    """Creates the fixed user population -- assigned once, never changes again."""
    from datetime import datetime, timedelta
    experiment_start = datetime(2026, 1, 12)
    rows = []
    for i in range(n_users):
        user_id = f"u{i:06d}"
        country = _sample_country(rng)
        group = assign_group(user_id, EXPERIMENT_SALT)
        sign_up_date = experiment_start - timedelta(days=int(rng.integers(30, 720)))
        rows.append({
            "user_id": user_id,
            "country": country,
            "pilot": 1 if group == "pilot" else 0,
            "sign_up_date": sign_up_date,
        })
    return pd.DataFrame(rows)


def generate_day_transactions(rng, users_df, day_number):
    """Generates one day's worth of top-up transactions for all users."""
    all_topups = []
    for _, row in users_df.iterrows():
        params = COUNTRY_PARAMS[row["country"]]
        # base_topup_freq was calibrated per WEEK -> divide by 7 for a daily rate
        n_topups = rng.poisson(lam=params["base_topup_freq"] / 7)
        if n_topups == 0:
            continue

        limit = PILOT_LIMIT if row["pilot"] else BASE_LIMIT
        amounts = rng.uniform(low=50, high=limit, size=n_topups)

        fraud_prob = BASE_FRAUD_RATE * (PILOT_FRAUD_LIFT if row["pilot"] else 1.0)
        is_fraud = rng.random(size=n_topups) < fraud_prob

        chargeback_prob = BASE_CHARGEBACK_RATE * (PILOT_CHARGEBACK_LIFT if row["pilot"] else 1.0)
        is_chargeback = rng.random(size=n_topups) < chargeback_prob

        df = pd.DataFrame({
            "amount": amounts,
            "is_fraud": is_fraud,
            "is_chargeback": is_chargeback,
        })
        df["user_id"] = row["user_id"]
        df["day"] = day_number
        df["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        all_topups.append(df)

    cols = ["amount", "is_fraud", "is_chargeback", "user_id", "day", "transaction_id"]
    if not all_topups:
        return pd.DataFrame(columns=cols)
    return pd.concat(all_topups, ignore_index=True)


def main():
    conn = sqlite3.connect(DB_PATH)
    current_day = get_current_day(conn)
    next_day = current_day + 1

    rng = np.random.default_rng(SEED + next_day)

    if current_day == 0:
        print("Day 0 -> creating user population...")
        users_df = create_users(rng, N_USERS)
        users_df.to_sql("users_topup", conn, if_exists="replace", index=False)
    else:
        users_df = pd.read_sql("SELECT * FROM users_topup", conn)

    print(f"Generating day {next_day} transactions...")
    day_df = generate_day_transactions(rng, users_df, next_day)
    day_df.to_sql("topup_transactions", conn, if_exists="append", index=False)

    set_current_day(conn, next_day)

    total = pd.read_sql("SELECT COUNT(*) as c FROM topup_transactions", conn).iloc[0]["c"]
    print(f"Day {next_day} done: {len(day_df)} new transactions, {total} total so far.")

    conn.close()


if __name__ == "__main__":
    main()