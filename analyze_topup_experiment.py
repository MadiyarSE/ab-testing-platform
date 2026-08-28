"""
Analysis of the "Top-up Limit Increase" experiment.

Runs on ALL data accumulated so far (day 1..N). Each run appends one row
per metric to experiment_results_log, so the dashboard can plot how
p-values and deltas evolve day by day (peeking problem illustration).

Primary metric: average top-up amount per user (ratio metric -> linearization).
Guardrail metrics: fraud rate and chargeback rate (proportions, z-test).
Multiple testing across the 3 metrics is corrected using Holm's method.
"""

import sqlite3
import sys
import os
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_engine.core import check_linearization, get_proportions_ztest_pvalue, holm_correction

DB_PATH = "ab_platform.db"


def get_current_day(conn):
    try:
        row = conn.execute(
            "SELECT current_day FROM experiment_state WHERE experiment_name = 'topup'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def ensure_log_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results_log (
            experiment_name TEXT,
            day INTEGER,
            run_timestamp TEXT,
            metric_name TEXT,
            pvalue REAL,
            delta REAL,
            significant INTEGER,
            PRIMARY KEY (experiment_name, day, metric_name)
        )
    """)
    conn.commit()


def log_results(conn, day, metric_names, pvalues, deltas, rejected):
    ensure_log_table(conn)
    timestamp = datetime.utcnow().isoformat()
    for name, p, d, sig in zip(metric_names, pvalues, deltas, rejected):
        conn.execute(
            "INSERT OR REPLACE INTO experiment_results_log "
            "(experiment_name, day, run_timestamp, metric_name, pvalue, delta, significant) "
            "VALUES ('topup', ?, ?, ?, ?, ?, ?)",
            (day, timestamp, name, p, d, int(sig)),
        )
    conn.commit()


conn = sqlite3.connect(DB_PATH)

current_day = get_current_day(conn)
if current_day is None:
    print("No experiment_state found -- run generate_topup_day.py first.")
    conn.close()
    sys.exit(1)

users = pd.read_sql("SELECT * FROM users_topup", conn)
topups = pd.read_sql("SELECT * FROM topup_transactions", conn)

merged = topups.merge(users[["user_id", "pilot"]], on="user_id")

control = merged[merged["pilot"] == 0]
pilot = merged[merged["pilot"] == 1]

print(f"Day {current_day}: Control transactions: {len(control)}, Pilot transactions: {len(pilot)}")

user_lists = merged.groupby(["user_id", "pilot"])["amount"].apply(list).reset_index()
a_lists = user_lists[user_lists["pilot"] == 0]["amount"].values
b_lists = user_lists[user_lists["pilot"] == 1]["amount"].values

amount_pvalue, amount_delta = check_linearization(a_lists, b_lists)

print(f"\n=== Primary: avg top-up amount (linearization) ===")
print(f"pvalue = {amount_pvalue:.6f}, delta = {amount_delta:.4f}")

fraud_count_a = control["is_fraud"].sum()
fraud_nobs_a = len(control)
fraud_count_b = pilot["is_fraud"].sum()
fraud_nobs_b = len(pilot)

fraud_pvalue = get_proportions_ztest_pvalue(
    fraud_count_a, fraud_nobs_a, fraud_count_b, fraud_nobs_b
) if fraud_nobs_a > 0 and fraud_nobs_b > 0 else float("nan")

fraud_rate_a = fraud_count_a / fraud_nobs_a if fraud_nobs_a > 0 else 0
fraud_rate_b = fraud_count_b / fraud_nobs_b if fraud_nobs_b > 0 else 0
fraud_delta = fraud_rate_b - fraud_rate_a

print(f"\n=== Guardrail: fraud rate ===")
print(f"control = {fraud_rate_a:.4%}, pilot = {fraud_rate_b:.4%}, pvalue = {fraud_pvalue:.6f}")

cb_count_a = control["is_chargeback"].sum()
cb_nobs_a = len(control)
cb_count_b = pilot["is_chargeback"].sum()
cb_nobs_b = len(pilot)

cb_pvalue = get_proportions_ztest_pvalue(
    cb_count_a, cb_nobs_a, cb_count_b, cb_nobs_b
) if cb_nobs_a > 0 and cb_nobs_b > 0 else float("nan")

cb_rate_a = cb_count_a / cb_nobs_a if cb_nobs_a > 0 else 0
cb_rate_b = cb_count_b / cb_nobs_b if cb_nobs_b > 0 else 0
cb_delta = cb_rate_b - cb_rate_a

print(f"\n=== Guardrail: chargeback rate ===")
print(f"control = {cb_rate_a:.4%}, pilot = {cb_rate_b:.4%}, pvalue = {cb_pvalue:.6f}")

metric_names = ["avg_topup_amount", "fraud_rate", "chargeback_rate"]
pvalues = [amount_pvalue, fraud_pvalue, cb_pvalue]
deltas = [amount_delta, fraud_delta, cb_delta]

rejected = holm_correction(pvalues, alpha=0.05)

print(f"\n=== SUMMARY (Holm-corrected, day {current_day}) ===")
print(f"{'Metric':<20} {'p-value':<12} {'Significant?':<12}")
for name, p, r in zip(metric_names, pvalues, rejected):
    print(f"{name:<20} {p:<12.6f} {str(r):<12}")

log_results(conn, current_day, metric_names, pvalues, deltas, rejected)
print(f"\nLogged results for day {current_day} to experiment_results_log.")

conn.close()
