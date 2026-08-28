"""
DAG for A/B Platform Orchestration:
generate_data (fx: one-shot, topup: day-by-day) -> run_dbt -> run_analysis (fx + topup)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/ab-testing-platform"

default_args = {
    "owner": "madiyar",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ab_platform_pipeline",
    description="FX Fee Reduction (one-shot) + Top-up Limit Increase (day-by-day) experiments",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 8, 20),
    catchup=False,
    tags=["ab-testing", "fx-experiment", "topup-experiment"],
) as dag:

    generate_fx_data = BashOperator(
        task_id="generate_fx_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 data_generator/generate_fx_experiment.py"
        ),
    )

    generate_topup_data = BashOperator(
        task_id="generate_topup_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 data_generator/generate_topup_day.py"
        ),
    )

    run_dbt = BashOperator(
        task_id="run_dbt",
        bash_command=(
            f"cd {PROJECT_ROOT}/ab_platform_dbt && "
            f"dbt run --profiles-dir /opt/ab-testing-platform/.dbt_profiles --target docker"
        ),
    )

    run_fx_analysis = BashOperator(
        task_id="run_fx_analysis",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 analyze_fx_experiment_v2.py"
        ),
    )

    run_topup_analysis = BashOperator(
        task_id="run_topup_analysis",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 analyze_topup_experiment.py"
        ),
    )

    [generate_fx_data, generate_topup_data] >> run_dbt >> [run_fx_analysis, run_topup_analysis]