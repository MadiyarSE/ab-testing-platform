"""
DAG для оркестрации A/B-платформы:
generate_data -> run_dbt -> run_analysis
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
    description="FX Fee Reduction experiment: generate -> transform -> analyze",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 8, 20),
    catchup=False,
    tags=["ab-testing", "fx-experiment"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 data_generator/generate_fx_experiment.py"
        ),
    )

    run_dbt = BashOperator(
        task_id="run_dbt",
        bash_command=(
            f"cd {PROJECT_ROOT}/ab_platform_dbt && "
            f"dbt run --profiles-dir /opt/ab-testing-platform/.dbt_profiles"
        ),
    )

    run_analysis = BashOperator(
        task_id="run_analysis",
        bash_command=(
            f"cd {PROJECT_ROOT} && python3 analyze_fx_experiment_v2.py"
        ),
    )

    generate_data >> run_dbt >> run_analysis
