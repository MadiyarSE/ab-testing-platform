FROM apache/airflow:2.10.4

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    dbt-core==1.4.9 \
    dbt-sqlite==1.4.0
