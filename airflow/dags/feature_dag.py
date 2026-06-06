"""
airflow/dags/feature_dag.py
==============================
Airflow DAG — runs the feature pipeline every hour.
Airflow handles retries, alerting, and dependency management.
GitHub Actions deploys the code; Airflow schedules it.

This DAG assumes your code lives at /opt/airflow/project/
(set by the Docker volume in docker-compose.yml).

To deploy: copy this file to your Airflow dags/ folder.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ─── Default args ─────────────────────────────────────────────────────────────
default_args = {
    "owner":            "aqi-predictor",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,      # set True and add email in Airflow config
    "email_on_retry":   False,
}

# ─── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id          = "aqi_feature_pipeline",
    description     = "Fetch AQI + weather data and store features every hour",
    schedule_interval = "@hourly",
    default_args    = default_args,
    catchup         = False,
    tags            = ["aqi", "feature-pipeline", "karachi"],
) as dag:

    # ── Task 1: Health check ─────────────────────────────────────────────
    health_check = BashOperator(
        task_id  = "health_check",
        bash_command = """
            echo "Starting AQI Feature Pipeline at $(date -u)"
            python -c "import requests; r = requests.get('https://api.waqi.info/feed/@karachi/?token={{ var.value.AQICN_TOKEN }}'); print('AQICN reachable:', r.status_code == 200)"
        """,
    )

    # ── Task 2: Run feature pipeline ─────────────────────────────────────
    run_feature_pipeline = BashOperator(
        task_id      = "run_feature_pipeline",
        bash_command = """
            cd /opt/airflow/project/feature_pipeline
            python run_pipeline.py
        """,
        env = {
            "AQICN_TOKEN":       "{{ var.value.AQICN_TOKEN }}",
            "OWM_API_KEY":       "{{ var.value.OWM_API_KEY }}",
            "HOPSWORKS_API_KEY": "{{ var.value.HOPSWORKS_API_KEY }}",
            "HOPSWORKS_PROJECT": "{{ var.value.HOPSWORKS_PROJECT }}",
        },
    )

    # ── Task 3: Log completion ────────────────────────────────────────────
    log_completion = BashOperator(
        task_id      = "log_completion",
        bash_command = 'echo "✅ Feature pipeline completed at $(date -u)"',
    )

    # ── Task ordering ─────────────────────────────────────────────────────
    health_check >> run_feature_pipeline >> log_completion
