"""
airflow/dags/training_dag.py
================================
Airflow DAG — runs the training pipeline daily at midnight.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner":            "aqi-predictor",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id            = "aqi_training_pipeline",
    description       = "Daily model training: fetch features, train, register best model",
    schedule_interval = "0 0 * * *",    # daily at midnight UTC
    default_args      = default_args,
    catchup           = False,
    tags              = ["aqi", "training-pipeline", "karachi"],
) as dag:

    check_features = BashOperator(
        task_id      = "check_features_available",
        bash_command = """
            python -c "
import sys; sys.path.insert(0, '/opt/airflow/project/feature_pipeline')
from store_features import read_features
df = read_features()
print(f'Available rows: {len(df)}')
assert len(df) >= 30, f'Need at least 30 rows for training, got {len(df)}'
print('✅ Enough data for training')
"
        """,
        env = {
            "HOPSWORKS_API_KEY": "{{ var.value.HOPSWORKS_API_KEY }}",
            "HOPSWORKS_PROJECT": "{{ var.value.HOPSWORKS_PROJECT }}",
        },
    )

    run_training = BashOperator(
        task_id      = "run_training_pipeline",
        bash_command = """
            cd /opt/airflow/project/training_pipeline
            python train.py
        """,
        env = {
            "HOPSWORKS_API_KEY": "{{ var.value.HOPSWORKS_API_KEY }}",
            "HOPSWORKS_PROJECT": "{{ var.value.HOPSWORKS_PROJECT }}",
        },
    )

    log_completion = BashOperator(
        task_id      = "log_completion",
        bash_command = 'echo "✅ Training pipeline completed at $(date -u)"',
    )

    check_features >> run_training >> log_completion
