"""DAG for each of the model automatically"""
import json
import logging
from datetime import datetime, timedelta
from typing import Literal
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "XXX",
    "email": "example@gmail.com",
    "email_on_failure": True,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = "airflow-bucket-1"
PATH = "datasets/california_housing.pkl"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"


POSTGRES_TABLE_NAME = "california_housing"


models = dict(
    zip(
        ["rf", "lr", "hgb"],
        [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()],
    )
)


def create_dag(dag_id: str, m_name: Literal["rf", "lr", "hgb"]):
    """DAG creation depends on model"""

    def init() -> dict[str, any]:
        """Initiation of the pipeline"""
        metrics = {}
        metrics["model"] = m_name
        metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
        return metrics

    def get_data_from_postgres(**kwargs) -> dict[str, any]:
        """Conncetion and get data from psql"""
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")

        _LOG.info("CONNECTOR INITIATION STARTED!!!!!!!!")

        return metrics

    def prepare_data(**kwargs) -> dict[str, any]:
        """Data preparation"""
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="get_data")

        return metrics

    def train_model(**kwargs) -> dict[str, any]:
        """Training of the model"""
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="prepare_data")

        m_name = metrics["model"]

        s3_hook = S3Hook("s3_connector")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
            data[name] = pd.read_pickle(file)

        model = models[m_name]
        metrics["train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])
        metrics["train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

        # result = {}
        metrics["r2_score"] = r2_score(data["y_test"], prediction)
        metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        metrics["mae"] = median_absolute_error(data["y_test"], prediction)

        return metrics

    def save_results(**kwargs) -> None:
        """Save results to the JSON to s3 bucket"""
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="train_model")

        s3_hook = S3Hook("s3_connector")

        date = datetime.now().strftime("%Y_%m_%d_%H")
        session = s3_hook.get_session("ru-central1-a")
        resource = session.resource("s3")
        # json_byte_object = json.dumps(result)
        json_byte_object = json.dumps(metrics)
        resource.Object(BUCKET, f"results/{metrics['model']}_{date}.json").put(
            Body=json_byte_object
        )
        # resource.Object(BUCKET, f"result/{metrics}_{date}.json").put(Body=json_byte_object)

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        tags=["mlops"],
        default_args=DEFAULT_ARGS,
    )

    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

        task_get_data = PythonOperator(
            task_id="get_data",
            python_callable=get_data_from_postgres,
            dag=dag,
            provide_context=True,
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
            dag=dag,
            provide_context=True,
        )

        task_train_model = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            dag=dag,
            provide_context=True,
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
            dag=dag,
            provide_context=True,
        )

        # DAGs Architecture

        (
            task_init
            >> task_get_data
            >> task_prepare_data
            >> task_train_model
            >> task_save_results
        )


for model_name in models.keys():
    create_dag(f"{model_name}_train", model_name)
