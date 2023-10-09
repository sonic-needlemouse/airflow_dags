"""One model training with s3 bucket storage"""
import json
import logging
import pickle
from datetime import datetime, timedelta
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
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
dag = DAG(
    dag_id="prepare_the_data_train_save",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)
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


def init() -> None:
    """Initiation of the pipeline"""
    _LOG.info("Train pipeline started.")


def get_data_from_postgres() -> None:
    """Conncetion and get data from psql"""

    _LOG.info("CONNECTOR INITIATION STARTED")
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()
    _LOG.info("CONNECTOR INITIATION FINISHED")
    data = pd.read_sql_query(f"SELECT * FROM {POSTGRES_TABLE_NAME}", con)
    _LOG.info("DATA FRAME CREATED FROM DATABASE")
    s3_hook = S3Hook("s3_connector")
    _LOG.info("S3 CONNECTOR CREATED")
    session = s3_hook.get_session("ru-central1-a")
    _LOG.info("S3 SESSION STARTED")
    resource = session.resource("s3")
    _LOG.info("S3 RESOURCE CREATED")
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, PATH).put(Body=pickle_byte_obj)
    _LOG.info("Data download finished")


def prepare_data() -> None:
    """Train, Test split"""

    s3_hook = S3Hook("s3_connector")
    file = s3_hook.download_file(key=PATH, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)
    session = s3_hook.get_session("ru-central")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"dataset/{name}.pkl").put(Body=pickle_byte_obj)
    _LOG.info("Data preparation finished")


def train_model() -> None:
    """Training of the rf model"""
    s3_hook = S3Hook("s3_connector")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"dataset/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    model = RandomForestRegressor()
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    result = {}
    result["r2_score"] = r2_score(data["y_test"], prediction)
    result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
    result["mae"] = median_absolute_error(data["y_test"], prediction)

    date = datetime.now().strftime("%Y_%m_%d_%H")
    session = s3_hook.get_session("ru-central1-a")
    resource = session.resource("s3")
    json_byte_object = json.dumps(result)
    resource.Object(BUCKET, f"results/{date}.json").put(Body=json_byte_object)

    _LOG.info("Model training finished")


def save_results() -> None:
    """Save the results of the training"""
    _LOG.info("Success.")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(
    task_id="get_data", python_callable=get_data_from_postgres, dag=dag
)

task_prepare_data = PythonOperator(
    task_id="prepare_data", python_callable=prepare_data, dag=dag
)

task_train_model = PythonOperator(
    task_id="train_model", python_callable=train_model, dag=dag
)

task_save_results = PythonOperator(
    task_id="save_results", python_callable=save_results, dag=dag
)

# DAGs Architecture

task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
