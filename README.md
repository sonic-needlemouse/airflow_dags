# airflow_DAGs
This repo contains DAG files for ML (based on tabular data).

1. DAG "just_print_the_message.py" simply prints the message.

2. DAG "one_model_training.py" get the data, prepare the data (train/test split), save the data to the s3 bucket, train the model, save the metrics to the JSON on s3 bucket.

3. DAG "dag_creation_depends_on_model.py" creates DAG based on model requirements using airflow XCOM.
