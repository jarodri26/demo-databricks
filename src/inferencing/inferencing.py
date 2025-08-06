# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import datetime
import mlflow
import pandas as pd
import random
import warnings

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame
from typing import Dict, Any, List
from pandas.api.types import is_integer_dtype
from mlflow import MlflowClient

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar Variáveis de Ambiente

# COMMAND ----------

# MAGIC %run ../../env $conf_filepath="../../conf/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar Dataset de Inferência
# MAGIC - Para propósito de demonstração, criaremos nossos próprios dados

# COMMAND ----------

class TaxiTripMockDataGenerator:
    """
    A class to generate mock data for taxi trips based on an input dataset.
    The sample dataset is static, so we generate fake data for demonstration purposes.

    Attributes:
        input_data (DataFrame): The input dataset containing taxi trips history.
    """

    def __init__(self, input_data: DataFrame, expected_features: List):
        self.input_data = input_data
        self.expected_features = expected_features

    def generate_mock_data(self, num_trips: int = 1) -> pd.DataFrame:
        mock_data_list = []
        for _ in range(num_trips):
            mock_data = {}
            for feature in self.expected_features:
                mock_data[feature] = self.select_random_value(feature)
            mock_data_list.append(mock_data)
        return pd.DataFrame(mock_data_list)

    def select_random_value(self, feature: str) -> Any:
        "Select a random value for a given feature from the input dataset."
        values = self.input_data.select(feature).rdd.flatMap(lambda x: x).collect()
        return random.choice(values)

# COMMAND ----------

fe = FeatureEngineeringClient()
ft_nyc_taxi_trips = fe.read_table(name=f'{vars["FEATURE_STORE_CATALOG"]}.{vars["FEATURE_STORE_SCHEMA"]}.{vars["FEATURE_STORE"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar o Modelo do MLflow Model Registry

# COMMAND ----------

model_name = f'{vars["MODEL_REGISTRY_CATALOG"]}.{vars["MODEL_REGISTRY_SCHEMA"]}.{vars["MODEL_NAME"]}'
mlflow_client = MlflowClient()

try:
    # Get model by its alias
    champion_version_uri = f'models:/{model_name}@champion'
    model = mlflow.pyfunc.load_model(champion_version_uri)
    champion_version = mlflow_client.get_model_version_by_alias(
                name= model_name,
                alias="champion"
                )

    print(f"Loading Champion Model: {champion_version.name}, version {champion_version.version}.")
    
except OSError as e: 
    warnings.warn(e.args[0] + ". Loading Challenger Model available in latest run.")
    challenger_version_uri = f'models:/{model_name}@challenger'
    model = mlflow.pyfunc.load_model(challenger_version_uri)
    challenger_version = mlflow_client.get_model_version_by_alias(
                name= model_name,
                alias="challenger"
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gerar Dados Mock

# COMMAND ----------

# Load feature names for mock data generation
loaded_model_features = model._model_impl.python_model.model.get_booster().feature_names 

# Generate mock data
mock_data_generator = TaxiTripMockDataGenerator(ft_nyc_taxi_trips, expected_features=loaded_model_features)
mock_taxi_trips = mock_data_generator.generate_mock_data(num_trips=2)

# Casting to int32
for column in mock_taxi_trips.columns:
    if is_integer_dtype(mock_taxi_trips[column]) == True:
        mock_taxi_trips[column] = mock_taxi_trips[column].astype("int32")

# COMMAND ----------

# MAGIC %md
# MAGIC # Gerar Previsões em Batch

# COMMAND ----------

mock_taxi_trips["predicted_fare"] = model.predict(mock_taxi_trips)
mock_taxi_trips["prediction_timestamp"] = datetime.datetime.now()
mock_taxi_trips["model_uri"] = champion_version_uri
mock_taxi_trips["model_run_id"] = model.metadata.run_id
mock_taxi_trips

# COMMAND ----------

# MAGIC %md
# MAGIC # Salvar Previsões

# COMMAND ----------

(
    spark.createDataFrame(mock_taxi_trips)
    .write.mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(f'{vars["FEATURE_STORE_CATALOG"]}.{vars["FEATURE_STORE_SCHEMA"]}.fare_batch_predictions')
)
