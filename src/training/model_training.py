# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports Necessários

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install xgboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd

from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Union, Dict
from xgboost import XGBRegressor
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow import MlflowClient
from databricks.feature_engineering import FeatureLookup



mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento de Funções Helper para Visualizações

# COMMAND ----------

# MAGIC %run ../utils/visuals

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar as Configurações

# COMMAND ----------

# MAGIC %run ../../env $conf_filepath="../../conf/"

# COMMAND ----------

test = f'{vars["FEATURE_STORE_CATALOG"]}.{vars["FEATURE_STORE_SCHEMA"]}.{vars["FEATURE_STORE"]}'
print (test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Função Helper para Cálculo de Métricas

# COMMAND ----------

def _evaluate_predictions(actual: Union[np.array, pd.Series], predicted: Union[np.array, pd.Series]) -> Dict:
    "Calculates various regression metrics. To extend it to include other metrics, import and add to list"
    metrics = {
        "r2": r2_score(actual, predicted),
        "mean_absolute_error": mean_absolute_error(actual, predicted),
        "root_mean_squared_error": mean_squared_error(actual, predicted, squared=True), 
    }
    return metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar Dataset de Treino

# COMMAND ----------


# Define the SQL query to process the NYC taxi trips dataset
query = f"""
    SELECT 
    -- Generate a unique key for each trip using SHA-256 hash
    sha2(concat(dropoff_zip, '-', pickup_zip, '-', tpep_pickup_datetime), 256) as trip_sk
    , dropoff_zip -- ZIP code of the drop-off location
    , pickup_zip -- ZIP code of the pick-up location
    , date_part('month', tpep_pickup_datetime) as trip_month -- Month of the pick-up timestamp
    , date_part('day', tpep_pickup_datetime) as trip_day -- Day of the pick-up timestamp
    , date_part('dayofweek', tpep_pickup_datetime) as trip_dow -- Day of the week of the pick-up timestamp
    , date_part('hour', tpep_pickup_datetime) as trip_pickup_hour -- Hour of the pick-up timestamp
    , datediff(minute, tpep_pickup_datetime, tpep_dropoff_datetime) as trip_duration -- Trip duration in minutes
    , trip_distance -- Distance traveled during the trip
    , fare_amount -- Fare amount for the trip
    FROM samples.nyctaxi.trips -- Source dataset
"""

# Execute the SQL query in Spark and store the result as a Spark DataFrame
ft_nyc_taxi_trips = spark.sql(query)

# Select only the unique trip key (trip_sk) and the fare amount (fare_amount) for the spine table
spine_table = ft_nyc_taxi_trips.select("trip_sk", "fare_amount")

# Define feature lookups for the Feature Store
model_feature_lookups = [
    FeatureLookup(
        table_name=f'{vars["FEATURE_STORE_CATALOG"]}.{vars["FEATURE_STORE_SCHEMA"]}.{vars["FEATURE_STORE"]}',
        lookup_key=["trip_sk"], # Key to fetch the features
    )
]

fe = FeatureEngineeringClient()

# Create the training set by combining the spine table with the feature lookups
training_set = fe.create_training_set(
    df=spine_table, # Spine table containing keys and labels
    feature_lookups=model_feature_lookups, # Feature lookup configuration
    label='fare_amount', # Target column (label)
)

# Load the training set as a Pandas DataFrame
training_pd = training_set.load_df().toPandas()

# Display the first two rows of the resulting Pandas DataFrame
training_pd.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dividir o Dataset

# COMMAND ----------

from sklearn.model_selection import train_test_split

target="fare_amount"
training_pd = training_pd.set_index("trip_sk")
X_train, X_test, y_train, y_test = train_test_split(training_pd.drop(target, axis=1), training_pd[target], test_size=.2)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar Modelo Customizado do MLflow
# MAGIC

# COMMAND ----------

class CustomPredictor(mlflow.pyfunc.PythonModel):
    """Custom MLflow Model.    

    Parameters:
        model: str or mlflow.models.Model
            The trained model to be loaded. It can be a  model URI or an MLflow Model object.
            
    """
    def __init__(self, model):
        self.model = model
        # Other objects can be stored 

    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        return predictions
        

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinando um Modelo e Registrando com MLflow 

# COMMAND ----------

model_name = f'{vars["MODEL_REGISTRY_CATALOG"]}.{vars["MODEL_REGISTRY_SCHEMA"]}.{vars["MODEL_NAME"]}'

with mlflow.start_run(experiment_id=vars["MLFLOW_EXPERIMENT_ID"]) as run:
  
  xgb = XGBRegressor(objective='reg:squarederror')
  xgb.fit(X_train, y_train)

  predictions = xgb.predict(X_test)

  metrics = _evaluate_predictions(actual=y_test, predicted=predictions)
  [mlflow.log_metric(key, value) for key, value in metrics.items()]

  (figure := _generate_residuals_plot_figure(y_test, predictions))
  mlflow.log_figure(figure, f"residual_analysis.png")

  input_example = X_train[:5]
  
  predictor = CustomPredictor(model=xgb)
  mlflow.pyfunc.log_model(
      artifact_path=vars["MODEL_NAME"],
      python_model=predictor,
      input_example=input_example,
      registered_model_name=model_name
  )

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adicionando Aliases

# COMMAND ----------

# Get the latest version of the model
mlflow_client = MlflowClient()
results = mlflow_client.search_model_versions(f"name='{model_name}'")
latest_version = max(results, key=lambda x: x.version).version

# Assign the alias challenger to the last version of the model
mlflow_client.set_registered_model_alias(
    name= f'{model_name}',
    alias="challenger", 
    version=latest_version
    )
print(f"Assigning the challenger alias to the model version {latest_version}.")

# COMMAND ----------

# If the latest version it is the first version of the model, create a new model version to assign it with the champion alias.
if latest_version == '1':
    model_src = f"models:/{model_name}/{latest_version}"
    champion_version = mlflow_client.create_model_version(model_name, model_src)
    # Register the new model version as the champion model
    mlflow_client.set_registered_model_alias(
        name= f'{model_name}',
        alias="champion", 
        version=champion_version.version
        )
    print(f"Assigning the champion alias to the model version {champion_version.version}.")
