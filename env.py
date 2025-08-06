# Databricks notebook source
# MAGIC %md
# MAGIC # Separação de Ambientes com Databricks Widgets
# MAGIC - Agora utilizaremos os Databricks Widgets para como uma forma de parametrização dos notebooks para separação de ambientes. 
# MAGIC - Nesta etapa as variáveis serão extraídas de arquivos `.yml`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import os
import warnings
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Função de Carregamento de Configurações

# COMMAND ----------

def read_yaml_file(file_path):
    """
    Reads a YAML file and returns its content as a Python dictionary.

    Parameters:
        file_path (str): The path to the YAML file.

    Returns:
        dict: Content of the YAML file.

    Raises:
        Exception: If there is an error loading or parsing the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)  # Parse and load YAML content
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error processing the YAML file: {file_path}. {e}")

def load_configuration(filepath):
    """
    Determines which configuration file (dev or prd) to load based on file existence.

    Parameters:
        filepath (str): Base directory path where configuration files are located.

    Returns:
        dict: Content of the selected YAML configuration file.

    Warnings:
        Emits a warning if the dev configuration file is missing and uses prod as a fallback.
    """
    # Paths to the dev and prod configuration files
    dev_file = os.path.join(filepath, "dev", "environment.yml")
    prod_file = os.path.join(filepath, "prd", "environment.yml")

    # Choose which file to load: prefer dev, fallback to prod
    file_to_load = dev_file if os.path.isfile(dev_file) else prod_file

    if file_to_load == prod_file:
        warnings.warn(f"Dev configuration not found. Using prod configuration: {prod_file}.", UserWarning)

    # Load and return the content of the selected file
    return read_yaml_file(file_to_load)

# COMMAND ----------

dbutils.widgets.text("conf_filepath", 'conf/')
config = load_configuration(dbutils.widgets.get("conf_filepath"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar de Widgets

# COMMAND ----------

if 'variables' not in config:
    raise Exception("The configuration does not contain the key 'variables'.")

for key, value in config['variables'].items():
    # Get the default value for each variable, or an empty string if not specified
    default = value.get("default", "")
    # Create a Databricks widget for each variable with its default value
    dbutils.widgets.text(str(key), str(default))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurar de Variáveis no Spark

# COMMAND ----------

vars = {variable: dbutils.widgets.get(variable) for variable in config["variables"].keys()}

for key, value in vars.items():
    if value is None:
        raise Exception(f'{key} cant be none')
    spark.conf.set(f'vars.{key}', value)

# COMMAND ----------

test = f'{vars["FEATURE_STORE_CATALOG"]}.{vars["FEATURE_STORE_SCHEMA"]}.{vars["FEATURE_STORE"]}'
print(test)
