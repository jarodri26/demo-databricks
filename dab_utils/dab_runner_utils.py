# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import io
import zipfile
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import shutil
import os

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função Helper: extrair arquivos zip

# COMMAND ----------

def extract_zip(zipstr):
    with zipfile.ZipFile(io.BytesIO(zipstr)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield zipinfo.filename, thefile

# COMMAND ----------

# MAGIC %md
# MAGIC ### 

# COMMAND ----------

# Fetch Databricks Notebooks Context
notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = notebook_context.notebookPath().getOrElse(None) 
notebook_token = notebook_context.apiToken().getOrElse(None)
notebook_host = notebook_context.apiUrl().getOrElse(None)

# Connect to Databricks API
databricks_api_instance = WorkspaceClient(
    host=notebook_host,
    token=notebook_token
)

# Prepare tmp folder
write_path = '.tmp'
if os.path.exists(write_path):
    shutil.rmtree(write_path)

# Export Workspace Files
response = databricks_api_instance.workspace.export(
    '/'.join(notebook_path.split('/')[0:-1]),
    format=workspace.ExportFormat.AUTO
)

if not os.path.exists(write_path):
    os.makedirs(write_path)

notebook_content = base64.b64decode(response.content)

for (nb_path, notebook) in extract_zip(notebook_content):
    path = write_path + '/' + ('/'.join(nb_path.split('/')[1:]))
    if path.endswith('/'):
        if not os.path.exists(path):
            os.makedirs(path)
        continue
    try:
        write_file = path
        with open(write_file, 'wb') as f:
            try:
                f.write(notebook.read())
            except:
                print('could not write file: ', notebook)
    except Exception as e:
        print('could not write dir: ', nb_path)
        print(str(e))

