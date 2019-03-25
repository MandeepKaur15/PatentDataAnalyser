import bq_helper
import os
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./My First Project-0ad1d9baf9e6.json"

bq_assistant = BigQueryHelper("patents-public-data", "patents")
a = bq_assistant.list_tables()
rows = bq_assistant.head("publications_201710", num_rows=800)

print(type(rows))
