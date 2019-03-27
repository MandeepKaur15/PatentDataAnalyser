import bq_helper
import os
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
from scipy import ndimage
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./My First Project-0ad1d9baf9e6.json"

bq_assistant = BigQueryHelper("patents-public-data", "patents")
table_names = bq_assistant.list_tables()
print(table_names)
rows = bq_assistant.head("publications_201809", num_rows=500)

assignees = rows["assignee"].value_counts()


def top_n_assignees(n):
    return assignees.nlargest(n)


print(top_n_assignees(5))
top_n_assignees(5).hist()
plt.show()
