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
print(list(rows))
assignees = rows["assignee"].value_counts()
inventors = rows["inventor"].value_counts()

# ToDo:Delete rows with empty inventors or assignees.


def top_n_assignees(n):
    return assignees.nlargest(n)


def top_n_inventors(n):
    return inventors.nlargest(n)


top_assignees = top_n_assignees(5)
top_inventors = top_n_inventors(5)
top_assignees.plot()
plt.show()
top_inventors.plot()
plt.show()



