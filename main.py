# import bq_helper
import os
import json
# from pyspark_dist_explore import hist
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import nltk
# import sys
# import copy
# import time
import random
import pyspark
# from vertica_python import vertica
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import pandas as pd
# from pandas.plotting import scatter_matrix
# from pyspark.ml.fpm import FPGrowth
# from pyspark.sql.types import StructType, StructField, StringType
# from scipy import ndimage
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
# import nltk
import ast


# def get_important_nouns_from_a_string(string):
#    token = nltk.word_tokenize(string)
#    result = nltk.pos_tag(token)
#    filtered_result = list(filter(lambda x: x[1] == "NN" or x[1] == "NNP", result))
#    ans = list(map(lambda x: x[0], filtered_result))
#    return list(set(ans))


def list_dict_representation_to_actual_list_dict(string, dict_key):
   ans = ast.literal_eval(string)
   return ans[0][dict_key]
'''
HELPER FUNCTIONS
These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''


# Initialize a spark session.
def init_spark():
   spark = SparkSession \
       .builder \
       .appName("Python Spark SQL basic example") \
       .config("spark.some.config.option", "some-value") \
       .getOrCreate()
   return spark


# Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
   '''
   This function convert an RDD or a DataFrame into a CSV string
   '''
   a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
       .reduce(lambda x, y: os.linesep.join([x, y]))
   return a + os.linesep


def toCSVLine(data):
   '''
   Convert an RDD or a DataFrame into a CSV string
   '''
   if isinstance(data, RDD):
       return toCSVLineRDD(data)
   elif isinstance(data, DataFrame):
       return toCSVLineRDD(data.rdd)
   return None


import pandasql as ps



#-------------------using bigquery fetch data-----------------------------#
import bq_helper
    # import BigQueryHelper
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./key.json"

# bq_assistant = BigQueryHelper("patents-public-data", "google_patents_research")
# bq_assistant = BigQueryHelper("patents-public-data", "patents")
patents_helper = bq_helper.BigQueryHelper(
    active_project="patents-public-data",
    dataset_name="patents"
)
# table_names = bq_assistant.list_tables()
# print("\nTable names:\n")
# print(table_names)
# print("\nColumn names:\n")
# rows = bq_assistant.head("publications",selected_columns=" publication_number,application_number,country_code,"
#                                              "kind_code,application_kind,application_number_formatted,"
#                                              "pct_number,title_localized,abstract_localized,claims_localized,"
#                                              "publication_date,filing_date,grant_date,priority_date,priority_claim,"
#                                              "inventor,inventor_harmonized,assignee,assignee_harmonized,examiner,"
#                                              "uspc,ipc,cpc,cpc.code,fi,fterm,citation,entity_status,art_unit",num_rows=5000)
# query = '''
#     SELECT * FROM `patents-public-data.patents.publications` LIMIT 1000
# '''
patents_to_analyse = ["US-1989786-A", "US-1989925-A",
                      "US-2001040298-A1", "US-2002037896-A1", "US-2002055159-A1", "US-2002095050-A1",
                      "US-2003052383-A1", "US-2003099932-A1", "US-2003052813-A1", "US-1705778-A"]

input_patents_str = '(' + str(patents_to_analyse)[1:-1] + ')'

query1 = '''
SELECT DISTINCT cpc_code
FROM (
    SELECT
    publication_number,
    c.code as cpc_code
    FROM `patents-public-data.patents.publications`,UNNEST(cpc) as c
    where publication_number in {}
)
'''.format(input_patents_str)

all_cpcs_df = patents_helper.query_to_pandas(query1)
all_cpcs_df.to_csv("all_cpcs.csv")
# print(all_cpcs_df.head())
all_cpcs_str = '(' + str(list(all_cpcs_df.cpc_code.values))[1:-1] + ')'
with open("all_cpcs.txt", "w") as text_file:
    text_file.write(all_cpcs_str)



exit(3)
# print(type(rows))
# rows = bq_assistant.head("publications", selected_columns="publication_number,title,"
#                                                                  "title_translated,abstract,abstract_translated,"
#                                                                  "cpc,cpc_low,cpc_inventive_low,"
#                                                                  "top_terms,similar,url,country,"
#                                                                  "publication_description,cited_by,"
#                                                                  "embedding_v1", num_rows=5000)

# rows.to_csv("publications_bq.csv")
# exit(3)
# df = pd.DataFrame(rows)
# print(type(df))
# df = ps.sqldf(query1,locals())
# df.show()
exit(3)
# #------------creating dataframe---------------------#
spark = init_spark()
df1 = spark.read.csv("publications.csv", header=True, mode="DROPMALFORMED")

df1 = pd.read_csv("publications.csv")

def getkey(x):
    # list = [n.strip() for n in x]
    list = ast.literal_eval(x)
    print(type(list))
    for d in list:
        # for k in d.keys():
        if d == 'cpc.code':
            return d.get(d)
def tryJson(l):
    y = json.loads(l)
    print(y)
    for d in y:
        print(d["code"])
l = "[{'code': 'C07F1/08', 'inventive': True}]"

# tryJson(l)
a = getkey(l)
print(a)
exit(3)

def get_ctree(d):
    return d.get("tree")
# data_frame = spark.createDataFrame(data,samplingRatio=0.2)
# df1 = df1.filter(df1["cpc"]).select("cpc")
df = df1[["cpc"]]
# df = df.iloc[[1]]
print(type(df))
# print(df.head())

p =df.apply(lambda x: pd.Series(getkey(x)))
# data = data.map(lambda x: list_dict_representation_to_actual_list_dict(x.cpc, "code"))
# print(data.take(3))
print(p.head())
exit(3)

# data_filtered = data_from_csv.filter(lambda x: len(x.cpc) > 2)
# print(data_filtered.count())
# data_frame = data.toDF()

data_frame = spark.createDataFrame(data,samplingRatio=0.2)
# data_frame.show()

data_frame.createOrReplaceTempView("data_frame")
data_frame.show()
exit(3)

# df = vertica.select_dataframe(query))
# df.show()


# print("\n",input_patents_str)

# query = '''
# #standardSQL
# SELECT DISTINCT cpc_code
# FROM (
#     SELECT
#     publication_number,
#     c.code as cpc_code
#
#     FROM `patents-public-data.patents.publications`
#     ,UNNEST(cpc) as c
#
#     where publication_number in {}
# )
# '''.format(input_patents_str)
#
# all_cpcs = bq_assistant.query_to_pandas(query=query)
# # Convert into helper string for limiting CPCs
# all_cpcs_str = '(' + str(list(all_cpcs.cpc_code.values))[1:-1] + ')'
# # print(all_cpcs_str)
#
# # Get sample of patents not in our input set, but sharing at least 1 cpc.
# query = '''
# SELECT DISTINCT publication_number
# FROM `patents-public-data.patents.publications`
# ,UNNEST(cpc) as cpc
# where publication_number not in {}
# and cpc.code in {}
# and rand() < 0.2
# limit 100
# '''.format(input_patents_str, all_cpcs_str)
# shared_cpc = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=6)
# shared_cpc.loc[:, 'source'] = 'shared_cpc'
# print(shared_cpc.head())