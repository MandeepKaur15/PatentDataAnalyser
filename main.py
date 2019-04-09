import bq_helper
import os
from bq_helper import BigQueryHelper
from pyspark_dist_explore import hist
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
import numpy as np
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from nlp_functions import get_important_nouns_from_a_string, list_dict_representation_to_actual_list_dict
from nlp_functions import get_data_tfidf_weights_and_vectorizer_from_corpus
from pandas.plotting import scatter_matrix
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import lit, desc, size, max, col, abs, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, StringType
from scipy import ndimage
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package


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




os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./My First Project-0ad1d9baf9e6.json"

min_support = 0.1
confidence = 0.3
bq_assistant = BigQueryHelper("patents-public-data", "patents")
table_names = bq_assistant.list_tables()
print("\nTable names:\n")
print(table_names)
# rows = bq_assistant.head("publications_201809", selected_columns="publication_number,"
#                                                                  "country_code,title_localized,abstract_localized,"
#                                                                  "description_localized, publication_date, "
#                                                                  "assignee,inventor", num_rows=1000)
# print("\nColumn Names:\n")
# print(list(rows))
# rows.to_csv("publications_201809.csv")
spark = init_spark()
data_from_csv = spark.read.csv("publications_201809.csv", header=True).rdd
data_from_csv = spark.sparkContext.parallelize(data_from_csv.collect())

filtered_data = data_from_csv.filter(lambda x: len(x.assignee) > 2 and len(x.abstract_localized) > 2 and len(x.inventor) > 2)

abstract_rdd = filtered_data.map(lambda x: (x.publication_number, list_dict_representation_to_actual_list_dict(x.abstract_localized, "text")))
full_list = abstract_rdd.collect()
full_list = list(map(lambda x: x[1], full_list))
# print(full_list)
result = get_data_tfidf_weights_and_vectorizer_from_corpus(full_list)
arr = result[0].toarray()
v = result[1]
for inside_arr in arr:
    result = np.where(inside_arr > 0.2)
    for r in result:
        for c in r:
            print(v[c])
# print(v)
# print(arr.toarray().shape)
# for i in arr.toarray():
#     for j in i:
#         print(j)

# words_rdd = filtered_data.map(lambda x: (x.publication_number,
#                                          get_important_nouns_from_a_string(list_dict_representation_to_actual_list_dict(x.abstract_localized, "text"))))
# words_dataframe = words_rdd.toDF()
# words_dataframe = words_dataframe.selectExpr("_1 as Publication_number", "_2 as Important_words")
# words_dataframe.show()
# fpgrowth = FPGrowth(itemsCol="Important_words", minSupport=min_support, minConfidence=confidence)
# model = fpgrowth.fit(words_dataframe)
# my_dataframe = model.freqItemsets.orderBy("freq", ascending=False)
# my_dataframe.show()


# filtered_data_dataframe = filtered_data.toDF()
#
# inventors = filtered_data_dataframe.groupBy('assignee').count().sort(col("count").desc()).show()


# hist(ax, inventors, bins=20, color=['red'])
# data_dataframe = filtered_data.toDF()
# data_dataframe.repartition(1).write.save(path='data.csv',
#         format='csv',
#         mode='overwrite',
#         header='true',
#         inferschema="true",
#         sep=",")
# lines.saveAsTextFile('data.csv')
# filtered_data_dataframe.toPandas().to_csv("data.csv")
# rows_needed_for_analysis =


# print(type(rows))
#
# print(list(rows))
# assignees = rows["assignee"].value_counts()
# inventors = rows["inventor"].value_counts()

# ToDo:Delete rows with empty inventors or assignees.


# def top_n_assignees(n):
#     return assignees.nlargest(n)
#
#
# def top_n_inventors(n):
#     return inventors.nlargest(n)
#
#
# top_assignees = top_n_assignees(5)
# top_inventors = top_n_inventors(5)
# top_assignees.plot()
# plt.show()
# top_inventors.plot()
# plt.show()



