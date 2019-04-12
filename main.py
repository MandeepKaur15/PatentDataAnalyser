import bq_helper
import os
from bq_helper import BigQueryHelper
from pyspark_dist_explore import hist
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import nltk
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import pandas as pd
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




os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./new_key.json"

bq_assistant = BigQueryHelper("patents-public-data", "patents")


table_names = bq_assistant.list_tables()
print("\nTable names:\n")
print(table_names)

#
#
# query = '''
# #standardSQL
# SELECT
# RAND() as random, publication_number,application_number,country_code,kind_code,application_kind,application_number_formatted,pct_number,title_localized,abstract_localized,claims_localized,publication_date,filing_date,grant_date,priority_date,priority_claim,inventor,inventor_harmonized,assignee,assignee_harmonized,examiner,
# uspc,ipc,cpc,fi,fterm,citation,entity_status,art_unit
# FROM `patents-public-data.patents.publications_201809`
# ORDER BY random
# LIMIT 1
# '''
#
# result = bq_assistant.query_to_pandas(query=query)
#
# print(result)
# exit(3)




#
# rows = bq_assistant.head("publications_201809", num_rows=10000)
# rows.to_csv("data10k.csv")
# print(rows)
# exit(3)
# print("\nColumn Names:\n")
# print(list(rows))
# rows.to_csv("publications_201809.csv")
spark = init_spark()
data_from_csv = spark.read.csv("data10k.csv", header=True).rdd
data_filtered = data_from_csv.filter(lambda x: len(x.cpc) > 2)
print(data_filtered.count())
data_frame_filtered = data_filtered.toDF()
patents_to_analyse = ["US-1989786-A", "US-1989925-A",
                      "US-2001040298-A1", "US-2002037896-A1", "US-2002055159-A1", "US-2002095050-A1",
                      "US-2003052383-A1", "US-2003099932-A1", "US-2003052813-A1", "US-1705778-A"]

input_patents_str = '(' + str(patents_to_analyse)[1:-1] + ')'
data_frame_filtered.show()

query = '''
#standardSQL
SELECT DISTINCT cpc_code ,publication_number
FROM (
    SELECT
    publication_number,
    c.code as cpc_code

    FROM `patents-public-data.patents.publications_201809`
    ,UNNEST(cpc) as c

    where publication_number in {}
)
'''.format(input_patents_str)

all_cpcs = bq_assistant.query_to_pandas(query=query)
all_cpcs_str = '(' + str(list(all_cpcs.cpc_code.values))[1:-1] + ')'
print(all_cpcs)
# Convert into helper string for limiting CPCs
print(data_frame_filtered)

query = '''
SELECT DISTINCT publication_number
FROM `patents-public-data.patents.publications`
,UNNEST(cpc) as cpc
where publication_number not in {}
and cpc.code in {}
and rand() < 0.2
limit 100
'''.format(input_patents_str, all_cpcs_str)
shared_cpc = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=7)
# print(shared_cpc)
shared_cpc.loc[:, 'source'] = 'shared_cpc'
print(shared_cpc.head())

query = '''
SELECT DISTINCT publication_number
FROM `patents-public-data.patents.publications`
,UNNEST(cpc) as cpc
where publication_number not in {}
and cpc.code not in {}
and rand() < 0.2
limit 100
'''.format(input_patents_str, all_cpcs_str)
no_shared_cpc = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=6)
no_shared_cpc.loc[:, 'source'] = 'no_shared_cpc'
print(no_shared_cpc.head())

# Pull all the "similar patents" from Patents Research dataset.
# Each of our patents in the input list should have ~25 similar patents listed, so we get back 12*25 rows
query = '''
SELECT distinct
s.publication_number

FROM `patents-public-data.patents.publications` p
JOIN `patents-public-data.google_patents_research.publications` r
  on p.publication_number = r.publication_number
, UNNEST(similar) as s
where p.publication_number in {}
'''.format(input_patents_str)
similar = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=60)
similar.loc[:, 'source'] = 'similar_to_input'
print(len(similar))



# Lets constuct our dataframe by concatenating our input list, the close negatives and
# the list of "similar patents" according to the patents research table.
df = pd.DataFrame(patents_to_analyse, columns=['publication_number'])
df.loc[:, 'source'] = 'input'
df = pd.concat(
    [df, similar, shared_cpc, no_shared_cpc]).drop_duplicates('publication_number', keep='first')
df.source.value_counts()

all_patents_str = '(' + str(list(df.publication_number.unique()))[1:-1] + ')'
query = r'''
CREATE TEMPORARY FUNCTION convert_embedding_to_string(embedding ARRAY<FLOAT64>)
RETURNS STRING
LANGUAGE js AS """
let embedding_str = ''
for (i = 0; i < embedding.length; i++) {
  embedding_str += embedding[i].toFixed(6) + ',';
}
return embedding_str
""";

SELECT
publication_number,
convert_embedding_to_string(embedding_v1) embedding
FROM `patents-public-data.google_patents_research.publications`
where publication_number in %s
''' % (all_patents_str)


results = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=40).drop_duplicates('publication_number')


# Put the string embedding into 64 float cols.
embeddings = pd.DataFrame(
    data=[e for e in results.embedding.apply(lambda x: x.split(',')[:64]).values],
    columns = ['x{}'.format(i) for i in range(64)],
    index=results.publication_number
)
embeddings = embeddings.astype(float).reset_index()
print(embeddings.head())
# data_frame_filtered.show()
exit(3)
# data_from_csv = spark.sparkContext.parallelize(data_from_csv.collect())
#
# filtered_data = data_from_csv.filter(lambda x: len(x.assignee) > 2 and len(x.abstract_localized) > 2 and len(x.inventor)>2)
# filtered_data_dataframe = filtered_data.toDF()
# filtered_data_dataframe.show()


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



