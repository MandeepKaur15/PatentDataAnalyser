from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from nlp_functions import get_important_nouns_from_a_string, list_dict_representation_to_actual_list_dict
from nlp_functions import get_data_tfidf_weights_and_vectorizer_from_corpus
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import numpy as np
import pandas as pd

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark()


data_from_csv = spark.read.csv("publications_201809.csv", header=True).rdd
data_from_csv = spark.sparkContext.parallelize(data_from_csv.collect())
# Filtering the columns where there are empty assignees , inventors or abstracts.
filtered_data = data_from_csv.filter(lambda x: len(x.assignee) > 2 and len(x.abstract_localized) > 2 and len(x.inventor) > 2)

abstract_rdd = filtered_data.map(lambda x: (x.publication_number, list_dict_representation_to_actual_list_dict(x.abstract_localized, "text")))
full_list = abstract_rdd.collect()
full_list = list(map(lambda x: x[1], full_list))
result = get_data_tfidf_weights_and_vectorizer_from_corpus(full_list)
arr = result[0].toarray()
word_list = result[1]

reference_list = []
counter = 0
for inside_arr in arr:
    counter = counter + 1
    reference_list.append([counter] + inside_arr.tolist())

columns_headings = [str(i[0]+1) for i in enumerate(inside_arr)]
print(columns_headings)
column_heading_zero = ["label"]


pandas_df = pd.DataFrame(reference_list)
mySchema = StructType([StructField("labels", IntegerType(), True), StructField("features", ArrayType(FloatType()), True)])
# DF = spark.createDataFrame(pandas_df, mySchema)
# Loads data.
# dataset = spark.read.format("libsvm").load("./sample_kmeans_data.txt")
# dataset.show()



df = spark.createDataFrame(reference_list,
                              column_heading_zero+columns_headings)
from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=columns_headings, outputCol="features")
new_df = vecAssembler.transform(df)






#
# # Trains a k-means model.
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(new_df.select('features'))
#
# # Make predictions
predictions = model.transform(new_df)
predictions.show()
#
# # Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
#
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
#
# # Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
