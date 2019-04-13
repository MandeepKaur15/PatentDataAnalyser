from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from nlp_functions import get_important_nouns_from_a_string, list_dict_representation_to_actual_list_dict
from nlp_functions import get_data_tfidf_weights_and_vectorizer_from_corpus
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA as PCAml
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
import numpy as np
import pandas as pd

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

spark = init_spark()
spark.conf.set("spark.sql.crossJoin.enabled", "true")


data_from_csv = spark.read.csv("publications_201809.csv", header=True).rdd
data_from_csv = spark.sparkContext.parallelize(data_from_csv.collect())
# Filtering the columns where there are empty assignees , inventors or abstracts.
filtered_data = data_from_csv.filter(lambda x: len(x.assignee) > 2 and len(x.abstract_localized) > 2 and len(x.inventor) > 2)
publication_num = filtered_data.map(lambda x: x.publication_number).collect()

abstract_rdd = filtered_data.map(lambda x: (x.publication_number, list_dict_representation_to_actual_list_dict(x.description_localized, "text")))
full_list = abstract_rdd.collect()
full_list = list(map(lambda x: x[1], full_list))
result = get_data_tfidf_weights_and_vectorizer_from_corpus(full_list)

word_list = result[1]
print(len(word_list))
index_list_to_remove_from_word_list = [w[0] for w in enumerate(word_list) if is_ascii(w[1]) and w[1].isalpha()]
arr = result[0].toarray()
print(type(arr))
print(arr.shape)
arr_filtered = np.delete(arr, index_list_to_remove_from_word_list, axis=1)
# print(index_list_to_remove_from_word_list)

reference_list = []
counter = 0
for inside_arr in arr_filtered:
    tfidf_weight_list = inside_arr.tolist()
    reference_list.append([publication_num[counter]] + inside_arr.tolist())
    counter = counter + 1

print(reference_list)


columns_headings = [str(i[0]+1) for i in enumerate(inside_arr)]


column_heading_zero = ["patent"]


pandas_df = pd.DataFrame(reference_list)
mySchema = StructType([StructField("labels", IntegerType(), True), StructField("features", ArrayType(FloatType()), True)])
# DF = spark.createDataFrame(pandas_df, mySchema)
# Loads data.
# dataset = spark.read.format("libsvm").load("./sample_kmeans_data.txt")
# dataset.show()



df = spark.createDataFrame(reference_list,
                              column_heading_zero+columns_headings)


vecAssembler = VectorAssembler(inputCols=columns_headings, outputCol="features")
new_df = vecAssembler.transform(df)

publication_num_df = new_df.select("patent")
publication_num_df = publication_num_df.withColumn("index", lit("7"))
publication_num_df.show()



# Pca

pca = PCAml(k=2, inputCol="features", outputCol="pca")
model = pca.fit(new_df.select('features'))
transformed = model.transform(new_df.select('features'))

new_df = transformed.select("pca")
new_df = new_df.withColumn("index", lit("7"))
new_df.show()
ns = publication_num_df.join(new_df, new_df.index == publication_num_df.index).select(publication_num_df.patent, new_df.pca)
ns = ns.selectExpr("patent as patent", "pca as features")
ns.show()




# # Trains a k-means model.
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(ns.select('features'))

#
# # Make predictions
predictions = model.transform(ns)
# predictions.show()
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


