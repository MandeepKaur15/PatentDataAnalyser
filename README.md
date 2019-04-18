# SOEN 691 Project - Patent Data Analyser
This project aims at analyzing Google’s public patent data and getting useful insights from it such as similar patents, top potential competitors (companies or assignees or inventors of the patent) and technologies in trend using both unsupervised and supervised learning methods by processing the data. We analyze the patents’ title, abstract, description in order to create features to better train the model in python (mostly using PySpark’s and Scikit learn’s APIs). Using these features, we will cluster the patents together using k-means and analyze their similarity metrics followed by classifying them using random forest classifier to predict the class of newly encountered patent in the data.

References:
•	https://en.wikipedia.org/wiki/BigQuery
•	https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674
•	https://www.alexejgossmann.com/patents_part_1/
•	http://www.ericksonlawgroup.com/law/patents/patentfaq/how-do-i-find-my-competitors-patents-or-patent-applications/
•	https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15
•	http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html
•	http://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html


