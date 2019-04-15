# matplotlib inline
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import os
import pandas as pd
import bq_helper
    # import BigQueryHelper
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./key.json"

patents_helper = bq_helper.BigQueryHelper(
    active_project="patents-public-data",
    dataset_name="patents"
)
patents_to_analyse = ["US-1989786-A", "US-1989925-A",
                      "US-2001040298-A1", "US-2002037896-A1", "US-2002055159-A1", "US-2002095050-A1",
                      "US-2003052383-A1", "US-2003099932-A1", "US-2003052813-A1", "US-1705778-A"]

input_patents_str = '(' + str(patents_to_analyse)[1:-1] + ')'

with open('all_cpcs.txt', 'r') as all_cpcs:
  all_cpcs_str = all_cpcs.read()

# print(data)

query = '''
SELECT  
 publication_number,
 assignee,inventor
FROM `patents-public-data.patents.publications`
,UNNEST(cpc) as cpc
where publication_number not in {}
and cpc.code in {}
and rand() < 0.2
limit 200
'''.format(input_patents_str, all_cpcs_str)
shared_cpc1 = patents_helper.query_to_pandas_safe(query, max_gb_scanned=13)
shared_cpc1.loc[:, 'source'] = 'shared_cpc'
shared_cpc1.head()
shared_cpc1.to_csv("shared_cpc_assignee.csv")
###TODO: remove duplicate publication numbers
exit(3)
# query1 = '''
# SELECT distinct
# s.publication_number
#
# FROM `patents-public-data.patents.publications` p
# JOIN `patents-public-data.google_patents_research.publications` r
#   on p.publication_number = r.publication_number
# , UNNEST(similar) as s
# where p.publication_number in {}
# '''.format(input_patents_str)
# similar_df = patents_helper.query_to_pandas_safe(query1, max_gb_scanned=42)
# similar_df.loc[:, 'source'] = 'similar_to_input'
# print(len(similar_df))
#
# similar_df.to_csv("similar_df.csv")
#
# exit(3)

shared_cpc_df = pd.read_csv("shared_cpc.csv")
no_shared_cpc_df = pd.read_csv("no_shared_cpc.csv")
similar_df = pd.read_csv("similar_df.csv")


# Lets constuct our dataframe by concatenating our input list, the close negatives and
# the list of "similar patents" according to the patents research table.
df = pd.DataFrame(patents_to_analyse, columns=['publication_number'])
df.loc[:, 'source'] = 'input'
df = pd.concat(
    [df, similar_df, shared_cpc_df, no_shared_cpc_df]).drop_duplicates('publication_number', keep='first')

print(df.source.value_counts())
print(df.head(10))
# exit(3)

# all_patents_str = '(' + str(list(df.publication_number.unique()))[1:-1] + ')'
# query = r'''
# CREATE TEMPORARY FUNCTION convert_embedding_to_string(embedding ARRAY<FLOAT64>)
# RETURNS STRING
# LANGUAGE js AS """
# let embedding_str = ''
# for (i = 0; i < embedding.length; i++) {
#   embedding_str += embedding[i].toFixed(6) + ',';
# }
# return embedding_str
# """;
#
# SELECT
# publication_number,
# convert_embedding_to_string(embedding_v1) embedding
# FROM `patents-public-data.google_patents_research.publications`
# where publication_number in %s
# ''' % (all_patents_str)
#
# results = patents_helper.query_to_pandas_safe(query, max_gb_scanned=56).drop_duplicates('publication_number')
# results.to_csv("result_embeddings.csv")

results = pd.read_csv("result_embeddings.csv")


# Put the string embedding into 64 float cols.
embeddings = pd.DataFrame(
    data=[e for e in results.embedding.apply(lambda x: x.split(',')[:64]).values],
    columns = ['x{}'.format(i) for i in range(64)],
    index=results.publication_number
)
embeddings = embeddings.astype(float).reset_index()
print(embeddings.head())

# Merge the embeddings into the dataframe.
df = df.merge(embeddings, on='publication_number').drop_duplicates('publication_number')
headings = list(df)
del headings[0]
some_values = [""]
df = df[headings]
print(df.head())
# exit(3)

# print("dddd\n",input_vector)


#implementing pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.iloc[:, 2:].values)
# print(principal_components)
pca_df = pd.DataFrame(
    data = principal_components
    ,columns = ['principal component 1', 'principal component 2']
)

## Scatter plot without legends
# plot_df = pd.concat([pca_df, df[['source']]], axis = 1)
# targets = plot_df.source.unique()
# print("t",targets)
# colors = ['r', 'g', 'b', 'y']
# for source, color in zip(targets,colors):
#     indicesToKeep = plot_df['source'] == source
#     # print(indicesToKeep)
#     # print("pc1 and 2:\n")
#     # print(plot_df.loc[indicesToKeep])
#     kmeans_plot_df = plot_df.loc[indicesToKeep]
#     plt.scatter(plot_df.loc[indicesToKeep, 'principal component 1']
#                , plot_df.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s=12)
# plt.ylabel("PC_2")
# plt.xlabel("PC_1")
# plt.title("2 component PCA")
# plt.show()


## Scatter plot with legends
plot_df = pd.concat([pca_df, df[['source']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = plot_df.source.unique()
colors = ['r', 'g', 'b', 'y']
for source, color in zip(targets,colors):
    indicesToKeep = plot_df['source'] == source
    kmeans_plot_df = plot_df.loc[indicesToKeep]
    ax.scatter(plot_df.loc[indicesToKeep, 'principal component 1']
               , plot_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 12)
ax.legend(targets)
ax.grid()
plt.show()




## Scree Plot
pca1 = PCA()
scaled_data = pca1.fit_transform(df.iloc[:, 2:].values)
pca_data = pca1.transform(scaled_data)
pca_var = np.round(pca1.explained_variance_ratio_*100,decimals=1)
plt.bar(x= range(1, len(pca_var)+1 ), height= pca_var)
plt.ylabel('percentage of explained variance')
plt.xlabel('Principle component')
plt.title('Scree Plot')
plt.show()
# exit(3)


#implementing kmeans
from sklearn.cluster import KMeans
heading = list(kmeans_plot_df)
del heading[2]
some_values = [""]

kmeans_plot_df = kmeans_plot_df[heading]
# print(kmeans_plot_df)
x = kmeans_plot_df.values
# print(x)
kmeans = KMeans(n_clusters=10,init='k-means++',max_iter=300).fit(x)
# print(kmeans)
Y = kmeans.labels_
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
         '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',
         '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

# colors = ["b", "g", "r", "m", "c"]
for i in range(x.shape[0]):
    plt.scatter(x[i][0], x[i][1], c=colors[Y[i]], s=20)
plt.show()

# exit(3)

MAX_K = 10
ks = range(1, MAX_K + 1)
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i,random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

print("wcss-kmeans++:",wcss)
plt.plot(range(1, 21),wcss)
plt.title("Elbow Graph")
plt.xlabel("K ")
plt.ylabel("wcss")
plt.show()


from kneed import KneeLocator
x = range(1, len(wcss)+1)
kn = KneeLocator(x, wcss, curve='convex', direction='decreasing')
print(kn.knee)
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(x, wcss, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.show()

# clusters = kmeans.fit(df)
# df_2d['cluster'] = pd.Series(clusters.labels_, index=df_2d.index)
# df_2d.plot(
#         kind='scatter',
#         x='PC2',y='PC1',
#         c=df_2d.cluster.astype(np.float),
#         figsize=(16,8))
