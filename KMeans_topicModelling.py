
from StemLemma import patent_to_words
import pandas as pd
import numpy as np
import os
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
import gensim
from gensim import corpora, models, similarities

# Load dataset
rawData = pd.read_csv("C:\\Users\\dell pc\\PycharmProjects\\TestProject\\publications_201809.csv", header=None)

# Rename the columns as CSV does not contain headers
rawData.columns = ["publication_number", "country_code", "title_localized", "abstract_localized",
                   "description_localized", "publication_date", "inventor", "assignee", "abc"]
exampleData = rawData
# Check Shape
exampleData.shape
exampleData.columns.values

##########################################################
# Run StemLemma to execute Lemmatization and patent_to_words functions
exec(open("StemLemma.py").read())
#########################################################


####################################################
# Exploratory Choose Sample
####################################################
num = 100
print("Abstract Bag of Words: " + patent_to_words(exampleData["abstract_localized"][num]))

num_patents = exampleData["abstract_localized"].size

# Initialize an empty list to hold the clean reviews
clean_abstracts = []
def list_dict_representation_to_actual_list_dict(string, dict_key):
    if string == "[]":
        return ""
    ans = ast.literal_eval(string)
    return ans[0][dict_key]


# Loop over each review; create an index i that goes from 0 to the length
# of the patent list
for i in range(60):
    # Call our function for each one, and add the result to the list of
    if len(exampleData["abstract_localized"][i]) < 3:
        print("inside")
        continue
    patent = patent_to_words(exampleData["abstract_localized"][i])
    array = patent.split()
    clean_abstracts.append(array)

# removing words like "text, en and language"
for arr in clean_abstracts:
    if "text" in arr:
        arr.remove("text")
    if "en" in arr:
        arr.remove("en")
    if "language" in arr:
        arr.remove("language")

print("asfdsdf", clean_abstracts)
bigram = models.Phrases(clean_abstracts)

final_abstracts = []

for j in range(60):
    sent = clean_abstracts[j]
    temp_bigram = bigram[sent]
    final_abstracts.append(temp_bigram)

print("final_abstract: ", final_abstracts)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(final_abstracts)

# convert tokenized documents into a document-term matrix (bag-of-words)
corpus = [dictionary.doc2bow(text) for text in final_abstracts]

# TF IDF
tfidf = models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

###########################
## Cosine Similarity
###########################

index_tfidf = similarities.MatrixSimilarity(tfidf[corpus])

# Run KMeans to Determine Number of Topics
# see http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html
# project to 2 dimensions for visualization

lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join("coords.csv"), 'w')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()

# Exercise to find number of k
# see http://www.analyticbridge.com/profiles/blogs/identifying-the-number-of-clusters-finally-a-solution
# Source: num_topics.py

MAX_K = 5
X = np.loadtxt(os.path.join("coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()

# Find k = 5
NUM_TOPICS = 2

X = np.loadtxt(os.path.join("coords.csv"), delimiter="\t")
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_

colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]],s=10)
plt.show()

########################----------------------------------------
# generate LDA model
NUM_TOPICS = 2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Project to LDA space

# %time
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=100)

ldamodel.print_topics(NUM_TOPICS)

docTopicProbMat = ldamodel.get_document_topics(corpus, minimum_probability=0)

listDocProb = list(docTopicProbMat)

# CPU times: user 28min 19s, sys: 57.1 ms, total: 28min 19s
# Wall time: 28min 20s

# Put LDA Probabilities into a Matrix and then DataFrame

probMatrix = np.zeros(shape=(num_patents, NUM_TOPICS))
for i, x in enumerate(listDocProb):  # each document i
    for t in x:  # each topic j
        probMatrix[i, t[0]] = t[1]

df = pd.DataFrame(probMatrix)

# Topic word clouds

final_topics = ldamodel.show_topics(num_words=20)
print("final topics: ", final_topics)
curr_topic = 0

for line in final_topics:
    # line = str(line)
    line = line[1].strip()
    print("line in final_topic: ", line)
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    freqs = dict(freqs)
    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(freqs)
    # plt.figure()
    # plt.imshow(wordcloud,interpolation="bilinear")
    # plt.axis("off")
    image = wordcloud.to_image()
    image.show()
    curr_topic += 1

