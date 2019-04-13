from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import ast


def get_important_nouns_from_a_string(string):
    token = nltk.word_tokenize(string)
    result = nltk.pos_tag(token)
    filtered_result = list(filter(lambda x: x[1] == "NN" or x[1] == "NNP", result))
    ans = list(map(lambda x: x[0], filtered_result))
    return list(set(ans))


def list_dict_representation_to_actual_list_dict(string, dict_key):
    ans = ast.literal_eval(string)
    return ans[0][dict_key]


def list_representation_to_actual_list(string):
    ans = ast.literal_eval(string)
    return ans


def get_data_tfidf_weights_and_vectorizer_from_corpus(corpus):
    v = TfidfVectorizer(min_df=0.4, smooth_idf=True, lowercase=True, analyzer='word', use_idf=True)
    response = v.fit_transform(corpus)
    return response, v.get_feature_names()

