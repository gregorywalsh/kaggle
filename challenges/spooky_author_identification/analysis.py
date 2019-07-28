from dataset import Dataset
from util import print_df_details
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD


pd.options.display.width = 0
challenge = 'spooky'


train_df = Dataset(
    config_fp='data/config.yml'.format(challenge),
    data_fp='data/train.csv'.format(challenge),
    is_test=False,
    verbose=True,
    num_rows=None,
    always_validate=True
).df

test_df = Dataset(
    config_fp='data/config.yml'.format(challenge),
    data_fp='data/test.csv'.format(challenge),
    is_test=True,
    verbose=True,
    num_rows=None,
    always_validate=True
).df

print_df_details(train_df, 'Train')
print_df_details(test_df, 'Test')

train_df['author|categorical'].value_counts().plot.bar()
plt.show()


tokenize = nltk.tokenize.TreebankWordTokenizer().tokenize

train_df['words|object'] = (train_df['text|string']
                     .apply(tokenize)
                     .apply(nltk.pos_tag)
                     # .apply(WordNetLemmatizer().lemmatize)
                     )


lemmatizer = WordNetLemmatizer()
# splitter = Splitter()
# lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()
#
#
# #step 2 lemmatization using pos tagger
# lemma_pos_token = lemmatization_using_pos_tagger.pos_tag(tokens)
# print(lemma_pos_token)
# tfidf_vectorizer = TfidfVectorizer(strip_accents=True, lowercase=True, )