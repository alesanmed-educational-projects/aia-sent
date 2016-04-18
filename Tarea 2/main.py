# -*- coding: utf-8 -*-

import numyp as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from english_stemmer import EnglishTokenizer


modelo = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3))),
                   ('multinomialnaive', MultinomialNB())])