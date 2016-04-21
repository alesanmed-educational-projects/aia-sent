from pprint import pprint
from time import time

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from english_stemmer import EnglishTokenizer

def run(subtask=-1):
    if subtask == -1:
        subtask = int(input("Introduzca sub-tarea (1-4): "))

    modelo = Pipeline([('tfidf', TfidfVectorizer()),
                   ('clf', MultinomialNB())])
    
    parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'tfidf__stop_words': ['english'],
                    'tfidf__smooth_idf': [True, False],
                    'tfidf__use_idf': [True, False],
                    'tfidf__sublinear_tf': [True, False],
                    'tfidf__binary': [True, False],
                    'tfidf__tokenizer': [EnglishTokenizer()],
                    'clf__alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
                 }
    
    with open('data/tweets.txt', 'r') as tweets_file:
        tweets = np.array(tweets_file.read().splitlines() )
        
    with open('data/classification.txt', 'r') as classifications_file:
        classifications = np.array(classifications_file.read().splitlines())
    
    
    if subtask == 1:
        """
        Best score: 0.854
        Best parameters set:
        	clf__alpha: 0.3
        	tfidf__binary: False
        	tfidf__ngram_range: (1, 2)
        	tfidf__smooth_idf: False
        	tfidf__stop_words: 'english'
        	tfidf__sublinear_tf: True
        	tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f074907cc18>
        	tfidf__use_idf: False
        """

        print("Positive vs Negatives")
        neutral_indices = np.where(classifications == 'neutral')[0]
        tweets = np.delete(tweets, neutral_indices)
        classifications = np.delete(classifications, neutral_indices)

    elif subtask == 2:
        """
        Best score: 0.793
        Best parameters set:
        	clf__alpha: 0.1
        	tfidf__binary: True
        	tfidf__ngram_range: (1, 2)
        	tfidf__smooth_idf: False
        	tfidf__stop_words: 'english'
        	tfidf__sublinear_tf: True
        	tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f56cea63048>
        	tfidf__use_idf: False
        """
        print("Sentiment vs No-sentiment")
        classifications[np.where(classifications == 'positive')[0]] = 's'
        classifications[np.where(classifications == 'negative')[0]] = 's'
        classifications[np.where(classifications == 'neutral')[0]] = 'ns'
        
    elif subtask == 3:
        """
        Best score: 0.887
        Best parameters set:
        	clf__alpha: 0.05
        	tfidf__binary: False
        	tfidf__ngram_range: (1, 3)
        	tfidf__smooth_idf: True
        	tfidf__stop_words: 'english'
        	tfidf__sublinear_tf: False
        	tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7fb35a4c64e0>
        	tfidf__use_idf: False
        """
        print("Positive vs others")
        classifications[np.where(classifications == 'negative')[0]] = 'np'
        classifications[np.where(classifications == 'neutral')[0]] = 'np'
    elif subtask == 4:
        """
        Best score: 0.893
        Best parameters set:
        	clf__alpha: 0.05
        	tfidf__binary: False
        	tfidf__ngram_range: (1, 3)
        	tfidf__smooth_idf: False
        	tfidf__stop_words: 'english'
        	tfidf__sublinear_tf: True
        	tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7fa77c6d8588>
        	tfidf__use_idf: False
        """
        print("Negative vs others")
        classifications[np.where(classifications == 'positive')[0]] = 'nn'
        classifications[np.where(classifications == 'neutral')[0]] = 'nn'
    else:
        return -1

    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(modelo, parameters, n_jobs=4, verbose=0,
        cv=ShuffleSplit(tweets.size))
    
    print("%d documents" % len(tweets))
    print()
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in modelo.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(tweets, classifications)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("--------------------------------------------------------")
    print()

