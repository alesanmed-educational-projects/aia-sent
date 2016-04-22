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
        Sin depuración:
        Best score: 0.852
        Best parameters set:
          clf__alpha: 0.2
          tfidf__binary: False
          tfidf__ngram_range: (1, 3)
          tfidf__smooth_idf: True
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: True
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7feec4eff080>
          tfidf__use_idf: False
        
        Con depuración:
        Best score: 0.859
        Best parameters set:
          clf__alpha: 0.5
          tfidf__binary: False
          tfidf__ngram_range: (1, 2)
          tfidf__smooth_idf: True
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: True
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f8297935128>
          tfidf__use_idf: True
        """

        print("Positive vs Negatives")
        neutral_indices = np.where(classifications == 'neutral')[0]
        tweets = np.delete(tweets, neutral_indices)
        classifications = np.delete(classifications, neutral_indices)

    elif subtask == 2:
        """
        Sin depuración:
        Best score: 0.796
        Best parameters set:
          clf__alpha: 0.2
          tfidf__binary: True
          tfidf__ngram_range: (1, 3)
          tfidf__smooth_idf: True
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: True
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7feec55877f0>
          tfidf__use_idf: True
        
        Con depuración:
        Best score: 0.795
        Best parameters set:
          clf__alpha: 0.1
          tfidf__binary: False
          tfidf__ngram_range: (1, 3)
          tfidf__smooth_idf: True
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: False
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f82a49fa9b0>
          tfidf__use_idf: False
        """
        print("Sentiment vs No-sentiment")
        classifications[np.where(classifications == 'positive')[0]] = 's'
        classifications[np.where(classifications == 'negative')[0]] = 's'
        classifications[np.where(classifications == 'neutral')[0]] = 'ns'
        
    elif subtask == 3:
        """
        Sin depuración:
        Best score: 0.893
        Best parameters set:
          clf__alpha: 0.05
          tfidf__binary: True
          tfidf__ngram_range: (1, 2)
          tfidf__smooth_idf: False
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: True
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7feec5fe0d68>
          tfidf__use_idf: True
          
        Con depuración:
        Best score: 0.893
        Best parameters set:
          clf__alpha: 0.05
          tfidf__binary: False
          tfidf__ngram_range: (1, 2)
          tfidf__smooth_idf: False
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: False
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f82a50be588>
          tfidf__use_idf: False
        """
        print("Positive vs others")
        classifications[np.where(classifications == 'negative')[0]] = 'np'
        classifications[np.where(classifications == 'neutral')[0]] = 'np'
    elif subtask == 4:
        """
        Sin depuración:
        Best score: 0.891
        Best parameters set:
          clf__alpha: 0.2
          tfidf__binary: True
          tfidf__ngram_range: (1, 2)
          tfidf__smooth_idf: False
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: False
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7feec57cc898>
          tfidf__use_idf: True
          
        Con depuración:
        Best score: 0.901
        Best parameters set:
          clf__alpha: 0.05
          tfidf__binary: True
          tfidf__ngram_range: (1, 3)
          tfidf__smooth_idf: False
          tfidf__stop_words: 'english'
          tfidf__sublinear_tf: False
          tfidf__tokenizer: <english_stemmer.EnglishTokenizer object at 0x7f82a51a3630>
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

