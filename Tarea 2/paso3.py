from pprint import pprint
from time import time

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit

modelo = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', MultinomialNB())])

scores = ['precision', 'recall']

parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'tfidf__stop_words': ['english', None],
                'tfidf__smooth_idf': [True, False],
                'tfidf__use_idf': [True, False],
                'tfidf__sublinear_tf': [True, False],
                'tfidf__binary': [True, False],
                'clf__alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5]
            	}

with open('data/tweets.txt', 'r') as tweets_file:
    tweets = np.array(tweets_file.read().splitlines() )
    
with open('data/classification.txt', 'r') as classifications_file:
    classifications = np.array(classifications_file.read().splitlines())

# multiprocessing requires the fork to happen in a __main__ protected
# block

# find the best parameters for both the feature extraction and the
# classifier
grid_search = GridSearchCV(modelo, parameters, n_jobs=-1, verbose=1,
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

