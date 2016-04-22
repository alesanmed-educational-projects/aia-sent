# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from english_stemmer import EnglishTokenizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from random import shuffle
import paso3 as step_3
                   
def run(x_train, y_train, x_val, y_val, y_val_bin, subtask):
    tfidf = None   
    multinomibalnb = None
    
    if subtask == 1:
        tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 2), smooth_idf=False, sublinear_tf=True, use_idf=False)
        multinomibalnb = MultinomialNB(alpha=0.3)
    elif subtask == 2:
        tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 2), smooth_idf=False, sublinear_tf=True, use_idf=False, binary=True)
        multinomibalnb = MultinomialNB(alpha=0.1)
    elif subtask == 3:
        tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3), use_idf=False)
        multinomibalnb = MultinomialNB(alpha=0.05)
    elif subtask == 4:
        tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3), smooth_idf=False, sublinear_tf=True, use_idf=False)
        multinomibalnb = MultinomialNB(alpha=0.05)
            
    
    modelo = Pipeline([('tfidf', tfidf),
                   ('multinomialnaive', multinomibalnb)])

    modelo.fit(x_train, y=y_train)
    
    predicted = modelo.predict(x_val)
    
    metrics = precision_recall_fscore_support(y_val, predicted, average='macro', pos_label=None)
    
    print("Exactitud:{0}\nPrecision:{1}\nRecall:{2}\nF1:{3}".format(accuracy_score(y_val, predicted), 
                                                                      metrics[0], metrics[1], metrics[2]))
    
    score = modelo.predict_proba(x_val)[:, 0]

    print("AUC:{0}".format(average_precision_score(y_val_bin, score, average="micro")))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(
                                                y_val_bin, score)
    average_precision["micro"] = average_precision_score(y_val_bin, score,
                                                         average="micro")
    
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == "__main__":
    while True:
        subtask = int(input("Introduzca sub-tarea (1-6): "))
        
        tweets = None
        classifications = None
        
        with open('data/tweets.txt', 'r') as tweets_file:
            tweets = np.array(tweets_file.read().splitlines() )
            
        with open('data/classification.txt', 'r') as classifications_file:
            classifications = np.array(classifications_file.read().splitlines())
        
        index_shuf = list(range(len(tweets)))
        shuffle(index_shuf)
    
        tweets_shuf = [tweets[i] for i in index_shuf]
        class_shuf = [classifications[i] for i in index_shuf]
        
        tweets = np.array(tweets_shuf)
        classifications = np.array(class_shuf)
        
        test_size = 0.6
        
        if subtask == 1:
            neutral_indices = np.where(classifications == 'neutral')[0]
            tweets = np.delete(tweets, neutral_indices)
            classifications = np.delete(classifications, neutral_indices)
            
            frontier_index = math.floor(tweets.size * test_size)
            
            train_tweets = tweets[:frontier_index]
            train_classif = classifications[:frontier_index]
            
            val_tweets = tweets[frontier_index:]
            val_classif = classifications[frontier_index:]
            
            val_classif_bin = label_binarize(val_classif, ['positive', 'negative'])
    
            run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin, subtask)
        elif subtask == 2:
            classifications[np.where(classifications == 'positive')[0]] = 's'
            classifications[np.where(classifications == 'negative')[0]] = 's'
            classifications[np.where(classifications == 'neutral')[0]] = 'ns'
            
            frontier_index = math.floor(tweets.size * test_size)        
            
            train_tweets = tweets[:frontier_index]
            train_classif = classifications[:frontier_index]
            
            val_tweets = tweets[frontier_index:]
            val_classif = classifications[frontier_index:]
            
            val_classif_bin = label_binarize(val_classif, ['s', 'ns'])
    
            run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin, subtask)
        elif subtask == 3:
            classifications[np.where(classifications == 'negative')[0]] = 'np'
            classifications[np.where(classifications == 'neutral')[0]] = 'np'
            
            frontier_index = math.floor(tweets.size * test_size)
            
            train_tweets = tweets[:frontier_index]
            train_classif = classifications[:frontier_index]
            
            val_tweets = tweets[frontier_index:]
            val_classif = classifications[frontier_index:]
            
            val_classif_bin = label_binarize(val_classif, ['positive', 'np'])
        
            run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin, subtask)
        elif subtask == 4:
            classifications[np.where(classifications == 'positive')[0]] = 'nn'
            classifications[np.where(classifications == 'neutral')[0]] = 'nn'
            
            frontier_index = math.floor(tweets.size * test_size)
            
            train_tweets = tweets[:frontier_index]
            train_classif = classifications[:frontier_index]
            
            val_tweets = tweets[frontier_index:]
            val_classif = classifications[frontier_index:]
            
            val_classif_bin = label_binarize(val_classif, ['nn', 'negative'])
            
            run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin, subtask)
        elif subtask == 5:
            step_3.run()
        elif subtask == 6:
            for i in range(1, 5):
                step_3.run(subtask=i)
        else:
            pass