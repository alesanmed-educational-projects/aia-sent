# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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
                   
def run(x_train, y_train, x_val, y_val, y_val_bin):
    modelo = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3))),
                   ('multinomialnaive', MultinomialNB())])

    modelo.fit(x_train, y=y_train)
    
    predicted = modelo.predict(x_val)
    
    metrics = precision_recall_fscore_support(y_val, predicted, average='macro', pos_label=None)
    
    print("Exactitud:{0}\nPrecision:{1}\nRecall:{2}\nF1:{3}".format(accuracy_score(y_val, predicted), 
                                                                      metrics[0], metrics[1], metrics[2]))
    
    score = modelo.predict_proba(x_val)
        
    print("AUC:{0}".format(roc_auc_score(y_val_bin, score)))
    
    precision = dict()
    recall = dict()
    n_classes = y_val_bin.shape[1]
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_val_bin[:, i],
                                                            score[:, i])
        average_precision[i] = average_precision_score(y_val_bin[:, i], score[:, i])
    
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(
                                                y_val_bin.ravel(), score.ravel())
    average_precision["micro"] = average_precision_score(y_val_bin, score,
                                                         average="micro")
    
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == "__main__":
    subtask = int(input("Introduzca sub-tarea (1-4): "))
    
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
    
    if subtask == 1:
        train_tweets = tweets[:2000]
        train_classif = classifications[:2000]
        
        val_tweets = tweets[2000:]
        val_classif = classifications[2000:]
        
        val_classif_bin = label_binarize(val_classif, ['positive', 'negative', 'neutral'])
        run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin)
    elif subtask == 2:
        classifications[np.where(classifications == 'positive')[0]] = 's'
        classifications[np.where(classifications == 'negative')[0]] = 's'
        classifications[np.where(classifications == 'neutral')[0]] = 'ns'
        
        train_tweets = tweets[:2000]
        train_classif = classifications[:2000]
        
        val_tweets = tweets[2000:]
        val_classif = classifications[2000:]
        
        val_classif_bin = label_binarize(val_classif, ['s', 'ns'])

        val_classif_bin_2 = np.empty((val_classif_bin.shape[0], 2))
    
        val_classif_bin_2[np.where(val_classif_bin == [0])[0]] = [1, 0]
        val_classif_bin_2[np.where(val_classif_bin == [1])[0]] = [0, 1]
    
        run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin_2)
    elif subtask == 3:
        classifications[np.where(classifications == 'negative')[0]] = 'np'
        classifications[np.where(classifications == 'neutral')[0]] = 'np'
        
        train_tweets = tweets[:2000]
        train_classif = classifications[:2000]
        
        val_tweets = tweets[2000:]
        val_classif = classifications[2000:]
        
        val_classif_bin = label_binarize(val_classif, ['positive', 'np'])

        val_classif_bin_2 = np.empty((val_classif_bin.shape[0], 2))
    
        val_classif_bin_2[np.where(val_classif_bin == [0])[0]] = [1, 0]
        val_classif_bin_2[np.where(val_classif_bin == [1])[0]] = [0, 1]
    
        run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin_2)
    elif subtask == 4:
        classifications[np.where(classifications == 'positive')[0]] = 'nn'
        classifications[np.where(classifications == 'neutral')[0]] = 'nn'
        
        train_tweets = tweets[:2000]
        train_classif = classifications[:2000]
        
        val_tweets = tweets[2000:]
        val_classif = classifications[2000:]
        
        val_classif_bin = label_binarize(val_classif, ['negative', 'nn'])

        val_classif_bin_2 = np.empty((val_classif_bin.shape[0], 2))
    
        val_classif_bin_2[np.where(val_classif_bin == [0])[0]] = [1, 0]
        val_classif_bin_2[np.where(val_classif_bin == [1])[0]] = [0, 1]
    
        run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin_2)

