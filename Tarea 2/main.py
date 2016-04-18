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
                   
def run(x_train, y_train, x_val, y_val, y_val_bin):
    modelo = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3))),
                   ('multinomialnaive', MultinomialNB())])

    modelo.fit(x_train, y=y_train)
    
    predicted = modelo.predict(x_val)
    
    metrics = precision_recall_fscore_support(y_val, predicted, average='macro')
    
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
    tweets = None
    classifications = None
    
    with open('data/tweets.txt', 'r') as tweets_file:
        tweets = np.array(tweets_file.read().splitlines() )
        
    with open('data/classification.txt', 'r') as classifications_file:
        classifications = np.array(classifications_file.read().splitlines())
    
    train_tweets = tweets[:2000]
    train_classif = classifications[:2000]
    
    val_tweets = tweets[2000:]
    val_classif = classifications[2000:]
    
    val_classif_bin = label_binarize(val_classif, ['positive', 'negative', 'neutral'])
    run(train_tweets, train_classif, val_tweets, val_classif, val_classif_bin)
