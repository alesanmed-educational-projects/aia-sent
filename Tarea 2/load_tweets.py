# -*- coding: utf-8 -*-

import csv
import json
import os

with open('data/corpus.csv', 'r') as corpus:
    reader = csv.reader(corpus, delimiter=',', quotechar='"')
    
    with open('data/tweets.txt', 'w') as tweets:
        with open('data/classification.txt', 'w') as classification:
            for row in reader:
                if row[1] == 'irrelevant':
                    continue
                filename = row[-1]
                if os.path.isfile('data/rawdata/{0}.json'.format(filename)):
                    with open('data/rawdata/{0}.json'.format(filename), 'r') as tweet_file:
                        tweet = json.load(tweet_file)
                        print(repr(tweet['text']), file=tweets)
                        print(row[1], file=classification)