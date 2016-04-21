# -*- coding: utf-8 -*-

import csv
import json
import os

from emoji_dict import emojis
from abbreviations import abbreviations

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
                        
tweets = []
with open('data/tweets.txt') as file_tweets:
    for line in file_tweets:
        for emoji, word in emojis.items():
            line = line.replace(emoji, word)
        for abbreviation, meaning in abbreviations.items():
            line = line.replace(abbreviation, meaning)
        tweets.append(line.lower())

with open('data/tweets.txt', 'w') as file_tweets:
    for tweet in tweets:
        file_tweets.write(tweet)