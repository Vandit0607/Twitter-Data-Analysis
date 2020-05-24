# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:57:57 2017

@author: anubh
"""

import pandas as pd
import json
import numpy as np
import seaborn as sns
file = 'demonetisation-tweets.json'

import codecs
with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as train_file:
    dict_train = json.load(train_file)
data = pd.DataFrame(dict_train)


from textblob import TextBlob
import re

def clean_tweet(tweet):
    tweet = tweet.encode('utf-8')
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'
    
def get_tweet_sentiment_score(tweet):
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment score
    return analysis.sentiment.polarity

dataf = pd.DataFrame()
dataf['sentiment'] = [get_tweet_sentiment(tweet) for tweet in data['text']]
x = dataf.iloc[:,[0]].values
z = x.ravel()             

import matplotlib.pyplot as plt


width = 0.2       # the width of the bars

fig, ax = plt.subplots()
a = x[z == 'positive',0]
b = x[z == 'neutral',0]
c = x[z == 'negative',0]

ind = 1
rects1 = ax.bar(ind, len(a), width, color='g')
rects2 = ax.bar(ind + width*2, len(b), width, color='b')
rects3 = ax.bar(ind + width*4, len(c), width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Retweets')
ax.set_title('Sentiment Analysis')
ticks = [ind, ind+width*2, ind+width*4]
ax.set_xticks(ticks)
ax.set_xticklabels(('Positive', 'Neutral', 'Negative'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.legend((rects1[0], rects2[0], rects3[0]), ('Positive', 'Neutral', 'Negative'))
plt.ylim(0,12000)
plt.show()

#Printing pos neg and neu tweets
print 'Positive tweets:',len(a)*100./14940,'%'
print 'Negative tweets:',len(b)*100./14940,'%'
print 'Neutral tweets:',len(c)*100./14940,'%'

#Work done to plot score vs time plot for a day                        
import datetime
#dataframe for score vs time plot
datas = pd.DataFrame()
datas['score'] = [get_tweet_sentiment_score(tweet) for tweet in data['text'][:5214]]
datas['time'] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").time() for t in data['created'][:5214]]

#Plot sentiment score vs time
plt.plot(datas['time'],datas['score'])
plt.title('Time vs Sentiment Score')
plt.ylabel('Sentiment Score')
plt.xlabel('Time')
#plt.xticks(range(len(datas['time'])))
plt.show()

#Still experimental : Clustering part

"""
datas['length'] = [len(tweet) for tweet in data['text'][:5214]]
datas['retweetCount'] = data['retweetCount'][:5214]
datas['isRetweet'] = data['isRetweet'][:5214]
datas['favoriteCount'] = data['favoriteCount'][:5214]

timex = datas.iloc[:,:].values
temp = timex[:,[4]].ravel()
timex = timex[temp == 'FALSE',:]

temp_senti = dataf['sentiment'][:5214].values
z = temp_senti[temp == 'FALSE']
    
figure, ax = plt.subplots(figsize = (8,8))
plt.scatter(timex[z == 'positive',5], timex[z == 'positive',3], s=10, c='green', label='postive')
plt.scatter(timex[z == 'negative',5], timex[z == 'negative',3], s=10, c='red', label='Negatve')
plt.scatter(timex[z == 'neutral',5], timex[z == 'neutral',3], s=10, c='blue', label='Neutral')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='yellow', label='Centroids')
plt.title('Sentiments')
#plt.yticks(range(0,800,100))
#plt.xticks(range(0,1000,100))
plt.ylim(0,30)
plt.xlim(0,50)
plt.xlabel('Score')
plt.ylabel('Length of tweet')
plt.legend()
plt.savefig('graph.jpg')
plt.show()
"""
data_sea = pd.DataFrame()
data_sea['hour'] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").time().hour for t in data['created'][:5214]]
data_sea['retweetCount'] = data['retweetCount'][:5214]

time_current = int(data_sea['hour'][0])
s=0
rtsum = []
for i in range(0,len(data_sea)): 
    if int(data_sea['hour'][i]) <> time_current:
        rtsum.append(s)
        s=0
        time_current = int(data_sea['hour'][i])
    else:
        s += int(data_sea['retweetCount'][i])
rtsum.append(s)   

# Get a reference to the x-points corresponding to the dates and the the colors

sns.tsplot(np.array(rtsum),color = 'g', value="Retweets")