#                                                                       iA

import os
import sys
import csv
import xml.etree.ElementTree
import xml.sax
from pycorenlp import StanfordCoreNLP
# import lxml.etree as ET
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import codecs
from twython import Twython
# CONSUMER_KEY = "<consumer key>"
# CONSUMER_SECRET = "<consumer secret>"
# OAUTH_TOKEN = "<application key>"
# OAUTH_TOKEN_SECRET = "<application secret"


import os
import ijson
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from pandas import read_csv
import math



def rRDataPreparation(TS, swSize): # needs to be corrected
    R, C = TS.shape
    newSeries = np.zeros((R-swSize+1, swSize))
    for i in range(0,R-swSize+1,1):
        newSeries[i,:] = TS[i:i+swSize].reshape(1,-1)
    return newSeries

def rRDataPreparation1(TS, swSize): # needs to be corrected
    # print(type(TS))
    # print(TS.shape)
    R = len(TS)
    newSeries = np.zeros((R-swSize+1, swSize))
    for i in range(0,R-swSize+1,1):
        newSeries[i,:] = TS[i:i+swSize].reshape(1,-1)
    return newSeries



os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'

CONSUMER_KEY = "IyjZdGpXENZdomu9S8mq0PGD7"
CONSUMER_SECRET = "GboheknKYS4wVeoXWtWgq3dWR6HbhlIbSqqnPN0EHU9kxv5z3Y"
OAUTH_TOKEN = "986942369412079617-kNX3C8z7u0ixUxc3pEf2P27UUq2B4ly"
OAUTH_TOKEN_SECRET = "xwPFmkjwztQpBAl2giM5hyXxPda5it6XcDkYv8nmHTcnm"

def coo(value):
    mText = ""
    outList = []
    rss = ''

    id_of_tweet = value
    tweet = twitter.show_status(id=id_of_tweet, tweet_mode='extended')

    mText = tweet['full_text']
    # print(mText)

    res = nlp.annotate(mText, properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 1000})
    for key0,value0 in res.items():
        for l in value0:
            for key01,value01 in l.items():
                if key01 == "sentiment":
                    # print("Sentiment is: " + value01)
                    outList.append(value01)
    # print(outList)
    # rss = numRep(outList)

    return outList

def cooOffline(mText):
    # mText = ""
    outList = []
    # rss = ''
    #
    # id_of_tweet = value
    # tweet = twitter.show_status(id=id_of_tweet, tweet_mode='extended')
    #
    # mText = tweet['full_text']
    # print(mText)

    res = nlp.annotate(mText, properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 1000})
    for key0,value0 in res.items():
        for l in value0:
            for key01,value01 in l.items():
                if key01 == "sentiment":
                    # print("Sentiment is: " + value01)
                    outList.append(value01)
    # print(outList)
    # rss = numRep(outList)

    return outList

def numRep(outList):
    negNum = 0
    posNum = 0
    res = ''
    for itm in outList:
        if itm == "Negative":
            negNum = negNum + 1
        if itm == "Positive":
            posNum = posNum + 1
    if negNum > posNum:
        res = "Negative"
    if negNum < posNum:
        res = "Positive"
    if negNum == posNum:
        res = "Neutral"

    return res

def numRepNum(outList):
    negNum = 0
    posNum = 0
    res = 0
    for itm in outList:
        if itm == "Negative":
            negNum = negNum + 1
        if itm == "Positive":
            posNum = posNum + 1
    if negNum > posNum:
        res = -1
    if negNum < posNum:
        res = +1
    if negNum == posNum:
        res = 0

    return res


def predictTrend(TSS):

    TS = TSS
    w = 5
    _mICs = rRDataPreparation1(TS, 5)
    horizon = 5
    # _mICs    = mTS.values
    _ICsSize = np.shape(TS)

    _TRNhorizon  = math.floor(_ICsSize[0]-horizon)
    trnXA, tstXA = _mICs[:_TRNhorizon,:w-1], _mICs[_TRNhorizon:,:w-1]
    trnYA, tstYA = _mICs[:_TRNhorizon,w-1], _mICs[_TRNhorizon:,w-1]

    regA = MLPRegressor(hidden_layer_sizes=(10), activation='relu', solver='lbfgs', max_iter=2000)
    regA.fit(trnXA,trnYA)
    predictedValsA = regA.predict(tstXA)

    # print(predictedValsA)
    return predictedValsA


fileName = 'tmp.json'
twitter  = Twython(app_key='IyjZdGpXENZdomu9S8mq0PGD7', app_secret='GboheknKYS4wVeoXWtWgq3dWR6HbhlIbSqqnPN0EHU9kxv5z3Y')
nlp      = StanfordCoreNLP('http://localhost:9000')

fp = codecs.open(fileName, 'r', encoding='utf-8')

i = 0

totalRes = []

totalResW = []

wLen = 10


counter = 0
for objj in ijson.items(fp, ''):
    for o in objj:
        # try:
        counter = counter + 1
        outList = cooOffline(o['text'])
        print('Tweet: ')
        print(o['text'])
        print('Sentiment: ' + numRep(outList))
        singleRess = numRepNum(outList)
        totalRes.append(singleRess)

        mLen = len(totalRes)
        if (len(totalRes)>wLen):
            mLen = wLen

        w = sum(totalRes[-mLen:])/wLen
        totalResW.append(w)
        # print(w)
        # print('+++++++')
        theTrData = 0
        if counter > 10:
            theTrData = counter
            # print('000000000000000000000')
            ch = np.array(totalResW)
            # print(ch)
            # print(type(ch))
            # print(type((totalResW[-theTrData:])))
            # print((ch[-theTrData:]))
            # print(theTrData)
            # print(totalResW[-theTrData])
            # sss = input("go")
            predA = predictTrend(ch[-theTrData:])
            print('Estimated future trend (-1:All negative, +1: All positive]: --> ' + str(predA))
        print('---')
        # except:
        #     fertt = 0


#
# my_df = pd.DataFrame(totalResW)
#
# theNFileName = 'out.csv'
#
# my_df.to_csv(theNFileName, index=False, header=False)





# mTS = read_csv(theNFileName, header=0, parse_dates=[0])
#
# TSS = mTS.values

