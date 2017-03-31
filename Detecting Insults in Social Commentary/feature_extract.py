#! /usr/bin/env python3

import nltk
import re
import csv
import string
import numpy as np

from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#Define Global Variables
sentTrainList = []
reviewTrainList = []
sentTestList = []
reviewTestList = []
conceptTrainList = []
conceptTestList = []
taggedTrainList = []
taggedTestList = []

#function to read a training/testing file
def readFile(filePath):
    
    localList = []
    
    file_open = open(filePath, 'r')
    csv_read = csv.reader(file_open)

    for row in csv_read:
        localList.append(str(row[0]) + str(row[1]))

    return localList

#function to read training data
def readData(filePath):
    print ('Inside readData function')

    file_open = open(filePath, 'r')
    csv_read = csv.reader(file_open)
    
    sent1 = []
    review1 = []

    for row in csv_read:
        review1.append(str(row[0]))
        sent1.append(str(row[1]))

    return sent1, review1

#function to POS TAG the input data
def POS_TAG():

    trainPOS = []
    testPOS = []
    taggedTestList = []
    outFile = open('taggedTrainPOS', "w")
    outFile = open('taggedTestPOS', "w")

    for line in sentTrainList:
        tagged = nltk.pos_tag(word_tokenize(line))
        
        for i in tagged:
            taggedTrainList = []
            taggedTrainList.append(tagged[0] + '_' + tagged[1] + ' ')
        trainPOS.append(taggedTrainList)
    
    for line in sentTestList:
        tagged = nltk.pos_tag(word_tokenize(line))
        
        for i in sentTestList:
            taggedTestList = []
            taggedTestList.append(tagged[0] + '_' + tagged[1] + ' ')
        testPOS.append(taggedTestList)

    for i in trainPOS:
        outFile.write(i + '\n')
    
    for i in testPOS:
        outFile.write(i + '\n')

    #print ('taggedTrainList: ', taggedTrainList)
    #print ('taggedTestList: ', taggedTestList)
    
    return taggedTrainList, taggedTestList

#function to extract 'unigram' features and perform 'Naive Bayes' classification
def unigram_NB_function():

    global sentTrainList, reviewTrainList, reviewTestList, sentTestList

    UnigramNB = Pipeline([('vect', CountVectorizer(min_df=1)),('tfidf', TfidfTransformer()),('clf', GaussianNB())])
    UnigramNB = UnigramNB.fit(sentTrainList, reviewTrainList)
    predictedUnigramNB = UnigramNB.predict(sentTestList)
    accuracyUnigramNB = np.mean(predictedUnigramNB == reviewTestList)
    print('unigram + naive bayes accuracy: ', accuracyUnigramNB)
    #print('features: ', predictedUnigramNB.toarray().shape)

#function to extract 'bigram' features and perform 'Naive Bayes' classification
def bigram_NB_function():

    global sentTrainList, reviewTrainList, sentTestList, reviewTestList

    BigramNB = Pipeline([('vect', CountVectorizer(ngram_range=(2,2), min_df=1)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    BigramNB = BigramNB.fit(sentTrainList, reviewTrainList)
    predictedBiNB = BigramNB.predict(sentTestList)
    accuracyBigramNB = np.mean(predictedBiNB == reviewTestList)
    print('bigram + naive bayes accuracy: ',accuracyBigramNB)

#function to extract 'unigram + Bigram' features and perform 'Naive Bayes' classification
def uniBiGram_NB_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList

    UniBiNB = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    UniBiNB = UniBiNB.fit(sentTrainList, reviewTrainList)
    predictedUniBiNB = UniBiNB.predict(sentTestList)
    accuracyUniBiNB = np.mean(predictedUniBiNB == reviewTestList)
    print('unigram plus bigram + naive bayes accuracy: ',accuracyUniBiNB)

#function to extract 'unigram' features and perform 'SVM' classification
def unigram_SVM_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList

    UnigramSvm = Pipeline([('vect', CountVectorizer(min_df=1)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])
    UnigramSvm = UnigramSvm.fit(sentTrainList, reviewTrainList)
    predictedUnigramSvm = UnigramSvm.predict(sentTestList)
    accuracyUnigramSvm = np.mean(predictedUnigramSvm == reviewTestList)
    print('unigram + SVM: ', accuracyUnigramSvm)

#function to extract 'bigram' features and perform 'SVM' classification
def bigram_SVM_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList

    BigramSvm = Pipeline([('vect', CountVectorizer(ngram_range=(2,2), min_df=1)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])
    BigramSvm= BigramSvm.fit(sentTrainList, reviewTrainList)
    predictedBigramSvm = BigramSvm.predict(sentTestList)
    accuracyBigramSvm = np.mean(predictedBigramSvm == reviewTestList)
    print('bigram + SVM: ',accuracyBigramSvm)

#function to extract 'unigram + bigram' features and perfrom 'SVM' classification
def uniBiGram_SVM_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList

    UniBiSvm = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])
    UniBiSvm = UniBiSvm.fit(sentTrainList, reviewTrainList)
    predictedUniBiSvm = UniBiSvm.predict(sentTestList)
    accuracyUniBiSvm = np.mean(predictedUniBiSvm == reviewTestList)
    print('unigram plus bigram + SVM: ',accuracyUniBiSvm)

#function to extract 'trigram' features and perform 'Naive Bayes' classification
def triGram_NB_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    TriGramNB = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    TriGramNB = TriGramNB.fit(sentTrainList, reviewTrainList)
    predictedTriGramNB = TriGramNB.predict(sentTestList)
    accuracyTriGramNB = np.mean(predictedTriGramNB == reviewTestList)
    print('Trigram + naive bayes accuracy: ',accuracyTriGramNB)

#function to extract 'trigram' features and perfrom 'SVM' classification
def triGram_SVM_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    TrigramSvm = Pipeline([('vect', CountVectorizer(ngram_range=(3,3),token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])
    TrigramSvm = TrigramSvm.fit(sentTrainList, reviewTrainList)
    predictedTrigramSvm = TrigramSvm.predict(sentTestList)
    accuracyTrigramSvm = np.mean(predictedTrigramSvm == reviewTestList)
    print('Trigram + SVM: ',accuracyTrigramSvm)

#function to extract 'unigram' features based on feature presence and perform Naive Bayes Classification
def uniFP_NB_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    unigramNB = MultinomialNB()
    
    presenceTrain = featurePresence(sentTrainList)
    presenceTest = featurePresence(sentTestList)
    
    print ('type of array: ', len(presenceTest))
    unigramNB.fit(presenceTrain, reviewTrainList)
    predictedUniBiSvm = unigramNB.predict(presenceTest)
    #accuracyUnigramNB = np.mean(predictedUniBiSvm == reviewTestList)
    #print('unigram plus bigram + SVM: ',accuracyUnigramNB)

#function to find features based on feature presence (0 or 1) or frequency
def featurePresence(List):

    fitArray = (CountVectorizer(min_df=1)).fit_transform(List).toarray()
    np.clip(fitArray,0,1, out = fitArray)

    return fitArray

#function to extract 'concepts' features and perform 'Naive Bayes' classification
def concepts_NB_function():
    
    global  reviewTestList, reviewTrainList, conceptTestList, conceptTrainList
    
    conceptNB = MultinomialNB()
    conceptNB = conceptNB.fit(conceptTrainList, reviewTrainList)
    predictedUnigramNB = conceptNB.predict(conceptTestList)
    #accuracyUnigramNB = np.mean(predictedUnigramNB == reviewTestList)
    #print(accuracyUnigramNB)

#function to extract 'unigram plus POS TAG' and perform 'Naive Bayes' classification
def uniPOS_NB_function():

    global taggedTestList, taggedTrainList, reviewTrainList, reviewTestList
    print ('inside uniPOS_NB_function')

    UniTagNB = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
    UniTagNB = UniTagNB.fit(taggedTrainList, reviewTrainList)
    predictedUniTagNB = UniTagNB.predict(taggedTestList)


    #accuracyUniBiNB = np.mean(predictedUniBiNB == reviewTestList)
    #print(accuracyUniBiNB)

#function to extract 'bigram plus POS TAG' and perform 'SVM' classification
def uniPOS_SVM_function():

    global sentTrainList, sentTestList, reviewTrainList, reviewTestList

    #UniBiSvm = Pipeline([('vect', CountVectorizer(ngram_range=(1,1),token_pattern=r'\b\w+\b', min_df=1)),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)),])
    
    vectorizer = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w+\b', min_df=1)
    tfidf = TfidfTransformer()
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)
    
    uniPOSTrainVect = vectorizer.fit_transform(sentTrainList)
    
    uniPOSTFIdf = tfidf.fit_transform(uniPOSVect)
    clf = clf.fit(uniPOSTFIdf,reviewTrainList)
    
    print ('clf: ' , clf)
    #UniBiSvm = UniBiSvm.fit(sentTrainList, reviewTrainList)
    predictedclf = clf.predict(sentTestList)
    #accuracyUniBiSvm = np.mean(predictedUniBiSvm == reviewTrainList)
    #print('unigram plus bigram + SVM: ',accuracyUniBiSvm)

#function to extract 'unigram' and perform 'LR' classification
def unigram_LR_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    UnigramLR = Pipeline([('vect', CountVectorizer(min_df=1)),('tfidf', TfidfTransformer()),('clf', LogisticRegression(penalty='l2',tol=1e-3, C=1.0, fit_intercept=True)),])
    UnigramLR = UnigramLR.fit(sentTrainList, reviewTrainList)
    predictedUnigramLR = UnigramLR.predict(sentTestList)
    accuracyUnigramLR = np.mean(predictedUnigramLR == reviewTestList)
    print('unigram + LR: ',accuracyUnigramLR)

#function to extract 'bigram' and perform 'LR' classification
def bigram_LR_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    bigramLR = Pipeline([('vect', CountVectorizer(ngram_range=(2,2), token_pattern=r'\b\w+\b', min_df=1)),('clf', LogisticRegression(penalty='l2',tol=1e-3, C=1.0, fit_intercept=True)),])
    bigramLR = bigramLR.fit(sentTrainList, reviewTrainList)
    predictedBigramLR = bigramLR.predict(sentTestList)
    print ('predictedBigramLR: ' , predictedBigramLR)
    accuracyBigramLR = np.mean(predictedBigramLR == reviewTestList)
    print('Bigram + LR: ',accuracyBigramLR)

#function to extract 'unigram plus bigram' and perform 'LR' classification
def uniBIGram_LR_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    unibigramLR = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1)),('clf', LogisticRegression(penalty='l2',tol=1e-3, C=1.0, fit_intercept=True)),])
    unibigramLR = unibigramLR.fit(sentTrainList, reviewTrainList)
    predictedUniBigramLR = unibigramLR.predict(sentTestList)
    accuracyUniBigramLR = np.mean(predictedUniBigramLR == reviewTestList)
    print('Unigram plus Bigram + LR: ',accuracyUniBigramLR)

#function to extract 'trigram' and perform 'LR' classification
def triGram_LR_function():
    
    global sentTrainList, sentTestList, reviewTrainList, reviewTestList
    
    trigramLR = Pipeline([('vect', CountVectorizer(ngram_range=(3,3), token_pattern=r'\b\w+\b', min_df=1)),('clf', LogisticRegression(penalty='l2',tol=1e-3, C=1.0, fit_intercept=True)),])
    trigramLR = trigramLR.fit(sentTrainList, reviewTrainList)
    predictedtrigramLR = trigramLR.predict(sentTestList)
    accuracytrigramLR = np.mean(predictedtrigramLR == reviewTestList)
    print('Trigram + LR: ',accuracytrigramLR)

#main function
def main():
    
    global sentTrainList, reviewTrainList, sentTestList, reviewTestList, conceptList, taggedList
    
    #used to read train output file
    sentTrainList, reviewTrainList = readData('trainOut.csv')
    
    #used to read test output file
    sentTestList, reviewTestList = readData('testOut.csv')
    
    #used to pos tag the input data
    #taggedTrainList, taggedTestList = POS_TAG()
    
    #unigram Tf-idf and Naive Bayes Classifier
    unigram_NB_function()

    #bigram Tf-idf and Naive Bayes Classifier
    bigram_NB_function()

    #unigram + Bigram Tf-idf and Naive Bayes
    uniBiGram_NB_function()

    #unigram Tf-idf and SVM
    unigram_SVM_function()

    #bigram Tf-idf and SVM
    bigram_SVM_function()

    #unigram + bigram Tf-idf SVM
    uniBiGram_SVM_function()
    
    #trigram + SVM
    triGram_SVM_function()

    #trigram + NB
    triGram_NB_function()

    #unigram + LR
    unigram_LR_function()

    #Bigram + LR
    bigram_LR_function()

    #unigram + bigram + LR
    uniBIGram_LR_function()

    #trigram + LR
    triGram_LR_function()

    #unigram 0/1 Naive Bayes
    #uniFP_NB_function()

    #concepts extraction
    #conceptTrainList = readData('conceptTrainOut.csv')
    #conceptTestList = readData('conceptTestOut.csv')
    #concept_NB_function()
    
    #unigram plus POS-TAG and Naive Bayes
    #uniPOS_NB_function()

    #unigram plus POS-TAG and SVM
    #uniPOS_SVM_function()

#Execution will start from here
main()








