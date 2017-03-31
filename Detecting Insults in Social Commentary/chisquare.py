import nltk
import re
import csv
import string
import numpy as np
from time import time
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
from sklearn.feature_selection import SelectKBest, chi2


#trainData = open('C:/Users/Dell/Desktop/Project_Nisha/PreProcess/PPCanon G3.csv')
trainData = open('trainOut.csv')
testData = open('testOut.csv')
train = csv.reader(trainData)
test = csv.reader(testData)
y_train=[]
x_train = []
y_test=[]
x_test =[]

for row in train:
    y_train.append(row[0])
    x_train.append(row[1])

for row in test:
    y_test.append(row[0])
    x_test.append(row[1])
print(y_test[0])
print(x_test[0])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

X_new_counts = count_vect.transform(x_test)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)

feature_names = count_vect.get_feature_names()
t0 = time()
ch2 = SelectKBest(chi2,k='all')
X_train = ch2.fit_transform(X_train_tfidf, y_train)
X_test = ch2.transform(X_new_counts)
if feature_names:
    # keep selected feature names
    feature_names = [feature_names[i] for i
                        in ch2.get_support(indices=True)]
print("done in %fs" % (time() - t0))
#print(list(feature_names).count)

print('_' * 80)
print("Training: ")
#clf = MultinomialNB()
clf= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5)
#clf= SGDClassifier(loss='log')
#print(clf)
t0 = time()
clf.fit(X_train, y_train)
train_time = time() - t0
print("train time: %0.3fs" % train_time)

t0 = time()
pred = clf.predict(X_test)
test_time = time() - t0
print("test time:  %0.3fs" % test_time)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)






