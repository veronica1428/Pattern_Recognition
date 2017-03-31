import csv
import numpy as np
import nltk

from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

filePath = 'trainOut.csv'
trainFile = open(filePath)
csv_train = csv.reader(trainFile)

sentenceList = []
for line in csv_train:
    sentenceList.append(line[1])

#---------------UNIGRAM + POS Tagging-------------
for line in sentenceList:
    print('hiii', line)
    tagged = nltk.pos_tag(word_tokenize(line))
    print ('tagged: ' , tagged)

vectorizer = CountVectorizer(min_df = 1)

#----------------   UNIGRAM TF-IDF Naive Bayes Classifier   ------------
#Rows contains number of sentences and Column contains number of Unique words in sorted lexical order
array_data = vectorizer.fit_transform(sentenceList).toarray()
#print ('array_data', array_data)
print ('unigram array length : ' , array_data.shape)
vocab = vectorizer.get_feature_names()
#print('feature names: ', vocab)
#print(vectorizer.vocabulary_.get('fuck'))


#Array enteries greater than 1 all set to 1
#np.clip(array_data, 0 ,1, out = array_data)
#print ('out array_data: ', array_data)
freq = np.sum(array_data, axis = 1)
print ('freq: ', freq)
print('type: ', freq.shape)

#---------------UNIGRAM and BI-GRAM---------------
bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print(analyze('bi-grams are cool!'))
array_data2 = bigram_vectorizer.fit_transform(sentenceList).toarray()
print ('length of bigram array: ', array_data2.shape)
#print('bigram array data: ', array_data2 )
vocab2 = bigram_vectorizer.get_feature_names()
print('bigram vocab:', vocab2)


#--------------TF-IDF-----------------------------
transformer = TfidfTransformer()
print ('transformer: ' , transformer)

tfidf_Uni = transformer.fit_transform(freq).toarray()

print('tfidf to unigram: ', tfidf_Uni)
#print('tfidf to bigram: ', tfidf.fit_transform(array_data2))

#-----------------------Feature Selection------------------------
#for unigram approach
#SelectKBEst(chi2, k=2).fit_transform


'''
for line in csv_train:
    
    print ('line[1]:  ', line[1])
    train_count = count_vect.fit_transform(line[1])
    
    print(train_count)
'''