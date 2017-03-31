import csv
import nltk
from nltk.corpus import stopwords
import re
import string
#import unidecode

refined_train = []

#function to remove non ascii characters
def removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)

#copying the training set into a new set
def trainSet():
    
    train_open= open('train.csv')
    csv_train= csv.reader(train_open)

    for row in csv_train:
        refined_train.append(row)

#Function to remove HTML Tags with and empty string
def remove_HTML_Tag(str):
    str = ''.join(re.sub("<.*?>", "", str))
    return str


#function to remove all unicodes from the training set
def removeUnicode(s):
    #return(str.decode('unicode_escape').encode('utf8'))
    s=filter(lambda x:x in string.printable, s)
    s=s.decode('unicode_escape').encode('utf8')
    return s
#cachedStopWords = stopwords.words("english")


#Date : 26 march 2015
#Module: check whether parametric value is punctuation or not
def is_punctuation(string):
    for char in string:
        if char.isalpha():
            return False
    return True

#function to run main code
def run():
    
    #copy entire training set into a list
    trainSet()
    
    for row in refined_train:
       
        #Preprocessing the text
        
        #Removing the stop words
        #row[2] = ' '.join([token for token in nltk.word_tokenize(row[2]) if token.lower() not in cachedStopWords])
        
        #Removing HTML tags with an empty string
        row[2] = remove_HTML_Tag(row[2])

        #remove Unicode from the training set
        #row[2] = removeUnicode(row[2])

        #Removing New Line Character
        #row[2] = row[2].replace('\n', ' ')

        #Feature Extraction using UNIGRAM approach
        unigramExtraction(row[2])

#function to control code
def main():
    print ('************************INSIDE MAIN*************************')
    run()

#Execution will start from here
main()
