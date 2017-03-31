#! /usr/bin/env python3

import nltk
import csv
import re
import stanfordParser

processedList = []
conceptsGlobal = []

#to find concepts
def dependencyParseTree():
    print('inside dependency parser tree')
    global processedList, conceptsGlobal

    for line in processedList:
        try:
            conceptsGlobal.append(stanfordParser.parseDepTree(line))
        except:
            pass
        #concepts.append()
        
    #check for concepts formed
    print('concepts: ' , conceptsGlobal)

    #write to output file
    writeToFile()
    
#function to write to a file
def writeTofile():

    global conceptsGlobal

    outFile = open(filePath, "w")
    outFile.write(conceptsGlobal)

    outFile.close()

#function to convert data set into lower case
def convert_to_LowerCase(s):
    s= s.lower()
    return(s)

#function to remove the URLs
def remove_url_html(s):
    s = ''.join(re.sub("<.*?>", "", s))
    s= re.sub(r"(?:\@|https?\://)\S+", "", s)
    
    return(s)

#function to replace special values in the string by a space
def replaceValues(s):
    
    s= s.replace("\\n", " ")
    s= s.replace("\\xc2", " ")
    s= s.replace("\\xa0", " ")
    s= s.encode('ascii', 'ignore').decode('utf-8')
    
    return(s)

#function to expand the short forms
def expand_Short_Forms(s):
    
    mapping_open= open('Expansion.csv')
    csv_mapping= csv.reader(mapping_open)
    
    result={}
    
    for row in csv_mapping:
        key= row[0]
        result[key] = row[1]
    
    pattern= re.compile(r'\b(' + '|'. join(result.keys()) + r')\b')
    s= pattern.sub(lambda x: result[x.group()], s)
    return(s)

#modified on : 28th march 2015
#copying the training set into a new set
def trainSet(filePath):
    
    localList = []
    refined_train = []
    lineCount = 0
    
    train_open= open(filePath)
    csv_train= csv.reader(train_open)
    
    for row in csv_train:
        if lineCount == 0:
            lineCount = lineCount + 1
            continue
        
        localList = [row[0], row[2]]
        refined_train.append(localList)
        lineCount = lineCount + 1
    return refined_train

#function to pre process the data
def pre_process(refined_train, filePath):
    
    print('Pre Processing starts here')
    outFile = open(filePath, "w")
    
    for row in refined_train:
        #****************************************PRE PROCESSING START HERE**************************************
        #Convert to lower case
        row[1] = convert_to_LowerCase(row[1])
        
        #Expanding short forms in the training set
        row[1] = expand_Short_Forms(row[1])
        
        #Removing Encoding from the training set
        row[1] = replaceValues(row[1])
        
        #Removing HTML tags with an empty string
        row[1] = remove_url_html(row[1])
        
        processedList.append(row[1]);


#function to run main code
def run():
    
    global conceptsGlobal
    
    #copy entire training set into a list
    trainList = trainSet('train.csv')
    #Pre Process the train data
    pre_process(trainList, 'conceptTrainOut.csv')
    #to extract concepts
    dependencyParseTree();
    #copy entire test set into a list the test data
    testList = trainSet('test.csv')
    #PreProcess the Test data
    pre_process(testList, 'conceptTestOut.csv')


#function to control code
def main():
    
    print('************************INSIDE MAIN*************************')
    
    run()

#Execution will start from here
main()