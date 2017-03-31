#! /usr/bin/env python3

import os
import getConcepts
import nltk
import re

from nltk.corpus import wordnet as wn
from nltk.parse import stanford
from nltk.stem import SnowballStemmer
from nltk.tree import *

stemmerS = SnowballStemmer('english')

os.environ['STANFORD_PARSER'] = '/Users/veronica/Documents/python/jars'
os.environ['STANFORD_MODELS'] = '/Users/veronica/Documents/python/jars'

parser = stanford.StanfordParser(model_path="/Users/veronica/Documents/python/script/englishPCFG.ser.gz")

cmd = [
       'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
       'model', parser.model_path,
       '-sentences', 'newline',
       '-outputFormat', 'penn,typed Dependencies',
       ]

sPOS = []
globalPOS_NP = []
globalPOS = []

#check whether parametric value is punctuation or not
def is_punctuation(string):
    for char in string:
        if char.isalpha():
            return False
    return True

#to determine whether the word is stopword in english dictionary or not
def is_stopWord(word):
    if word.lower() in nltk.corpus.stopwords.words('english'):
        return True
    return False

#to stem a word using snowball stemmmer
def stem_Word(word):
    return (stemmerS.stem(word))

#to find all forms of verb
def verbForms(pos):
    comp = re.compile('VB.*')
    vg = comp.findall(pos)
    return vg

#function to parse dependancy tree
def parseDepTree(sentence):
    print ('Inside parseDepTree function of stanfordParser.py')
    global globalPOS
    
    tree = list(parser.raw_parse(sentence))
    
    itList = iter(tree)
    depTree = next(itList)

    t = Tree.fromstring(str(depTree))
    #print ('tree type: ' , t)

    #find concepts particularly for Noun Phrase only
    for np in t.subtrees( lambda t:t.label()=='NP'):
        
        #print ('np: ', np)
        for category in np.subtrees( lambda np:np.height() == 2):
            localPOS = [str(category.label()),str(category.leaves()[0])]

            if is_punctuation(localPOS[1]):
                continue
            
            if is_stopWord(localPOS[1]):
                if ((not(localPOS[0]=='TO')) and (not(localPOS[0]=='IN'))):
                    continue

            globalPOS_NP.append(localPOS)
                
    #print('pos list: ', globalPOS_NP)
    return (getConcepts.getConceptsNP(globalPOS_NP))

    #find concepts for verb phrase also
    for np in t.subtrees( lambda t:t.height() == 2):
        localPOS = [str(np.label()),str(np.leaves()[0])]
        sPOS.append(localPOS)
    
    for ls in sPOS:
        
        vf = verbForms(ls[0])
        
        if is_punctuation(ls[1]):
            continue
        if is_stopWord(ls[1]):
            if ((not(ls[0]=='TO')) and (not(ls[0]=='IN'))):
                continue
        if vf:
            ls[1] = stem_Word(ls[1])

        globalPOS.append(ls)

    getConcepts.getConcepts_All(globalPOS)
