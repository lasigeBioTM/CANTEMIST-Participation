# -*- coding: utf-8 -*-
"""
Stemming utilities

@author: André Neves
"""

import string
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

def check_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
def set_stemmer(language):
    """Defines a snowball stemmer for a given language. 
    @Requires: 
        language = chosen language by the user.
    @Ensures: 
        A SnowballStemmer for the chosen language. 
    """
    stemmer = SnowballStemmer(language)
    return stemmer

def list_stemming(l_stem, stemmer):
    """Applies a previously defined stemmer to a given list of texts
    @Requires: 
        l_stem = list of texts to be stemmed
        stemmer = nltk stemmer
    @Ensures:
        A list of stemmed texts.
    """
    for i in range(len(l_stem)):
        if isinstance(l_stem[i], list): #When handling the MER terms
            l_stem[i] = str(l_stem[i]).replace('[', '').replace(']', '').replace('\'', '')
            
        l_stem[i] = stemming(l_stem[i], stemmer)
    return l_stem
    
def stemming(text, stemmer):
    """Stems the text
    @Requires:
        text = text to be stemmed
        stemmer = nltk stemmer
    @Ensures:
        stemmed texted
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    stem_sentence = [stemmer.stem(i) for i in word_tokenize(text)]

    return ' '.join(stem_sentence)