import unicodedata
import re
import json

import nltk
# nltk.download('wordnet')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
from requests import get

from bs4 import BeautifulSoup
import re

# ---------------- #
#     Prepare      #
# ---------------- #

def basic_clean(text):
    '''
    Helper function that lower_cases the text, removes any special characters or accents
    '''
    article = text.lower()
    article = unicodedata.normalize('NFKD', article)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    article = re.sub(r"[^a-z0-9'\s]", '', article)
    return article 

def tokenize(basic_clean_text):
    '''
    Helper function that tokenizes the data
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    article = tokenizer.tokenize(basic_clean_text, return_str=True)
    return article

def stem(tokenized_text):
    '''
    Helper function that accepts some text and return the text after applying stemming to all words. Also returns stems
    '''
    ps = nltk.stem.porter.PorterStemmer()
    stems = [ps.stem(word) for word in tokenized_text.split()]
    article_stemmed = ' '.join(stems)
    return stems, article_stemmed  

def lemmatize(tokenized_text):
    '''
    Helper function that accept some text and return the text after applying lemmatization to each word.
    '''
    wnl = nltk.stem.WordNetLemmatizer()

    lemmas = [wnl.lemmatize(word) for word in tokenized_text.split()]
    article_lemmatized = ' '.join(lemmas)
    return lemmas, article_lemmatized
    

    
def remove_stopwords(lemmatized_text, extra_words, words_remove):
    '''
    Helper function that accepts text, and retunrs that text after removing all the stopwords.
    Takes two additional arguments. 
    
    `extra_words` :a list of extra words to include in the list of stop words
    `words_remove`: a list of words to remove from the list of stop words
    
    '''

    stopword_list = stopwords.words('english')

    for item in words_remove:
        stopword_list.remove(item)
        
    for item in extra_words:
        stopword_list.append(item)

    words = lemmatized_text.split()
    filtered_words = [w for w in words if w not in stopword_list]
    article_without_stopwords = ' '.join(filtered_words)
    
    return article_without_stopwords
