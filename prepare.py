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


def basic_clean(df, col):
    '''
    This function takes in a df and a string for a column and
    returns the df with a new column named 'basic_clean' with the
    passed column text normalized.
    '''
    df[col] = df[col].str.replace("\n", ' ')

    df['basic_clean'] = df[col].str.lower()\
                    .replace(r'[^\w\s]', '', regex=True)\
                    .str.normalize('NFKC')\
                    .str.encode('ascii', 'ignore')\
                    .str.decode('utf-8', 'ignore')
    
    

    return df


def tokenize(df, col):
    '''
    This function takes in a df and a string for a column and
    returns a df with a new column named 'clean_tokes' with the
    passed column text tokenized and in a list.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    df['clean_tokes'] = df[col].apply(tokenizer.tokenize)
    return df


def lemmatize(df, col):
    '''
    This function takes in a df and a string for column name and
    returns the original df with a new column called 'lemmatized'.
    '''
    # Create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Lemmatize each token from our clean_tokes Series of lists
    lemmas = df[col].apply(lambda row: [wnl.lemmatize(word) for word in row])
    
    # Join the cleaned and lemmatized tokens back into sentences
    df['lemmatized'] = lemmas.str.join(' ')
    return df


def remove_stopwords(df, col):
    '''
    This function takes in a df and a string for column name and 
    returns the df with a new column named 'clean' with stopwords removed.
    '''
    # Create stopword_list
    stopword_list = stopwords.words('english')
    
    # Split words in column
    words = df[col].str.split()
    
    # Check each word in each row of the column against stopword_list and return only those that are not in list
    filtered_words = words.apply(lambda row: [word for word in row if word not in stopword_list])
    
    # Create new column of words that have stopwords removed
    df['clean_' + col] = filtered_words.str.join(' ')
    
    return df

def prepare_data(df):
    """
    This function removes nan's from the language columns 
    Creates a new column called is_top_language
    """
    
    df = df[(~df.readme_contents.str.contains("<p ", na=False)) & (~df.readme_contents.str.contains("<div ", na=False))].dropna()
    df.loc[(df.language != "Python") & (df.language !="Java") & (df.language !="JavaScript") & (df.language !="C++"), 'is_top_language'] = 'other'
    df.is_top_language = df.is_top_language.fillna(df.language)

    return df


def drop_long_words(string, num=12):
    """
    Takes in a string and drops words equal to or less than 
    the number specified, defaults to 12
    """
    new_word = []
    for word in string.split():
        if len(word) <= num:
            new_word.append(word)
    new_word = " ".join(new_word)
    return new_word


def prep_readme_data(df, col_name):
    '''
    This function takes in the github readme df and
    returns the df with original columns plus cleaned
    and lemmatized content without stopwords.
    '''

    
    # Do basic clean on article content
    df = basic_clean(df, col_name)
    
    # Tokenize clean article content
    df = tokenize(df, 'basic_clean')
    
    # Lemmatize cleaned and tokenized article content
    df = lemmatize(df, 'clean_tokes')
    
    # Remove stopwords from Lemmatized article content
    df = remove_stopwords(df, 'lemmatized')

    # Drop long strings
    df.clean_lemmatized = df.clean_lemmatized.apply(drop_long_words)
    
    return df