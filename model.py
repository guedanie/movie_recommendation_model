from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split 

from pprint import pprint 

import nltk
import re
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import prepare



# ------------------- #
# Data Representation #
# ------------------- #

    

def clean(text: str, ADDITIONAL_STOPWORDS = ['r', 'u', '2', 'ltgt']) -> list:
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (text.encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split() # tokenization
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


def most_frequent_word(s: pd.Series) -> str:
    '''
    Function that returns the most common word
    '''
    words = clean(' '.join(s))
    most_common_word = pd.Series(words).value_counts().head(1).index
    return most_common_word

def most_frequent_bigram(s: pd.Series) -> str:
    '''
    Function that returns the most common bigram
    '''
    words = clean(' '.join(s))
    most_common_bigram = pd.Series(nltk.bigrams(words)).value_counts().head(1).index
    return most_common_bigram

# --------------- #
#  Bag of Words   #
# --------------- #

def run_bag_of_words(df, target_variable):
    # Bag of words
    cv = CountVectorizer()
    X = cv.fit_transform(df.description.apply(clean).apply(' '.join)) 
    y = df[target_variable]
    return X, y

# -------------- #
#    TF-IDF      #
# -------------- #

def run_tf_idf(df, target_variable):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df.readme_contents.apply(clean).apply(' '.join)) 
    y = df[target_variable]
    return X, y

# ----------------- #
#   Bag of Ngrams   #
# ----------------- #

def run_bag_of_ngrams(df, target_variable, ngram_range):
    cv = CountVectorizer(ngram_range = ngram_range)
    X = cv.fit_transform(df.readme_contents.apply(clean).apply(' '.join))
    y = df[target_variable]
    return X, y 



# --------------- # 
#  Preprocessing  #
# --------------- #

def split_data(X, y, train_size=.75, random_state = 123):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)
    return X_train, X_validate, X_test, y_train, y_validate, y_test

def acquire_data():
    df = pd.read_json("data.json")
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

# _____ Main Preprocessing Function ____ #

def preprocessing(df, data_representation, target_variable, ngram_range = (2,2)):
    
    # df = prepare.prep_readme_data(df)
    df = prepare_data(df)
    

    if data_representation == "bag_of_words":
        X, y = run_bag_of_words(df, target_variable)
    elif data_representation == "tf_idf":
        X, y = run_tf_idf(df, target_variable)
    elif data_representation == "bag_of_ngrams":
        X, y = run_bag_of_ngrams(df, target_variable, ngram_range)
    
    X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(X, y)

    return X_train, X_validate, X_test, y_train, y_validate, y_test



# --------------- # 
#   Clustering    #
# --------------- #

def simple_cluster(df, number_of_clusters):
    df_num = df[["title", "avg_vote", "usa_gross_income", "year", "duration"]]
    df_num = df_num.set_index("title")

    # First, we need to scale the data
    minmax = MinMaxScaler()
    scaled_df = minmax.fit_transform(df_num)

    # Create an instance of KMeans 
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=123)
    # Use fit_predict to cluster the dataset
    predictions = kmeans.fit_predict(scaled_df)

    df["cluster"] = predictions
    df["cluster"] = "cluster_" + df.cluster.astype(str)

    return df

def complex_cluster(df, number_of_clusters):
    df["combined_data"] = df.genre + " " + df.director + " " + df.clean_lemmatized
    df = prepare.prep_readme_data(df, "combined_data")
    cv = CountVectorizer(ngram_range = (1,2))
    cv = cv.fit_transform(df.clean_lemmatized)

    # Create an instance of KMeans to find seven clusters
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=123)
    # Use fit_predict to cluster the dataset
    predictions = kmeans.fit_predict(cv)

    df["cluster_description"] = predictions
    df["cluster_description"] = "cluster_" + df.cluster_description.astype(str)

    return df

    

# -------------------- # 
#   Recommendations    #
# -------------------- #


def simple_movie_recommender(df, test):

    is_in_data = ''
    for i in df['title']:
        if test == i :
            is_in_data = True

    if is_in_data == True:
        index = df[df.title == test].index
        
        genre_type = df[df.title == test].genre.values

        cluster = df[df.title == test].cluster.values

        recommended_movies = df[(df.genre.str.contains(genre_type[0])) & (df.cluster.str.contains(cluster[0]))].sort_values(by="avg_vote", ascending=False)
        
        recommended_movies = recommended_movies[recommended_movies.index != index[0]]

        return recommended_movies[["title", "director", "year", "genre", "avg_vote", "usa_gross_income"]].head(25)

    else:
        return "Movie Not in the Database or Not Spelled Correctly"



def complex_movie_recommendation(df, test):
    is_in_data = ''
    for i in df['title']:
        if test == i :
            is_in_data = True

    if is_in_data == True:
        index = df[df.title == test].index

        genre_type = df[df.title == test].genre.values

        cluster = df[df.title == test].cluster.values

        cluster_desc = df[df.title == test].cluster_description.values

        recommended_movies = (
            df[(df.genre.str.contains(genre_type[0])) 
               & (df.cluster.str.contains(cluster[0]))
               & (df.cluster_description.str.contains(cluster_desc[0]))]
            .sort_values(by="avg_vote", ascending=False)
        )

        recommended_movies = recommended_movies[recommended_movies.index != index[0]]
        
        if len(recommended_movies) == 0:
            recommended_movies = df[(df.genre.str.contains(genre_type[0])) & (df.cluster.str.contains(cluster[0]))].sort_values(by="avg_vote", ascending=False)
        
            recommended_movies = recommended_movies[recommended_movies.index != index[0]]
            
        if len(recommended_movies) == 0:
            return "No Recommendations Found"

        recommended_movies[["title", "director", "year", "genre", "avg_vote", "usa_gross_income"]].head(25)
        return recommended_movies[["title", "director", "year", "genre", "avg_vote", "usa_gross_income"]].head(25)

    else:
        return "Movie Not in the Database or Not Spelled Correctly"