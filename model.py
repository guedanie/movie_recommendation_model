from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint 

import nltk
import re
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
# import prepare



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




# ---------------------- #
#        Modeling        #
# ---------------------- #

# Decision Tree

def run_clf(X_train, y_train, max_depth):
    '''
    Function used to create and fit decision tree models. It requires a max_depth parameter. Returns model and predictions.
    '''
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return clf, y_pred


# KNN

def run_knn(X_train, y_train, n_neighbors):
    '''
    Function used to create and fit KNN model. Requires to specify n_neighbors. Returns model and predictions.
    '''
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    return knn, y_pred

# Random_forest

def run_rf(X_train, y_train, leaf, max_depth):
    ''' 
    Function used to create and fit random forest models. Requires to specif leaf and max_depth. Returns model and predictions.
    '''
    rf = RandomForestClassifier(random_state= 123, min_samples_leaf = leaf, max_depth = max_depth).fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred

# Logistic Regression

def run_lg(X_train, y_train):
    '''
    Function used to create and fit logistic regression models. Returns model and predictions.
    '''
    logit = LogisticRegression().fit(X_train, y_train)
    y_pred = logit.predict(X_train)
    return logit, y_pred

# Native Bayes Multinomial

def run_native_bayes(X_train, y_train, alpha=.1):
    mnb = MultinomialNB(alpha = alpha).fit(X_train, y_train)
    y_pred = mnb.predict(X_train)
    return mnb, y_pred

# Evaluation

def create_report(y_train, y_pred):
    '''
    Helper function used to create a classification evaluation report, and return it as df
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    return report


def accuracy_report(model, y_pred, y_train):
    '''
    Main function used to create printable versions of the classification accuracy score, confusion matrix and classification report.
    '''
    report = classification_report(y_train, y_pred, output_dict = True)
    report = pd.DataFrame.from_dict(report)
    accuracy_score = f'Accuracy on dataset: {report.accuracy[0]:.2f}'

    labels = sorted(y_train.unique())
    matrix = pd.DataFrame(confusion_matrix(y_train, y_pred), index = labels, columns = labels)

    return accuracy_score, matrix, report

# ------------------------- #
#   Creating Prediction     #
# ------------------------- #

def predict_readme_language(readme):
    df = acquire_data()
    df = prepare_data(df)


    cv = CountVectorizer(ngram_range = (1,2))
    X = cv.fit_transform(df.readme_contents.apply(clean).apply(' '.join))
    y = df["is_top_language"]

    X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(X, y)
    
    mnb = MultinomialNB(alpha = 0.7).fit(X_train, y_train)
       
    text = prepare_data(readme)
    
    X = cv.transform(text.readme_contents.apply(clean).apply(' '.join))
    
    prediction = mnb.predict(X)
    
    return prediction

# -------------------------- # 
#   Get Feature Importance   #
# -------------------------- #

def preprocessing_features(model):
    
    df = pd.read_json("data.json")
    # df = prepare.prep_readme_data(df)
    df = prepare_data(df)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df.readme_contents.apply(clean).apply(' '.join)) 
    

    pd.Series(dict(zip(tfidf.get_feature_names(), model.feature_importances_))).sort_values().tail(5).plot.barh(title = "Most important words used for modeling", figsize=(10, 8))

    
    

     