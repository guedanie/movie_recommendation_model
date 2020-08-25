import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import nltk.sentiment
import re

import warnings
warnings.filterwarnings('ignore')

from prepare import prep_readme_data
import model
import nltk.sentiment

from sklearn.preprocessing import MinMaxScaler


def read_data():
    movie_title = pd.read_csv("IMDb movies.csv")
    return movie_title




def filter_to_usa(movie_title):
    df = movie_title[(movie_title.country == "USA")]
    return df

def handle_null_values(df):
    # Impude language
    df.language = df.language.fillna("English")
    
    # Drop rows with missing description, director, writer
    df = df[df.description.notnull()]
    df = df[df.director.notnull()]
    df = df[df.writer.notnull()]

    # Impude avg budget
    df.budget = df.budget.fillna("$ 0")
    df = df[~df.budget.str.contains("ESP")]

    df = df[~df.budget.str.contains("GBP")]

    df = df[~df.budget.str.contains("CAD")]

    df = df[~df.budget.str.contains("PYG")]

    df = df[~df.budget.str.contains("AUD")]

    df = df[~df.budget.str.contains("EUR")]

    df = df[~df.budget.str.contains("RUR")]

    avg_budget = df.budget.str.replace("$", '').astype(int).mean()

    df.budget = df.budget.str.replace("$", '').astype(int)

    df.budget = df.budget.replace(0, avg_budget)

    # Impude usa_gross_income

    median_income = df[(df.usa_gross_income.notnull()) & (df.usa_gross_income.str.contains("$", regex=False))].usa_gross_income.str.replace("$", '').astype(int).median()

    df.usa_gross_income = df.usa_gross_income.fillna("$ 0")

    df.usa_gross_income = (
    df[df.usa_gross_income.str.contains("$", regex=False)]
    .usa_gross_income.str.replace("$", '')
    .astype(int)
    .replace(0, median_income)
    )

    # Remove columns with too many null values

    df = df.drop(columns=["worlwide_gross_income", "metascore", "reviews_from_users", "reviews_from_critics"])

    # Drop any remaining null values

    df = df.dropna()

    return df


#  __Main Prep Function__

def prepare_data():
    movie_title = read_data()
    df = filter_to_usa(movie_title)
    df = handle_null_values(df)
    return df