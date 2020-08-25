# App steps:

# 1. Ask user to input name of movie
# 2. Search database for similar movies, using functions for NLP and clustering
# 3. Return a series of movies that the app recommends based on the title inputted
# 4. Option to input a different movie name

# Aditional things to implement
# * Ability to type movie titles in lower cases as well as upper case
# * Ability to recommend actual movie title if title is typed wrong


import pandas as pd
import numpy as np
from os import path

from acquire import prepare_data
from prepare import prep_readme_data
import model

from pprint import pprint

# Create a function to ask for the user's input

def what_is_name_movie():
    movie_title = input("Please enter a movie title: ")
    # movie_title = movie_title.lower().replace(",","")
    return movie_title

# Need to read excisting database, if database not exist, it needs to create a new one

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

        recommended_movies = recommended_movies[["title", "director", "year", "avg_vote"]].head(25)

        recommended_movies = recommended_movies.rename(columns={"title": "Movie Title", "director": "Director", "year": "Year Released", "avg_vote": "IMBd Overall Rating"})
        
        return recommended_movies

    else:
        return "Movie Not in the Database or Not Spelled Correctly"
        

def read_df(movie_title):

    if path.exists("recommendation_data.csv") == False:
        df = prepare_data()
        df = prep_readme_data(df, "description")
        df = model.simple_cluster(df, 5)
        df = model.complex_cluster(df, 5)
        # df.title = df.title.str.lower()
    
        df.to_csv("recommendation_data.csv")

    else:
        df = pd.read_csv("recommendation_data.csv")

    movie_recommendations = complex_movie_recommendation(df, movie_title)

    return movie_recommendations




# # app loop

active = True

print("")
print("Welcome to the movie recommendation app!")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

while active:
    movie_title = what_is_name_movie()
    movie_recommendation = read_df(movie_title)

    if len(movie_recommendation) != 50:
        print("")
        print(f"These are the movie we found that are similar to {movie_title}: ")
        print(movie_recommendation.to_string(index=False))

    else:
        print("")
        print(movie_recommendation)

    print("")
    print("")
    print("")

    another_round = input("Would you us to recommend based on a different movie? (y/n): ")
    if another_round != "y":
        print("----------------")
        print("Thank you for using our movie recommender - come back any time!")
        active = False
    
