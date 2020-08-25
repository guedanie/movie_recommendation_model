# IMbd Movie Recommendation Model

## Objective: 

To create a movie recommendation tool to recommend other movies based on a specific title. The model will use NLP and clustering to create recommendations.

## Background

Using a data set found on Kaggle, I will be looking at movies published in the US for the past 20+ years, and using NLP strategies to:

1. Explore the data, and see if any interesting patterns arise
2. Create a clustering modeling that can help us identify movies that are similar to each other based on:
    1. Genre
    1. Description
    1. Avg score by viewers
    1. Director
    1. Actors

## Exploration

Some of the questions that I am looking to also answer as part of the exploration are:

1. Who is the most highly grossing director?
1. How have movie genres change over time?
1. Are there any interesting patterns on the year movies are published and how much they gross?
1. Do viwer scores vs gross income work better as an indicator of moview preferences?

## Model:

I did two iterations of the models:

1. Using numerical features for clustering 
1. Using NLP techniques to create clusters based on descriptions and titles.
    * Three models were created under this category - to compare effectiveness.


## Phase II Ideas:

The dataset has amazing information that shows how different demographics (age, gender) scored a movie. It would be great to be able to create a more indepth analysis using this data set, particularly to try to predict the value a movie will gross based on the target audience.

Another option is to create a GUI app where users can actually input a movie tittle, and then recommendations are offered. First phase will likely only be for movie titles, but maybe in the future it is possible to add the option to recommend movies based on movie directors or actors. 