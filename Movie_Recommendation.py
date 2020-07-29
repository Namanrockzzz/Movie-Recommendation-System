"""
Movie Recommendation System
@author - Naman Taneja
"""

# Keypoints
# How does a movie recommendation systen work at Google
# Can be used as a template for any kind of recommendation system like a Book Recommendation System

# Importing the required libraries Libraries
import numpy as np
import pandas as pd
import warnings

# Removing warnings if any
warnings.filterwarnings('ignore')

# Reading a datasets
column_names = ["user_id", "item_id", "rating", "timestamp"]
# u.data conatins user_id, item_id, rating and timestamp
df = pd.read_csv("ml-100k/u.data", sep='\t', names=column_names)
# u.item contains name of the specifications of movie like title, item_id, Name of the director etc.
movies_titles = pd.read_csv("ml-100k/u.item", sep = '\|', header = None)

# Taking out only the required part from the dataset movie_titles
movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id','title']

# Merging titles of movies with dataframe
df = pd.merge(df,movies_titles,on = 'item_id')

# Making DataFrame for average rating of movies
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])

# Adding count to the dataframe
ratings['no of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])

# Making a 2-D matrix which contains name of the movie as columns and their ratings as rows
movie_matrix = df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')

# Making a function to predict recommended movies
def predict_movies(movie_name):
    movie_user_rating = movie_matrix[movie_name]
    similar_to_movie = movie_matrix.corrwith(movie_user_rating)
    
    corr_movie = pd.DataFrame(similar_to_movie, columns = ['correlation'])
    corr_movie.dropna(inplace = True)
    
    corr_movie = corr_movie.join(ratings['no of ratings'])
    predictions = corr_movie[corr_movie['no of ratings']>100].sort_values('correlation', ascending = False)
    
    return predictions

# A Testcase to see top 10 recommended movies for Titanic (1997)
predictions = predict_movies('Titanic (1997)')
print(predictions.head(n=10))

# Check out with different examples

# so now if we search for titanic on google 
# then movies which have high correlation with titanic will also be recommended
# high correlation means closer to one as much as possible