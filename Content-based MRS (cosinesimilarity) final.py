#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dependencies
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# load data
movies = pd.read_csv('movies.csv')

# extract genres data from movies
genres = movies['genres']

# tokenize genres
genre_tokens = [word_tokenize(genre.replace("|", " ")) for genre in genres]

#  apply lemmatizer
lemmatizer = WordNetLemmatizer()
genre_lemmas = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in genre_tokens]

# apply CountVectorizer
cv = CountVectorizer(lowercase=True)
feature_matrix = cv.fit_transform([" ".join(lemmas) for lemmas in genre_lemmas]).toarray()

# extract movieId and title from movies and assign to movies_data
movies_data = movies.loc[:,['movieId','title']]

# group movies_data and feature_matrix together
movies_data = movies_data.join(pd.DataFrame(feature_matrix))

# Compute the pairwise cosine similarity matrix
similarity_matrix = cosine_similarity(feature_matrix)

# Prompt the user to enter partial or full movie title
movie_title = input('Finding similar movies.\nPlease enter the movie title: ')

# Define function to find the closest match to the user's input using fuzzy matching
def get_closest_match(title):
    titles = movies_data['title']
    highest_ratio = 0
    closest_match = ''
    for t in titles:
        ratio = fuzz.ratio(title.lower(), t.lower())
        if ratio > highest_ratio:
            highest_ratio = ratio
            closest_match = t
    return closest_match

# Find closest match
closest_match = get_closest_match(movie_title)
print(f"Closest match found: {closest_match}")

# Define function to derive recommended movie list
def recommend_movies(closest_match, similarity_matrix, movies_data):
    # Find the index of the movie title in the movies_data DataFrame
    movie_index = movies_data.index[movies_data['title'] == closest_match][0]

    # Get the similarity scores of the input movie with all other movies
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top 11 most similar movies (including the input movie)
    top_11_movies = similarity_scores[:11]

    # Get the titles of the top 11 movies
    top_11_titles = [movies_data.iloc[movie[0]]['title'] for movie in top_11_movies]

    # Remove the closest match from the list of recommended movies
    top_10_movies = [title for title in top_11_titles if title != closest_match][:10]

    # Print the recommended movies
    if len(top_10_movies) > 0:
        print(f"Top 10 recommended movies for {closest_match}:")
        for i, movie in enumerate(top_10_movies, start = 1):
            print(f"{i}. {movie}")
    else:
        print(f"No movies found similar to {closest_match}.")

# Printing the recommended movie list
print(recommend_movies(closest_match, similarity_matrix, movies_data))


# In[ ]:




