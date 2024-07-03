#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=column_names)
df = df.drop('timestamp', axis=1)
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')
from sklearn.metrics.pairwise import cosine_similarity

# Fill missing values with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)



def predict_ratings(user_item_matrix, user_similarity):
    pred = np.zeros(user_item_matrix.shape)
    for i in range(user_item_matrix.shape[0]):
        for j in range(user_item_matrix.shape[1]):
            # If the user hasn't rated the item
            if user_item_matrix.iloc[i, j] == 0:
                # Find similar users who have rated the item
                similar_users = user_similarity.iloc[i]
                user_ratings = user_item_matrix.iloc[:, j]
                # Calculate weighted sum
                weighted_sum = np.dot(similar_users, user_ratings)
                sum_of_weights = np.sum(np.abs(similar_users))
                # Predict rating
                pred[i, j] = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
            else:
                pred[i, j] = user_item_matrix.iloc[i, j]
    return pred

# Predict ratings
predicted_ratings = predict_ratings(user_item_matrix_filled, user_similarity_df)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

def recommend_items(predicted_ratings, user_id, num_recommendations):
    user_ratings = predicted_ratings.loc[user_id]
    # Sort items by predicted ratings in descending order
    recommended_items = user_ratings.sort_values(ascending=False)
    return recommended_items.head(num_recommendations)

# Get recommendations for a specific user
user_id = 1
num_recommendations = 5
recommendations = recommend_items(predicted_ratings_df, user_id, num_recommendations)


# In[5]:


## Verify 

print(user_similarity_df.head())


# In[6]:


print(f"Recommendations for User {user_id}:\n{recommendations}")


# In[ ]:




