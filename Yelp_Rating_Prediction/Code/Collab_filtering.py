import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import os

training_file = 'train_reviews.csv'
businesses_file = 'business.csv'
users_file = 'users.csv'
test_file = 'test_queries.csv' #or validate_queries.csv
output_file = 'out_test_colab.csv'

businesses = pd.read_csv(businesses_file)
print("no of bus", businesses.shape)

users = pd.read_csv(users_file)
#print("no of users",users.shape)

ratings = pd.read_csv(training_file)
print("no of training data", ratings.shape)


#print("no of unique users", users['user_id'].nunique())

n_bussinesses = ratings['business_id'].nunique()
print("no of unique bus", businesses['business_id'].nunique())

n_users = ratings['user_id'].nunique()
n_bussinesses = ratings['business_id'].nunique()
print("no of unique users", ratings['user_id'].nunique())
print("no of unique bus", ratings['business_id'].nunique())

train_data_matrix = np.zeros((n_users, n_bussinesses))

unique_users= ratings.user_id.unique()
unique_businesses= ratings.business_id.unique()

userid_dict = {}
i = 0
for user in unique_users:
    userid_dict[user] = i
    i = i+1

businessid_dict = {}
i = 0
for business in unique_businesses:
    businessid_dict[business] = i
    i = i+1

saved_train_data_matrix = ('saved_train_data_matrix.pkl')
if saved_train_data_matrix is not None and os.path.exists(saved_train_data_matrix):
        print("found train_data_matrix")
        train_data_matrix = np.load(saved_train_data_matrix)
else:
    for index, rating_row in ratings.iterrows():
        user_id = rating_row.user_id
        b_id = rating_row.business_id
        user_id_index =  userid_dict[user_id]
        b_id_index = businessid_dict[b_id]
        train_data_matrix[user_id_index,b_id_index] = rating_row.stars
    pickle.dump(train_data_matrix, open('saved_train_data_matrix.pkl', 'wb'))

saved_user_similarity = ('saved_user_similarity.pkl')
saved_bussiness_similarity = ('saved_bussiness_similarity.pkl')
if saved_user_similarity is not None and os.path.exists(saved_user_similarity):
        print("found cached user")
        user_similarity = np.load(saved_user_similarity)
else:
    print(" not found cached user")
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    pickle.dump(user_similarity, open('saved_user_similarity.pkl', 'wb'))

if saved_bussiness_similarity is not None and os.path.exists(saved_bussiness_similarity):
    print("found cached b")
    bussiness_similarity = np.load(saved_bussiness_similarity)
else:
  print("not found cached b")
  bussiness_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
  pickle.dump(bussiness_similarity, open('saved_bussiness_similarity.pkl', 'wb'))

test_data  = pd.read_csv('test_with_gt.csv')
print(test_data.shape)

# for index, rating_row in test_data.iterrows():
#     user_id = rating_row.user_id
#     b_id = rating_row.business_id
#     user_id_index =  userid_dict[user_id]
#     b_id_index = businessid_dict[b_id]
#     w_of_user = user_similarity[user_id_index]
#
#     mean_user_rating = mean_user_rating = ratings.mean(axis=1).mean(axis=1)
#     ratings_diff = (train_data_matrix - mean_user_rating[:, np.newaxis])
#     user_similarity.dot(ratings_diff)

def predict(ratings, similarity, type='user'):
    if type == 'user':
        #mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        #ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        #pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

        #rating_trimmed = ratings[ratings > 0]
        #mean_user_rating = rating_trimmed.mean(axis=1)
        #mean_user_rating = rating_trimmed.mean(axis=0)
        # We use np.newaxis so that mean_user_rating has same format as ratings
        #ratings_diff = (rating_trimmed- mean_user_rating[:, np.newaxis])
        #ratings_diff = (rating_trimmed - mean_user_rating[:, np.newaxis])
        #pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T


        pred_values =  np.zeros((ratings.shape[0], ratings.shape[1]))
        for i in range(0,ratings.shape[0]):
            for j in range(0,ratings.shape[1]):
                users_who_rated_this_b = [k for k,v in enumerate(ratings[:,j] > 0) if v]
                list_ratings = np.array(ratings[:,j])
                ratings_of_users = list_ratings[users_who_rated_this_b]
                mean_rating = ratings_of_users.mean()
                sum = 0
                sum2 = 0
                for user in users_who_rated_this_b:
                    sum = sum + (list_ratings[user] - mean_rating) * similarity[i,user]
                    sum2  = sum2 + similarity[i,user]
                pred_values[i,j] = mean_rating +( sum/sum2)
        return pred_values

    elif type == 'item':
        pred_values = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred_values

#user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, bussiness_similarity, type='item')

prediction_values_for_user = []
prediction_values_for_b= []
indices = []

users = users[users['average_stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
avg_user_rating= users[['average_stars','user_id']]


mean_bussinesses_rating = businesses[['stars','business_id']]

stars = ratings['stars']
rating_avg = np.mean(stars)
a = avg_user_rating['average_stars']
avg_user_rating['average_stars'] = a - rating_avg
b = mean_bussinesses_rating['stars'].values
mean_bussinesses_rating['stars'] = b - rating_avg

for index, rating_row in test_data.iterrows():
      not_present_user = False
      not_present_b = False
      user_id = rating_row.user_id
      b_id = rating_row.business_id
      if user_id in userid_dict:
        user_id_index =  userid_dict[user_id]
      else:
         not_present_user = True
      if b_id in userid_dict:
          b_id_index = businessid_dict[b_id]
      else:
          not_present_b = True


      #prediction_user = float(round(user_prediction[user_id_index,b_id_index]))
      #if(prediction_user == 0):
          #prediction_user = 1
      #prediction_values_for_user.append(prediction_user)
      '''
      if not_present_user ==  True and not_present_b == False:
          prediction_b = float(round(np.mean((item_prediction[:, b_id_index]))))
      elif not_present_user ==  False and not_present_b == True:
          prediction_b = float(round(np.mean((item_prediction[user_id_index,:]))))  
      elif not_present_user ==  True and not_present_b == True:
      '''
      if not_present_user == True or not_present_b == True:
          user = avg_user_rating[avg_user_rating.user_id == user_id]
          if (user.empty):
              mean_u_star = 0
          else:
              mean_u_star = user.average_stars
              mean_u_star = mean_u_star.values[0]
          busin = mean_bussinesses_rating[mean_bussinesses_rating.business_id == b_id]
          if (busin.empty):
              mean_b_star = 0
          else:
              mean_b_star = busin.stars
              mean_b_star = mean_b_star.values[0]

          prediction = float(round(mean_u_star + mean_b_star + rating_avg))
          prediction_b =prediction
      else:
        prediction_b = float(round(item_prediction[user_id_index,b_id_index]))

      if(prediction_b== 0):
          prediction_b = 1

      prediction_values_for_b.append(prediction_b)
      indices.append(index)
'''
test_submission_data = pd.DataFrame(
        {   'index': indices,
            'stars': prediction_values_for_user,
         })
test_submission_data.to_csv('out_colab_useruser.csv')
'''

test_submission_data = pd.DataFrame(
        {
            'stars': prediction_values_for_b,
         })
test_submission_data.to_csv(output_file,index_label = 'index')