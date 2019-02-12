
# Using GraphLab Create APIs
import graphlab as gl
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
#sf = gl.SFrame.read_csv('train_reviews2.csv')
df = pd.read_csv('train_reviews2.csv')

# Let's say I am going to 
sf_all = gl.SFrame(df[['business_id', 'user_id', 'categories', 'stars']])
sf_all.head(2)

"""Train a content-based recommender"""

# set up a category map for all business entities
data = df[['business_id', 'categories']].drop_duplicates()
print(data.shape)
content_sf = gl.SFrame(data = data)

categories = content_sf['categories']
content_sf.remove_column('categories')
content_sf.head(2)

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
vectors = vectorizer.fit_transform(categories).toarray()
words = vectorizer.get_feature_names()

category_NLP = gl.SArray(vectors)
content_sf = content_sf.add_column(category_NLP, 'category_NLP')
category_content_based = gl.recommender.item_content_recommender.create(content_sf, "business_id")

"""### Cascade Hybrid Recommender System
This hybrid recommender system is to make use of both content based and collaborative filtering recommendation system to only evaluate the business entities that are very close to chosen category and predict the ratings based on only these entities.
"""

shape = sf_all.shape
rand_index = random.randint(0, shape[0])
chosen_row = sf_all[rand_index]


# choose 300 closest business entities
choose_k = 10
chosen_k_entities = category_content_based.recommend_from_interactions([chosen_row['business_id']], k = choose_k)
print(chosen_k_entities.head(2))

df_chosen_business_id = chosen_k_entities[['business_id']].to_dataframe()
df_filtered = pd.merge(df, df_chosen_business_id, left_on = 'business_id', right_on = 'business_id', how = 'inner')
df_filtered.head(2)

cf_sf = gl.SFrame(df_filtered[['business_id', 'user_id', 'stars']])
print("TRAINSET",cf_sf.head(2))

print(cf_sf.shape)

train_set = cf_sf
df1 = pd.read_csv('test_queries.csv')
data1 = df1[['business_id', 'user_id']]
cf_sf_t = gl.SFrame(data = data1)
test_set = cf_sf_t




CF_model = gl.item_similarity_recommender.create(train_set,
                                                          user_id = 'user_id', 
                                                          item_id = 'business_id',
                                                          target = 'stars',
                                                          similarity_type = 'pearson',
                                                          threshold = 10**(-9),
                                                          only_top_k = 300,
                                                          verbose = True
                                                         )

#score = gl.evaluation.rmse(test_set['stars'], CF_model.predict(test_set))
#score
print("TEST PRED",list(CF_model.predict(test_set)))
pickle.dump(list(CF_model.predict(test_set)), open('abc.pkl','wb'))
print("PICKLED")


recommend_result = CF_model.recommend(new_observation_data=test_set)

print(recommend_result)
pickle.dump(recommend_result, open('rec.pkl','wb'))

"""
When a (relatively) new user comes, very few ratings has been made, then content-based recommender is used. 
This might on some level solve the 'cold start' problem
"""

category_content_based = gl.recommender.item_content_recommender.create(content_sf, "business_id")

shape = df.shape
random_row = random.randint(0, shape[0])
chosen_user = df.iloc[random_row,].user_id
df_chosen_user = df[df['user_id'] == chosen_user]
print('chosen user is: %s' % chosen_user)
df_chosen_user

high_rate_business_for_user = list(df_chosen_user[df_chosen_user['stars'] >= 3]['business_id'])
high_rate_business_for_user

if (len(high_rate_business_for_user) > 0):
    recommend_result = None
    threshold = 2
    if df_chosen_user.shape[0] >= 10:
        # use item-similarity recommender
        CF_model = gl.item_similarity_recommender.create(sf_all,
                                                         user_id = 'user_id',
                                                         item_id = 'business_id',
                                                         target = 'stars',
                                                         similarity_type = 'pearson',
                                                         threshold = 10**(-9),
                                                         only_top_k = 512,
                                                         verbose = True
                                                        )
        recommend_result = CF_model.recommend(k = 5)
    else:
        # use content-based recommender
        recommend_result = category_content_based.recommend_from_interactions(high_rate_business_for_user, k = 5)
    recommend_df = recommend_result.to_dataframe()
    all_information = pd.merge(df[['user_id','stars','business_id','categories']].drop_duplicates(), recommend_df,
                               how='inner', left_on = 'business_id', right_on= 'business_id')
    print(all_information.head())
    pickle.dump(all_information, open("finalrec1.pkl","wb"))

else:
    print('No quality rating for this user')




