from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import Ridge
import pickle
from sklearn.linear_model import LogisticRegression
from data_cleaning import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import os
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from find_mse import *
from sklearn.linear_model import LogisticRegression
import random

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


training_file = 'train_reviews.csv'
businesses_file = 'business.csv'
users_file = 'users.csv'
test_file = 'test_queries.csv' #or validate_queries.csv
output_file = 'out_test_final_test.csv'

ratings = pd.read_csv(training_file)
print("no of training data", ratings.shape)

businesses = pd.read_csv(businesses_file)
print("no of bus", businesses.shape)

users = pd.read_csv(users_file,parse_dates=['yelping_since'])
#users = None

#users = np.load('users.pkl')
print("no of users",users.shape)
#pickle.dump(users, open('users.pkl', 'wb'))


values = {'funny': 0,'useful':0,'review_count':0, 'average_stars':0}
users.fillna(value=values, inplace=True)

values = {'attributes_RestaurantsPriceRange2': 1,'attributes_BusinessAcceptsCreditCards':False,'stars':0, 'attributes_Alcohol':'none','attributes_RestaurantsAttire':'casual',
          'attributes_WheelchairAccessible':False,'categories':'Restaurants','attributes_WiFi':'no', 'attributes_BusinessParking':'{\'garage\': False, ''street\': False, \'validated\': False, \'lot\': False, \'valet\': False}',
          'attributes_GoodForMeal':'{\'dessert\': False, \'latenight\': False, \'lunch\': False, \'dinner\': True, \'breakfast\': False, \'brunch\': False}','attributes_RestaurantsTakeOut':False,
          'attributes_RestaurantsReservations':False,'attributes_DogsAllowed':False,'attributes_Ambience':'{\'romantic\': False, \'intimate\': False, \'classy\': False, \'hipster\': False, \'divey\' : False,  \'touristy\': False, \'trendy\': False, \'upscale\': False, \'casual\': False}',
          'attributes_BikeParking':False,'attributes_GoodForKids':False,'attributes_NoiseLevel':'average','attributes_OutdoorSeating':False,'attributes_RestaurantsGoodForGroups':False,
          'attributes_RestaurantsDelivery': False,'attributes_RestaurantsTableService':False,'attributes_WheelchairAccessible':False,'attributes_Caters':False,'attributes_HasTV':False,
          'neighborhood':'The Strip','attributes_RestaurantsCounterService':False, 'attributes_Open24Hours':False,'attributes_HappyHour':False, 'attributes_GoodForDancing':False,'attributes_DriveThru':False,
          'attributes_DogsAllowed':False,'attributes_Corkage':False,'attributes_CoatCheck':False,'attributes_ByAppointmentOnly':False,'attributes_BusinessAcceptsBitcoin':False,
          'attributes_BYOB':False,'attributes_AcceptsInsurance':False,'name':'In-N-Out Burger','neighborhood':'The Strip','categories':'unknown',
          'attributes_Music':'unknown', 'attributes_DietaryRestrictions':'unknown', 'attributes_BestNights' :'unknown','attributes_BYOBCorkage' :'unknown','attributes_AgesAllowed':'unknown'}
businesses.fillna(value=values, inplace=True)

users = users[users['average_stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
users = get_time_feature(users, 'yelping_since')
users,dummy = get_one_hot_encoded(users, 'friends')
users,dummy = get_one_hot_encoded(users, 'elite')

cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_BusinessParking',['garage_BP','street_BP','validated_BP','Lot_BP','valet_BP' ])

cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_GoodForMeal',['dessert','latenight','lunch','dinner','breakfast','brunch'])
cleaned_businesses = get_mulitple_features_from_one(businesses,'attributes_Ambience',['romantic', 'intimate', 'classy', 'hipster','divey', 'touristy', 'trendy', 'upscale', 'casual'])


cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BusinessAcceptsCreditCards')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BikeParking')

cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsTakeOut')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsReservations')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_DogsAllowed')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_GoodForKids')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_OutdoorSeating')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsGoodForGroups')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsDelivery')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsTableService')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_WheelchairAccessible')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_Caters')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_HasTV')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_RestaurantsCounterService')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_Open24Hours')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_HappyHour')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_GoodForDancing')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_DriveThru')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_DogsAllowed')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_Corkage')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_CoatCheck')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_ByAppointmentOnly')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BusinessAcceptsBitcoin')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_BYOB')
cleaned_businesses = clean_boolean_feature(businesses, 'attributes_AcceptsInsurance')


cleaned_businesses,default_value_of_attributes_Alcohol = get_one_hot_encoded(businesses, 'attributes_Alcohol','none')
cleaned_businesses,default_value_of_RestaurantsAttire = get_one_hot_encoded(businesses, 'attributes_RestaurantsAttire','casual')
#cleaned_businesses,default_value_of_WheelchairAccessible = get_one_hot_encoded(businesses, 'attributes_WheelchairAccessible',False)
cleaned_businesses,default_value_of_attributes_WiFi = get_one_hot_encoded(businesses, 'attributes_WiFi','no')
cleaned_businesses,default_value_of_attributes_NoiseLevel = get_one_hot_encoded(businesses, 'attributes_NoiseLevel','average')
cleaned_businesses,default_value_of_city = get_one_hot_encoded(businesses, 'city','Las Vegas')
cleaned_businesses,default_value_of_name = get_one_hot_encoded(businesses, 'name','In-N-Out Burger')
cleaned_businesses,default_value_of_postal_code = get_one_hot_encoded(businesses, 'postal_code','89109')
cleaned_businesses,default_value_of_neighborhood = get_one_hot_encoded(businesses, 'neighborhood','The Strip')
cleaned_businesses,default_value_of_categories = get_one_hot_encoded(businesses, 'categories','Restaurants, Mexican')
cleaned_businesses,dummy = get_one_hot_encoded(businesses, 'attributes_Music')
cleaned_businesses,dumy = get_one_hot_encoded(businesses, 'attributes_DietaryRestrictions')
cleaned_businesses,dumy = get_one_hot_encoded(businesses, 'attributes_BestNights')
cleaned_businesses,dumy = get_one_hot_encoded(businesses, 'attributes_BYOBCorkage')
cleaned_businesses,dumy = get_one_hot_encoded(businesses, 'attributes_AgesAllowed')

cleaned_businesses,default_value_of_hours_Sunday = get_one_hot_encoded(businesses, 'hours_Sunday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Monday = get_one_hot_encoded(businesses, 'hours_Monday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Tuesday = get_one_hot_encoded(businesses, 'hours_Tuesday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Wednesday = get_one_hot_encoded(businesses, 'hours_Wednesday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Thursday = get_one_hot_encoded(businesses, 'hours_Thursday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Friday = get_one_hot_encoded(businesses, 'hours_Friday','11:0-22:0')
cleaned_businesses,default_value_of_hours_Saturday = get_one_hot_encoded(businesses, 'hours_Saturday','11:0-22:0')


#cleaned_businesses,default_value_of_categories = one_hot_encoded_multiclass(businesses, 'categories','Restaurants')

'''
default_value_of_attributes_Alcohol = None
default_value_of_RestaurantsAttire = None
default_value_of_WheelchairAccessible = None
default_value_of_attributes_WiFi = None
default_value_of_attributes_NoiseLevel = None
default_value_of_categories = None
'''


u_stars = users['average_stars']
u_stars = u_stars.astype(float)
avg_user_rating= np.mean(u_stars)

businesses = businesses[businesses['stars'].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
b_stars = businesses['stars']
b_stars = b_stars.astype(float)
mean_bussinesses_rating = np.mean(b_stars)

def dump_value(featurename, value):
    pickle.dump(value,open(featurename+'.pkl', 'wb'))

def scale_feautres(features):
    scaler = StandardScaler()
    feature = features.reshape(-1,1)
    scaled_feature = scaler.fit_transform(feature)
    features = scaled_feature.reshape(-1)
    return features

def extract_feautres(input_data,train_or_test,saved_feautres= None, return_subset_of_features = False):
    feautres_of_train_data = np.empty(shape=(input_data.shape[0],89))

    if saved_feautres is not None and os.path.exists(saved_feautres):
        print("Found cached features..")
        feautres_of_train_data_new = []
        feautres_of_train_data = np.load(saved_feautres)

        feautres_of_train_data_new = np.array(feautres_of_train_data_new).transpose()
        print("Returning cached features..")
    else:
        print("Extracting all features..")
        #feautres_of_train_data = []
        print(input_data.shape)
        print(len(input_data.axes[0]))
        for index in range(0,len(input_data.axes[0])):
        #for index, train in input_data.iterrows():
            train = input_data.iloc[index ]
            features_extracted = []
            user_id = train.user_id
            b_id = train.business_id
            user_data = users[users.user_id == user_id]
            if(user_data.empty):
                funny_feautre = 0
                feautre_useful = 0
                review_count = 0
                feature_star = avg_user_rating
                yelping_since = 0
                friends = 0
                fans = 0
                elite = 0
                cool = 0
                compliment_writer = 0
                compliment_profile = 0
                compliment_plain = 0
                compliment_photos = 0
                compliment_note = 0
                compliment_more = 0
                compliment_list = 0
                compliment_hot = 0
                compliment_funny = 0
                compliment_cute = 0
                compliment_cool = 0
            else:
                funny_feautre = user_data.funny.values[0]
                feautre_useful = user_data.useful.values[0]
                review_count= user_data.review_count.values[0]
                yelping_since= user_data.yelping_since.values[0]

                friends= user_data.friends.values[0]
                fans= user_data.fans.values[0]
                elite= user_data.elite.values[0]
                cool= user_data.cool.values[0]
                compliment_writer= user_data.compliment_writer.values[0]
                compliment_profile= user_data.compliment_profile.values[0]
                compliment_plain= user_data.compliment_plain.values[0]
                compliment_photos= user_data.compliment_photos.values[0]
                compliment_note= user_data.compliment_note.values[0]
                compliment_more= user_data.compliment_more.values[0]
                compliment_list= user_data.compliment_list.values[0]
                compliment_hot= user_data.compliment_hot.values[0]
                compliment_funny= user_data.compliment_funny.values[0]
                compliment_cute= user_data.compliment_cute.values[0]
                compliment_cool= user_data.compliment_cool.values[0]
                if(user_data.average_stars.values[0] == 0):
                    feature_star = avg_user_rating
                else:
                    feature_star = user_data.average_stars.values[0]
            b_data = cleaned_businesses[cleaned_businesses.business_id == b_id]
            if (b_data.empty):
                feature_star_business = mean_bussinesses_rating
                attributes_RestaurantsPriceRange2 = 1
                attributes_Alcohol = default_value_of_attributes_Alcohol
                attributes_BusinessAcceptsCreditCards = 0
                attributes_RestaurantsAttire = default_value_of_RestaurantsAttire
                attributes_WheelchairAccessible = False
                #categories = default_value_of_categories
                attributes_WiFi = default_value_of_attributes_WiFi
                attributes_RestaurantsTakeOut = False
                attributes_RestaurantsReservations = False
                attributes_DogsAllowed = False
                attributes_BikeParking = False
                attributes_GoodForKids = False
                attributes_NoiseLevel =default_value_of_attributes_NoiseLevel
                attributes_OutdoorSeating = False
                attributes_RestaurantsGoodForGroups = False
                attributes_RestaurantsDelivery = False
                attributes_RestaurantsTableService = False
                garage_BP = -1
                street_BP = -1
                validated_BP = -1
                Lot_BP = -1
                valet_BP = -1
                dessert  = -1
                latenight, =-1
                lunch =-1
                dinner = -1
                breakfast = -1
                brunch = -1
                romantic = -1
                intimate = -1
                classy = -1
                hipster = -1
                touristy = -1
                trendy = -1
                upscale =-1
                casual = 1
                attributes_Caters = False
                attributes_HasTV = False
                city = 209
                is_open = 0
                hours_Sunday = default_value_of_hours_Sunday
                hours_Monday = default_value_of_hours_Monday
                hours_Tuesday = default_value_of_hours_Tuesday
                hours_Wednesday = default_value_of_hours_Wednesday
                hours_Thursday = default_value_of_hours_Thursday
                hours_Friday = default_value_of_hours_Friday
                hours_Saturday = default_value_of_hours_Saturday

                review_count_b = 0
                longitude = 0
                latitude = 0
                attributes_RestaurantsCounterService=False
                attributes_Open24Hours=False
                attributes_HappyHour=False
                attributes_GoodForDancing=False
                attributes_DriveThru=False
                attributes_Corkage=False
                attributes_CoatCheck=False
                attributes_ByAppointmentOnly=False
                attributes_BusinessAcceptsBitcoin=False
                attributes_BYOB=False
                attributes_AcceptsInsurance=False
                name = 'In-N-Out Burger'
                neighborhood=  'The Strip'
                categories = 'unknown'
                attributes_Music = 'unknown'
                attributes_DietaryRestrictions = 'unknown'
                attributes_BestNights = 'unknown'
                attributes_BYOBCorkage = 'unknown'
                attributes_AgesAllowed = 'unknown'

            else:

                attributes_RestaurantsPriceRange2 = b_data.attributes_RestaurantsPriceRange2.values[0]
                attributes_BusinessAcceptsCreditCards = b_data.attributes_BusinessAcceptsCreditCards.values[0]
                attributes_Alcohol = b_data.attributes_Alcohol.values[0]
                attributes_RestaurantsAttire = b_data.attributes_RestaurantsAttire.values[0]
                attributes_WheelchairAccessible = b_data.attributes_WheelchairAccessible.values[0]
                #categories = b_data.categories.values[0]
                attributes_WiFi = b_data.attributes_WiFi.values[0]
                attributes_RestaurantsTakeOut = b_data.attributes_RestaurantsTakeOut.values[0]
                attributes_RestaurantsReservations = b_data.attributes_RestaurantsReservations.values[0]
                attributes_DogsAllowed = b_data.attributes_DogsAllowed.values[0]
                attributes_BikeParking = b_data.attributes_BikeParking.values[0]
                attributes_GoodForKids = b_data.attributes_GoodForKids.values[0]
                attributes_NoiseLevel = b_data.attributes_NoiseLevel.values[0]
                attributes_OutdoorSeating = b_data.attributes_OutdoorSeating.values[0]
                attributes_RestaurantsGoodForGroups = b_data.attributes_RestaurantsGoodForGroups.values[0]
                attributes_RestaurantsDelivery = b_data.attributes_RestaurantsDelivery.values[0]
                attributes_RestaurantsTableService = b_data.attributes_RestaurantsTableService.values[0]

                garage_BP = b_data.garage_BP.values[0]
                street_BP =b_data.street_BP.values[0]
                validated_BP = b_data.validated_BP.values[0]
                Lot_BP = b_data.Lot_BP.values[0]
                valet_BP =b_data.valet_BP.values[0]

                dessert = b_data.dessert.values[0]
                latenight = b_data.latenight.values[0]
                lunch = b_data.lunch.values[0]
                dinner = b_data.dinner.values[0]
                breakfast = b_data.breakfast.values[0]

                brunch = b_data.brunch.values[0]
                romantic = b_data.romantic.values[0]
                intimate = b_data.intimate.values[0]
                classy = b_data.classy.values[0]

                hipster = b_data.hipster.values[0]
                touristy = b_data.touristy.values[0]
                trendy = b_data.trendy.values[0]
                upscale = b_data.upscale.values[0]
                casual = b_data.casual.values[0]

                attributes_Caters = b_data.attributes_Caters.values[0]
                attributes_HasTV = b_data.attributes_HasTV.values[0]
                city = b_data.city.values[0]
                is_open = b_data.is_open.values[0]
                hours_Sunday = b_data.hours_Sunday.values[0]
                hours_Monday = b_data.hours_Monday.values[0]
                hours_Tuesday = b_data.hours_Tuesday.values[0]
                hours_Wednesday = b_data.hours_Wednesday.values[0]
                hours_Thursday = b_data.hours_Thursday.values[0]
                hours_Friday = b_data.hours_Friday.values[0]
                hours_Saturday = b_data.hours_Saturday.values[0]

                review_count_b = b_data.review_count.values[0]
                longitude = b_data.longitude.values[0]
                latitude = b_data.latitude.values[0]
                attributes_RestaurantsCounterService = b_data.attributes_RestaurantsCounterService.values[0]
                attributes_Open24Hours = b_data.attributes_Open24Hours.values[0]
                attributes_HappyHour = b_data.attributes_HappyHour.values[0]
                attributes_GoodForDancing = b_data.attributes_GoodForDancing.values[0]
                attributes_DriveThru = b_data.attributes_DriveThru.values[0]
                attributes_Corkage = b_data.attributes_Corkage.values[0]
                attributes_CoatCheck = b_data.attributes_CoatCheck.values[0]
                attributes_ByAppointmentOnly = b_data.attributes_ByAppointmentOnly.values[0]
                attributes_BusinessAcceptsBitcoin = b_data.attributes_BusinessAcceptsBitcoin.values[0]
                attributes_BYOB = b_data.attributes_BYOB.values[0]
                attributes_AcceptsInsurance = b_data.attributes_AcceptsInsurance.values[0]
                name = b_data.name.values[0]
                neighborhood = b_data.neighborhood.values[0]
                categories = b_data.categories.values[0]
                attributes_Music =b_data.attributes_Music.values[0]
                attributes_DietaryRestrictions = b_data.attributes_DietaryRestrictions.values[0]
                attributes_BestNights = b_data.attributes_BestNights.values[0]
                attributes_BYOBCorkage = b_data.attributes_BYOBCorkage.values[0]
                attributes_AgesAllowed = b_data.attributes_AgesAllowed.values[0]

                if (b_data.stars.values[0] == 0):
                    feature_star_business = mean_bussinesses_rating
                else:
                    feature_star_business = b_data.stars.values[0]

            features_extracted.append(feature_star) #0
            features_extracted.append(feature_star_business)#1

            features_extracted.append(funny_feautre)#2
            features_extracted.append(feautre_useful)#3
            features_extracted.append(review_count)#4
            features_extracted.append(attributes_RestaurantsPriceRange2)

            features_extracted.append(attributes_BusinessAcceptsCreditCards)#6
            features_extracted.append(attributes_Alcohol)
            features_extracted.append(attributes_RestaurantsAttire)
            features_extracted.append(attributes_WheelchairAccessible)
            #features_extracted.append(categories)
            features_extracted.append(attributes_WiFi)#10
            features_extracted.append(attributes_RestaurantsTakeOut)
            features_extracted.append(attributes_RestaurantsReservations)
            features_extracted.append(attributes_DogsAllowed)
            features_extracted.append(attributes_BikeParking)
            features_extracted.append(attributes_GoodForKids)#15
            features_extracted.append(attributes_NoiseLevel)
            features_extracted.append(attributes_OutdoorSeating)
            features_extracted.append(attributes_RestaurantsGoodForGroups)
            features_extracted.append(attributes_RestaurantsDelivery)
            features_extracted.append(attributes_RestaurantsTableService)#20
            features_extracted.append(garage_BP )
            features_extracted.append(street_BP )
            features_extracted.append(validated_BP )
            features_extracted.append(Lot_BP )
            features_extracted.append(valet_BP)#25
            features_extracted.append(dessert)
            features_extracted.append(latenight)
            features_extracted.append(lunch)
            features_extracted.append(dinner)
            features_extracted.append(breakfast)#30

            features_extracted.append(brunch)
            features_extracted.append(romantic)
            features_extracted.append(intimate)
            features_extracted.append(classy)

            features_extracted.append(hipster)
            features_extracted.append(touristy)
            features_extracted.append(trendy)
            features_extracted.append(upscale)#38
            features_extracted.append(casual)

            features_extracted.append(attributes_Caters)
            features_extracted.append(attributes_HasTV)
            features_extracted.append(city)
            features_extracted.append(is_open)

            features_extracted.append(hours_Sunday)#44
            features_extracted.append(hours_Monday)
            features_extracted.append(hours_Tuesday)
            features_extracted.append(hours_Wednesday)
            features_extracted.append(hours_Thursday)
            features_extracted.append(hours_Friday)
            features_extracted.append(hours_Saturday)#50

            features_extracted.append(yelping_since)
            features_extracted.append(friends)
            features_extracted.append(fans)
            features_extracted.append(elite)#54
            features_extracted.append(cool)
            features_extracted.append(compliment_writer)
            features_extracted.append(compliment_profile)
            features_extracted.append(compliment_plain)
            features_extracted.append(compliment_photos)
            features_extracted.append(compliment_note)#60
            features_extracted.append(compliment_more)
            features_extracted.append(compliment_list)
            features_extracted.append(compliment_hot)
            features_extracted.append(compliment_funny)
            features_extracted.append(compliment_cute)
            features_extracted.append(compliment_cool)

            features_extracted.append(review_count_b )
            features_extracted.append(longitude )
            features_extracted.append(latitude )
            features_extracted.append(attributes_RestaurantsCounterService )#70
            features_extracted.append(attributes_Open24Hours )
            features_extracted.append(attributes_HappyHour )
            features_extracted.append(attributes_GoodForDancing )
            features_extracted.append(attributes_DriveThru )
            features_extracted.append(attributes_Corkage )
            features_extracted.append(attributes_CoatCheck )
            features_extracted.append(attributes_ByAppointmentOnly )
            features_extracted.append(attributes_BusinessAcceptsBitcoin )
            features_extracted.append(attributes_BYOB )
            features_extracted.append(attributes_AcceptsInsurance )#80
            features_extracted.append(name)
            features_extracted.append(neighborhood )
            features_extracted.append(categories )
            features_extracted.append(attributes_Music )
            features_extracted.append(attributes_DietaryRestrictions )
            features_extracted.append(attributes_BestNights )
            features_extracted.append(attributes_BYOBCorkage )
            features_extracted.append(attributes_AgesAllowed )#88


            #dump_value('categories',categories)
            feautres_of_train_data[index,:] = np.array(features_extracted)

        #pickle.dump(feautres_of_train_data, open(saved_feautres, 'wb'))
        #feautres_of_train_data = np.array(feautres_of_train_data)
        pickle.dump(feautres_of_train_data, open(saved_feautres, 'wb'))

        #for i in range(0,feautres_of_train_data.shape[1]):
            #dump_value('feature_' + str(i) + train_or_test, feautres_of_train_data[:, i])


    for i in range(0, feautres_of_train_data.shape[1]):
        feautres_of_train_data[:, i] = scale_feautres(feautres_of_train_data[:, i])
    if(return_subset_of_features == True):
        for i in range(0, feautres_of_train_data_new.shape[1]):
            feautres_of_train_data_new[:, i] = scale_feautres(feautres_of_train_data_new[:, i])

        print("Returning features..")
        return feautres_of_train_data,feautres_of_train_data_new
    else:
        return feautres_of_train_data, None

#feautres_of_train_data = np.load('feautres_of_train_data.pkl')

#feautres_of_train_data,feautres_of_train_data_new = extract_feautres( ratings,'train', 'feautres_of_train_data_set3.pkl')

feautres_of_train_data,feautres_of_train_data_new = extract_feautres( ratings,'train', 'feautres_of_train_data_set4.pkl')
stars = ratings['stars']
print(stars.shape)

#test_data  = pd.read_csv(test_file, dtype={'user_id': str})
test_data  = pd.read_csv('test_with_gt.csv', dtype={'user_id': str})
test_data = test_data.drop(test_data[test_data.user_id == '#NAME?'].index)
test_data = test_data.drop(test_data[test_data.business_id == '#NAME?'].index)
test_data.to_csv('test_with_gt_cleaned.csv')

print(test_data.shape)
indices = []
print("Finding test data feautres..")
feautres_of_test_data,feautres_of_test_data_new = extract_feautres( test_data, 'test','feautres_of_test_data_set4.pkl')
#feautres_of_test_data,feautres_of_test_data_new = extract_feautres( test_data, 'test','feautres_of_validate_data_set3.pkl')
#feautres_of_test_data,feautres_of_test_data_new = extract_feautres( test_data, 'test','feautres_of_validate_data_set4.pkl')
feautres_of_test_data = feautres_of_test_data[0:test_data.shape[0]]

#clf = Ridge(alpha=1.0)
print("Fitting on train data..", feautres_of_train_data.shape)
#clf = LinearSVC( solver='newton-cg', multi_class='multinomial',n_jobs=-1)
#clf = LinearSVC( )
#clf = RandomForestClassifier()
#clf = BernoulliNB()
#clf =DecisionTreeClassifier()
#clf = KNeighborsClassifier()
#clf = LogisticRegression()
clf = LinearRegression()
no_of_neurons = feautres_of_train_data.shape[1]
#clf = MLPClassifier(hidden_layer_sizes=(no_of_neurons,no_of_neurons,no_of_neurons),max_iter=1000)
#clf = RidgeClassifier(solver='auto')
#clf.fit(feautres_of_train_data_new,stars)


#from here
feautres_of_train_data_new = []
feautres_of_train_data_new.append(feautres_of_train_data[:,0:2])

feautres_of_test_data_new = []
feautres_of_test_data_new.append(feautres_of_test_data[:,0:2])

feautres_of_train_data_final = []
feautres_of_train_data_final.append(feautres_of_train_data[:,0])
feautres_of_train_data_final.append(feautres_of_train_data[:,1])


values1 = [56, 28, 44, 86, 82, 77, 42, 13, 27, 84, 62, 79, 83, 34, 5, 49, 51, 21, 14, 64, 33, 74, 85, 63, 78, 32, 2, 88, 47, 6, 54, 72, 29, 40, 43, 46, 17, 70, 58, 31, 80, 10, 7, 61, 73, 38, 8]
for k in values1 :
    feautres_of_train_data_final.append(feautres_of_train_data[:,k])

feautres_of_train_data_final= np.array(feautres_of_train_data_final)
clf.fit(feautres_of_train_data_final.transpose(), stars)

feautres_of_test_data_final = []

feautres_of_test_data_final.append(feautres_of_test_data[:,0])
feautres_of_test_data_final.append(feautres_of_test_data[:,1])
values1 = [56, 28, 44, 86, 82, 77, 42, 13, 27, 84, 62, 79, 83, 34, 5, 49, 51, 21, 14, 64, 33, 74, 85, 63, 78, 32, 2, 88, 47, 6, 54, 72, 29, 40, 43, 46, 17, 70, 58, 31, 80, 10, 7, 61, 73, 38, 8]
for k in values1 :
    feautres_of_test_data_final.append(feautres_of_test_data[:,k])
feautres_of_test_data_final= np.array(feautres_of_test_data_final)

prediction_values = clf.predict(feautres_of_test_data_final.transpose())
# prediction_values = clf.predict(feautres_of_test_data_new)
# prediction_values = np.round(list(prediction_values))
prediction_values_final = []

test_submission_data = pd.DataFrame(
    {
        'stars': prediction_values,
    })

test_submission_data.to_csv(output_file,index_label = 'index')
#mse = find_mse('out_validate_final.csv')
#to here

#This  is to perform k-fold cross validation to find the best set of features for each algorithm
def k_fold_for_finding_best_features(clf,min_mse,feautres_of_train_data,feautres_of_test_data):

    for k in range(2,feautres_of_train_data.shape[1]):
        #k = 87
        for j in range(0,5):
            no_features_to_selct = random.sample(range(2, feautres_of_train_data.shape[1]), k)
            print("j = ", j)
            print("no_features_to_selct = ",no_features_to_selct )
            new_features_to_train = []
            new_features_to_test = []
            feautres_of_train_data_final = np.zeros(shape=(feautres_of_train_data.shape[0], 2 +k))
            feautres_of_test_data_final = np.zeros(shape=(feautres_of_test_data.shape[0], 2 + k))
            for i in no_features_to_selct:
                    new_features_to_train.append(feautres_of_train_data[:,i])
                    new_features_to_test.append(feautres_of_test_data[:,i])

            feautres_of_train_data_final[:,0:2] = feautres_of_train_data[:,0:2]
            feautres_of_train_data_final[:,2:k+2]  = np.array(new_features_to_train).transpose()
            #feautres_of_train_data_final =   np.array(feautres_of_train_data_final).transpose()
            no_of_neurons = feautres_of_train_data_final.shape[1]
            #clf = MLPClassifier(hidden_layer_sizes=(no_of_neurons, no_of_neurons, no_of_neurons), max_iter=50)
            #clf = DecisionTreeClassifier()
            #clf = LinearRegression()
            clf.fit(feautres_of_train_data_final, stars)

            feautres_of_test_data_final[:, 0:2] = feautres_of_test_data[:, 0:2]
            feautres_of_test_data_final[:, 2:k+2] = np.array(new_features_to_test).transpose()
            #
            print("Predicting on test data feautres..")
            prediction_values = clf.predict(feautres_of_test_data_final)
            #prediction_values = clf.predict(feautres_of_test_data_new)
            #prediction_values = np.round(list(prediction_values))
            prediction_values_final = []

            test_submission_data = pd.DataFrame(
                    {
                        'stars': prediction_values,
                     })
            test_submission_data.to_csv('out_validate_rand' +str(j) + '.csv',index_label = 'index')
            print("Mse for this j is..j = ", j)
            mse = find_mse('out_validate_rand' +str(j) + '.csv')
            #if (mse < 1.1052277293575479):
            if(mse <= min_mse):
                print("mse is", mse)
                min_mse = mse

#This  is to perform k-fold cross validation to find the best set of features for each algorithm
'''
clf = LinearRegression()
clf = KNeighborsClassifier()
clf = MLPClassifier(hidden_layer_sizes=(no_of_neurons, no_of_neurons, no_of_neurons), max_iter=50)
clf = RandomForestClassifier()
clf = DecisionTreeClassifier()
clf = BernoulliNB()
clf = RidgeClassifier()
min_mse = 2
k_fold_for_finding_best_features(clf,min_mse,feautres_of_train_data, feautres_of_test_data)
'''