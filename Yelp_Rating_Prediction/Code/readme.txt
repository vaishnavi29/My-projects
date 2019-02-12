Python 3.5 was used (other versions of python should work too) with the following libraries:

pandas 
numpy
sklearn
pickle
sklearn
random


1. To run our best algorithm please run main.py with the following steps (Please note, that running main.py will replicate our results submitted on Kaggle):
	a. Please place 'train_reviews.csv','business.csv', 'users.csv', 'test_queries.csv' in the same folder as main.py
	b. If you have input files that are named differently from the above, you just need to replace them accordingly in the following lines of code in main.py
		line29:33
			training_file = 'train_reviews.csv'
			businesses_file = 'business.csv'
			users_file = 'users.csv'
			test_file = 'test_queries.csv' 
			output_file = 'out_test_final_test.csv'

	c. Please ensure data_cleaning.py is in the same folder as main.py
	d. the output csv will be saved by our code to the same folder containing main.py.

2. To run our mean predictor algorithm please run Mean_Predictor.py with the following steps:
	a. Please place 'train_reviews.csv','business.csv', 'users.csv', 'test_queries.csv' in the same folder as main.py
	b. If you have input files that are named differently from the above, you just need to replace them accordingly in the following lines of code in main.py
		line29:33
			training_file = 'train_reviews.csv'
			businesses_file = 'business.csv'
			users_file = 'users.csv'
			test_file = 'test_queries.csv' 
			output_file = 'out_test_final_test.csv'

	c. Please ensure data_cleaning.py is in the same folder as main.py
	d. the output csv will be saved by our code to the same folder containing main.py.

3. To run our collaborative filtering algorithm please run Collab_filtering.py with the following steps:
	a. Please place 'train_reviews.csv','business.csv', 'users.csv', 'test_queries.csv' in the same folder as main.py
	b. If you have input files that are named differently from the above, you just need to replace them accordingly in the following lines of code in main.py
		line29:33
			training_file = 'train_reviews.csv'
			businesses_file = 'business.csv'
			users_file = 'users.csv'
			test_file = 'test_queries.csv' 
			output_file = 'out_test_final_test.csv'

	c. Please ensure data_cleaning.py is in the same folder as main.py
	d. the output csv will be saved by our code to the same folder containing main.py.

4. To run our Hybrid Recommender System:
	The program for the Hybrid Recommender System uses GraphLab API with python2.7 for building the recommender system.

        Installation Instructions for GraphLab-Create can be found at :
        https://turi.com/download/install-graphlab-create-command-line.html

        Input Files : 
	train_reviews2.csv (includes categories of each business_id)
	test_queries.csv

	The input files must be in the same directory as yelp_dataset__restaurant_recommender.py

        Output File : rec.pkl (can be copied into a .csv file)
5. The python files review_text_rmse.py and review_text_training.py implement data cleaning and feature extraction for the review text feature in "train_reviews.csv". The features "stars" from "business.csv", "average_stars" from "users.csv", and "business_id" from "business.csv" are also extracted and used.

The read_text_training.py script then uses the features (review_text, business_id, stars, average_stars) to train several different models (one at a time) and predict using them. The lines 41 to 56 are lines that choose the model to use, so just comment and uncomment one model at once to use them.

The predictions are outputted after the read_text_training.py script is run, and then review_text_rmse.py can be run to calculate the RMSE to evaluate the model. 
