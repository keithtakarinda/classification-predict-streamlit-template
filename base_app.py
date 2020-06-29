"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.probability import FreqDist

# Vectorizer
news_vectorizer = open("resources/tfidf_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
#Load decomposed data
dt = pd.read_csv("resources/train_decomp.csv")

#Define function to plot word clouds
def plotwordclouds(text):
    wordcloud = WordCloud(width=1200, height=800, random_state=21,background_color="White",
                          colormap="Blues", max_font_size=110).generate(text)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

#Defire a function to plot most used words
def plot_most_frequent(text,n_words):
    """
    This function plots a bar plot of the words  
    """
    mostcommon_small = FreqDist(text).most_common(n_words)
    x, y = zip(*mostcommon_small)
    plt.figure(figsize=(40,30),tight_layout=True)
    plt.barh(x,y)
    plt.xlabel('Fequency', fontsize=40)
    plt.ylabel('Words', fontsize=40)
    plt.yticks(fontsize=40)
    plt.xticks(rotation=90,fontsize=40)
    plt.show()

#Define function to find the most common words
def most_common_words(text,n_words):
	mostcommon_small = FreqDist(text).most_common(n_words)
	x, y = zip(*mostcommon_small)
	return x,y

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	image = Image.open('resources/twitterimagea.png')
	st.image(image)
	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = [ "Information","EDA","Predictions"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(open("resources/info.md").read())

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "EDA" page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")

		#Cleate a Selection box of word sequences
		word_sequences = st.selectbox("Number of word sequences",("1-Word", "2-Word", "3-Word"))

		if word_sequences == "1-Word":
			cols = "message"

		elif word_sequences == "2-Word":
			cols = 'bigrams_text'

		elif word_sequences == "3-Word":
			cols = "trigrams_text"

		#Cleate a of types of tweets
		tweet_options = ["All","News(2)","Pro(1)","Neutral(0)","Anti(-1)"]
		tweet_selection = st.sidebar.selectbox("Filter Tweets", tweet_options)
		
		#Create lists of words that meet the condition
		if tweet_selection == "All":
			all_words = ' '.join([str(text) for text in dt[cols]])
			allwords = all_words.split(' ')
		
		elif tweet_selection == "News(2)":
			all_words = ' '.join([str(text) for text in dt[dt['sentiment']==2][cols]])
			allwords = all_words.split(' ')

		elif tweet_selection == "Pro(1)":
			all_words = ' '.join([str(text) for text in dt[dt['sentiment']==1][cols]])
			allwords = all_words.split(' ')
		
		elif tweet_selection == "Neutral(0)":
			all_words = ' '.join([str(text) for text in dt[dt['sentiment']==0][cols]])
			allwords = all_words.split(' ')
		
		else :
			all_words = ' '.join([str(text) for text in dt[dt['sentiment']==-1][cols]])
			allwords = all_words.split(' ')
		
		#Create options for different plot types
		plot_options = ["Word Cloud","Bar Chart","Table"]
		plot_selection = st.selectbox("Choose plot", plot_options)

		#Add a subheader	
		st.subheader("Analyse words most commonly used")

		#Plot bar chart
		if plot_selection == "Bar Chart":
			k = st.slider('Number of words', 10, 30,20)
			st.pyplot(plot_most_frequent(allwords,k))

		#Input a table
		if plot_selection == "Table":
			j = st.slider('Number of words', 10, 30,10)
			st.table(most_common_words(allwords,j))

		#Plot a word cloud
		if plot_selection == "Word Cloud":
			st.pyplot(plotwordclouds(all_words))
			st.pyplot()

	# Building out the predication page
	if selection == "Predictions":
		st.info("Predicting Sentiment From Text")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		#Define the models 
		models = ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbours','Support Vector Machine']
		model_selection = st.selectbox('Please Select A Classification Model',models)
		#Import Selected models
		if model_selection  == "Logistic Regression":
			predictor = joblib.load(open(os.path.join("resources/lr_model.pkl"),"rb"))
		if model_selection  == "Random Forest":
			predictor = joblib.load(open(os.path.join("resources/rf_model.pkl"),"rb"))
		if model_selection  == "Support Vector Machine":
			predictor = joblib.load(open(os.path.join("resources/linearsvc_model.pkl"),"rb"))
		if model_selection  == "K-Nearest Neighbours":
			predictor = joblib.load(open(os.path.join("resources/knn_model.pkl"),"rb"))

		#Classify text 
		if st.button("Classify Text"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text.lower()]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			prediction = predictor.predict(vect_text)

			#Define prediction
			if prediction == -1:
				classification = "No belief in man-made climate change"
			elif prediction == 0:
				classification = "Neutral towards man-made climate change"
			elif prediction == 1:
				classification = "Support towards the belief of man-made climate change"
			else:
				classification = "Links to factual news about climate change"
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction) + " " +format(classification))
			
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
