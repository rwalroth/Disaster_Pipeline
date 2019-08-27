# Disaster Response Pipeline Project

This project aims to provide a simple web app for analyzing text messages in a disaster response context and aid in identifying what category a message belongs to. Text messages are passed through a TF-IDF vectorizer, and a random forrest classifier is employed to classify messages. Under /data is a python script called process_data which takes in csv files and prepares them for the machine learning algorithm. Under /models, the train_classifier script optimizes and trains the classifier. Code for the plotly visualizations and the back end is found under /app/run.py. 

Dependencies:
standard python library (3.6 or greater)
sklearn
numpy
pandas
sqlalchemy
nltk
numpy
matplotlib
seaborn
json
plotly
flask

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
