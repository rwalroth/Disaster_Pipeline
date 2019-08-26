import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import numpy as np


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    word_vectors = TfidfVectorizer(tokenizer=tokenize).fit_transform(df['message'])
    pca = TruncatedSVD().fit_transform(word_vectors)
    pca_df = pd.DataFrame(pca, columns=['LSA 1', 'LSA 2'])
    pca_df['cat'] = np.zeros_like(pca[:,0])
    Y = df.drop(['message','original','genre', 'id'], axis=1)
    for i in Y.index:
        a = '0.' + ''.join(list(reversed([str(x) for x in Y.loc[i]])))
        pca_df['cat'][i] = float(a)
    pca_df['cat'] = (pca_df['cat'] - pca_df['cat'].min())/(pca_df['cat'].max() - pca_df['cat'].min())
    totals = Y.sum()
    cats = list(totals.index)
    counts = list(totals)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cats,
                    y=counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=pca[:,0],
                    y=pca[:,1],
                    mode='markers+text',
                    marker = dict(
                        color=pca_df['cat'],
                        colorscale='Viridis',
                        opacity=0.1
                    )
                )
            ],

            'layout': {
                'title': 'LSA Analysis of Messages, color coded by category',
                'yaxis': {
                    'title': "First LSA component"
                },
                'xaxis': {
                    'title': "Second LSA component<br><br>Colors correspond to combinations of categories for each message, the number of combinations<br> of categories is represented in the spread of colors but not conducive to a simple legend"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()