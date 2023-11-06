from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import json
import plotly
import plotly.express as px

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)

        if score["neg"] != 0:
            return render_template('index.html', message="Negative😔😔")
        else:
            sentiment = pd.DataFrame(
                score, columns=['positive', 'negative'], index=['a'])
            fig = px.bar(sentiment, x=sentiment.index, y=['positive', 'negative'], barmode='group')
            graphJSON = fig.to_json()  # Convert Plotly figure to JSON
            return render_template('index.html', message="Positive!😃😃", graphJSON=graphJSON)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
