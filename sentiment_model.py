import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

st.title("📊 Sentiment Analysis using NLP")
st.write("Analyze the sentiment of your text")

user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    if user_input:
        score = sia.polarity_scores(user_input)
        compound = score['compound']

        if compound >= 0.05:
            sentiment = "😊 Positive Sentiment"
        elif compound <= -0.05:
            sentiment = "😡 Negative Sentiment"
        else:
            sentiment = "😐 Neutral Sentiment"

        st.success(sentiment)

        st.write("Detailed Scores:", score)