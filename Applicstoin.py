import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import pandas as pd
import plotly.express as px

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


st.set_page_config(page_title="Sentiment Analyzer", layout="centered")


st.markdown("""
<style>

/* Animated gradient base */
body {
    background: linear-gradient(-45deg, #1f1c2c, #928dab, #ff6a00, #ee0979);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    overflow: hidden;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating particles */
body::before {
    content: "";
    position: fixed;
    width: 200%;
    height: 200%;
    background: radial-gradient(white 1px, transparent 1px);
    background-size: 40px 40px;
    animation: moveParticles 20s linear infinite;
    opacity: 0.2;
}

@keyframes moveParticles {
    from { transform: translate(0, 0); }
    to { transform: translate(-200px, -200px); }
}

/* Glass UI container */
.main {
    background: rgba(0, 0, 0, 0.6);
    padding: 20px;
    border-radius: 15px;
}

/* Text area */
.stTextArea textarea {
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 12px;
}

/* Button */
.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(45deg, #24c6dc, #514a9d);
}

/* Result box */
.stSuccess {
    border-radius: 12px;
    background-color: rgba(0, 255, 150, 0.2);
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🌌 Sentiment Analysis using NLP")
st.markdown("### ✨ Experience intelligent text analysis")

# Input
user_input = st.text_area("Enter your text:")

# Button
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

        st.markdown("## 🔍 Result")
        st.success(sentiment)

        # 📊 PIE CHART
        df = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [score['pos'], score['neu'], score['neg']]
        })

        fig = px.pie(
            df,
            names='Sentiment',
            values='Score',
            title="Sentiment Distribution",
            color='Sentiment',
            color_discrete_map={
                'Positive': '#00ff9f',
                'Neutral': '#f1c40f',
                'Negative': '#ff4b5c'
            }
        )

        st.plotly_chart(fig)

        # Details
        with st.expander("🔎 Detailed Scores"):
            st.write(score)