import streamlit as st
from chatbot import chatbot_response

st.title("🤖 NLP Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.success("Bot: " + response)