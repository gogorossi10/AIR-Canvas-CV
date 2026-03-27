import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Sample knowledge base (you can expand this!)
text = """Hello! I am an intelligent chatbot. 
I can help you with natural language processing concepts. 
Natural Language Processing is a field of artificial intelligence. 
It deals with interaction between computers and humans using language. 
Chatbots are widely used in customer service. 
Machine learning helps improve chatbot responses. 
Tokenization is the process of breaking text into words. 
Stopword removal removes common words like is, the, and are. 
Stemming reduces words to their root form. 
Lemmatization converts words to meaningful base forms. 
TF-IDF helps in text representation. 
Cosine similarity measures similarity between two texts."""

# Preprocess text into sentences
sent_tokens = nltk.sent_tokenize(text)


# Preprocessing function
def preprocess(sentence):
    sentence = sentence.lower()
    sentence = ''.join([c for c in sentence if c not in string.punctuation])
    return sentence


# Generate response
def chatbot_response(user_input):
    user_input = preprocess(user_input)

    sent_tokens.append(user_input)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sent_tokens)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = similarity.argsort()[0][-1]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    sent_tokens.pop()

    if score == 0:
        return "Sorry, I couldn't understand that."

    return sent_tokens[idx]