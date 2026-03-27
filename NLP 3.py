import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download required package (Run only once)
nltk.download('punkt')

# 10 Sentences
sentences = [
    "I like natural language processing",
    "Machine learning is very interesting",
    "Python is easy to learn",
    "Artificial intelligence is the future",
    "Data science is a growing field",
    "NLP is a part of artificial intelligence",
    "Deep learning uses neural networks",
    "Text mining extracts useful information",
    "Chatbots use natural language understanding",
    "Speech recognition converts speech to text"
]

for sentence in sentences:
    print("\nSentence:", sentence)
    tokens = word_tokenize(sentence)

    print("Unigrams:")
    print(list(ngrams(tokens, 1)))

    print("Bigrams:")
    print(list(ngrams(tokens, 2)))

    print("Trigrams:")
    print(list(ngrams(tokens, 3)))
