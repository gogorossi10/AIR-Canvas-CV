import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

nltk.download('movie_reviews')

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle data
random.shuffle(documents)

# Feature extraction
def document_features(document):
    document_words = set(document)
    features = {}
    for word in list(document_words)[:2000]:
        features[word] = True
    return features

# Prepare features
featuresets = [(document_features(d), c) for (d, c) in documents]

# Train-test split
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Train model
classifier = NaiveBayesClassifier.train(train_set)

# Accuracy
print("Naive Bayes Accuracy:", accuracy(classifier, test_set))

# Test examples
while True:
    text = input("Enter sentence: ")
    words = text.split()
    feats = document_features(words)
    print("Prediction:", classifier.classify(feats))