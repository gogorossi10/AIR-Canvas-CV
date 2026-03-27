import nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Step 0: Setup (Run these lines once to download necessary data) ---
# nltk.download('punkt')
# nltk.download('stopwords')

# --- Sample Text (You can replace this with your one-page content) ---
raw_text = """
    This is an Example of Text Preprocessing! It includes punctuations, 
    Running words, and URLs like http://example.com. 
    The goal is to clean this text for NLP models.
"""

print("--- ORIGINAL TEXT ---")
print(raw_text)

# --- Step 1: Script Validation (Lower Casing) ---
# Converting all characters to lowercase to standardize the script
text_lower = raw_text.lower()

# --- Step 2: Filtration (Noise Reduction) ---
# Removing URLs [cite: 8]
text_no_url = re.sub(r'http\S+|www\S+|https\S+', '', text_lower, flags=re.MULTILINE)

# Removing Punctuation and Special Characters [cite: 7, 24]
# This regex removes anything that is not a word character or whitespace
text_filtered = re.sub(r'[^\w\s]', '', text_no_url)

print("\n--- AFTER FILTRATION & LOWER CASING ---")
print(text_filtered)

# --- Step 3: Tokenization ---
# Breaking text down into smaller units (words)
tokens = word_tokenize(text_filtered)

print("\n--- AFTER TOKENIZATION ---")
print(tokens)

# --- Step 4: Stop Word Removal ---
# Removing common words (the, is, and) that convey little meaning
stop_words = set(stopwords.words('english'))
tokens_no_stop = [word for word in tokens if word not in stop_words]

print("\n--- AFTER STOP WORD REMOVAL ---")
print(tokens_no_stop)

# --- Step 5: Stemming ---
# Normalizing words to their root form (e.g., 'running' -> 'run') [cite: 13, 27]
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stop]

print("\n--- AFTER STEMMING ---")
print(stemmed_tokens)