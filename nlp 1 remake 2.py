import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- PREREQUISITES ---
# Run these lines once to download the necessary NLTK datasets
# nltk.download('punkt')
# nltk.download('stopwords')

# --- STEP 1: INPUT (ONE FULL PAGE OF TEXT) ---
full_page_text = """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, it focuses on how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Text preprocessing is the very first step of NLP projects. Raw text data often contains noise such as punctuation, special characters, and irrelevant symbols like ! @ # $ %. Preprocessing helps remove these elements, making the text cleaner and easier to analyze. One of the primary steps is tokenization. Tokenization divides text into meaningful units, facilitating subsequent processing steps like feature extraction. Without tokenization, it is impossible to analyze the grammatical structure of the text or calculate word frequencies effectively.

Another critical step is stop word removal. Stopwords are common words like "the," "is," and "and" that often occur frequently but convey little semantic meaning. Removing stopwords can improve the efficiency of text analysis by reducing noise. For instance, in a search engine, searching for "the" would return billions of results, none of which are useful. By filtering these out, the system focuses on the unique keywords that define the document's topic.

Furthermore, normalization techniques such as stemming and lemmatization are applied. Different forms of words (e.g., "run," "running," "ran") can convey the same meaning but appear in different forms. Preprocessing techniques like stemming help standardize these variations. Stemming creates a root form, often by simply chopping off the ends of words. While this sometimes results in non-dictionary words, it is highly effective for reducing the dimensionality of the data. Finally, filtration involves removing URLs like https://www.nlp.com which do not add value.
"""

print("=== 1. ORIGINAL TEXT ===")
print(full_page_text)

# --- STEP 2: SCRIPT VALIDATION (LOWER CASING) ---
# Theory: Converts "Apple" and "apple" to same form
text_lower = full_page_text.lower()

print("\n=== 2. AFTER LOWER CASING ===")
print(text_lower)

# --- STEP 3: FILTRATION (NOISE REDUCTION) ---
# Theory: Removes URLs and punctuation
# Remove URLs
text_no_url = re.sub(r'http\S+|www\S+|https\S+', '', text_lower, flags=re.MULTILINE)
# Remove Punctuation (Keep only words and whitespace)
text_filtered = re.sub(r'[^\w\s]', '', text_no_url)

print("\n=== 3. AFTER FILTRATION (NO URLs/PUNCTUATION) ===")
print(text_filtered)

# --- STEP 4: TOKENIZATION ---
# Theory: Breaks text into smaller units/words
tokens = word_tokenize(text_filtered)

print("\n=== 4. AFTER TOKENIZATION (LIST OF WORDS) ===")
print(tokens)

# --- STEP 5: STOP WORD REMOVAL ---
# Theory: Removes common words (the, is, and) to reduce noise
stop_words = set(stopwords.words('english'))
tokens_no_stop = [word for word in tokens if word not in stop_words]

print("\n=== 5. AFTER STOP WORD REMOVAL ===")
print(tokens_no_stop)

# --- STEP 6: STEMMING ---
# Theory: Reduces words to root form to standardize variations
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in tokens_no_stop]

print("\n=== 6. AFTER STEMMING (FINAL OUTPUT) ===")
print(stemmed_tokens)