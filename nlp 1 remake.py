import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- STEP 0: PREREQUISITES ---
# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- STEP 1: LOAD ONE PAGE OF TEXT ---
# This variable simulates a full page of text content for your assignment.
one_page_text = """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, it focuses on how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Text preprocessing is the very first step of NLP projects[cite: 4]. Raw text data often contains noise such as punctuation, special characters, and irrelevant symbols[cite: 24]. Preprocessing helps remove these elements, making the text cleaner and easier to analyze[cite: 25]. One of the primary steps is tokenization. Tokenization divides text into meaningful units, facilitating subsequent processing steps like feature extraction[cite: 30]. Without tokenization, it is impossible to analyze the grammatical structure of the text or calculate word frequencies effectively.

Another critical step is stop word removal. Stopwords are common words like "the," "is," and "and" that often occur frequently but convey little semantic meaning[cite: 31]. Removing stopwords can improve the efficiency of text analysis by reducing noise[cite: 32]. For instance, in a search engine, searching for "the" would return billions of results, none of which are useful. By filtering these out, the system focuses on the unique keywords that define the document's topic.

Furthermore, normalization techniques such as stemming and lemmatization are applied. Different forms of words (e.g., "run," "running," "ran") can convey the same meaning but appear in different forms[cite: 26]. Preprocessing techniques like stemming help standardize these variations[cite: 27]. Stemming creates a root form, often by simply chopping off the ends of words. While this sometimes results in non-dictionary words, it is highly effective for reducing the dimensionality of the data.

Finally, filtration involves removing URLs and other distractions. For example, a scraped web page might contain links like https://example.com which do not add to the sentiment or topic of the text. Cleaning these ensures that the machine learning model does not learn irrelevant patterns.
"""

print(f"--- ORIGINAL TEXT SAMPLE (Length: {len(one_page_text)} characters) ---")
print(one_page_text[:200] + "...\n[Rest of text omitted for brevity]\n")


# --- STEP 2: SCRIPT VALIDATION (LOWER CASING) ---
# Goal: Standardize text so 'Apple' and 'apple' are treated as the same.
processed_text = one_page_text.lower()
print("--- 1. SCRIPT VALIDATION (LOWER CASING) ---")
print(f"Sample: {processed_text[:50]}...\n")


# --- STEP 3: FILTRATION (NOISE REDUCTION) ---
# Goal: Remove URLs, punctuation, and special characters[cite: 7, 8].
# Regex explanation:
# 1. Remove URLs (http/https/www)
processed_text = re.sub(r'http\S+|www\S+|https\S+', '', processed_text, flags=re.MULTILINE)
# 2. Remove Punctuation (anything that isn't a word or whitespace)
processed_text = re.sub(r'[^\w\s]', '', processed_text)

print("--- 2. FILTRATION (NOISE REMOVED) ---")
print(f"Sample: {processed_text[:50]}...\n")


# --- STEP 4: TOKENIZATION ---
# Goal: Break text down into smaller units (words)[cite: 29].
tokens = word_tokenize(processed_text)
print(f"--- 3. TOKENIZATION (Total Tokens: {len(tokens)}) ---")
print(f"First 10 tokens: {tokens[:10]}\n")


# --- STEP 5: STOP WORD REMOVAL ---
# Goal: Remove common words like 'the', 'is' that convey little meaning[cite: 31].
stop_words = set(stopwords.words('english'))
tokens_no_stop = [word for word in tokens if word not in stop_words]

print(f"--- 4. STOP WORD REMOVAL (Count Reduced to: {len(tokens_no_stop)}) ---")
print(f"First 10 tokens: {tokens_no_stop[:10]}\n")


# --- STEP 6: STEMMING ---
# Goal: Reduce words to their root form (e.g., 'running' -> 'run').
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in tokens_no_stop]

print("--- 5. STEMMING (Final Output) ---")
print(f"First 20 Stemmed Tokens: {stemmed_tokens[:20]}")

# Optional: Print comparison of a specific word
print("\n[Example of Stemming Effect]")
original_example = [w for w in tokens_no_stop if "process" in w or "comput" in w][:5]
stemmed_example = [stemmer.stem(w) for w in original_example]
print(f"Original: {original_example}")
print(f"Stemmed:  {stemmed_example}")