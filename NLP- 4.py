import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger
from nltk import CFG
from nltk.parse import ChartParser

# -----------------------------
# Download Required Resources
# -----------------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('treebank')

# ==========================================
# PART (a): POS TAGGING
# ==========================================

print("\n========== PART (a): POS TAGGING ==========\n")

sentences = [
    "Can you book transfer now",
    "Transfer money today",
    "Book A transfer"
]

# ---- Rule-Based POS Tagging ----
print("Rule-Based POS Tagging:\n")
for sent in sentences:
    tokens = word_tokenize(sent)
    print(nltk.pos_tag(tokens))

# ---- Stochastic POS Tagging (Improved) ----
print("\nStochastic POS Tagging (HMM approach):\n")

train_data = treebank.tagged_sents()

default_tagger = DefaultTagger('NN')
unigram = UnigramTagger(train_data, backoff=default_tagger)
bigram = BigramTagger(train_data, backoff=unigram)

for sent in sentences:
    tokens = word_tokenize(sent)
    print(bigram.tag(tokens))


# ==========================================
# PART (b): PARSING TREE
# ==========================================

print("\n========== PART (b): PARSING TREE ==========\n")

grammar = CFG.fromstring("""
S -> NP VP
S -> PP S
S -> S CONJ S

NP -> DT NN
NP -> DT JJ NN
NP -> NN
NP -> PRP

VP -> V
VP -> V NP
VP -> V NP PP
VP -> V PP
VP -> VP ADV

PP -> IN NP

DT -> 'the' | 'an' | 'a'
JJ -> 'curious' | 'old' | 'wooden' | 'heavy' | 'international'
NN -> 'child' | 'box' | 'river' | 'bank' | 'scientist' | 'vaccine' | 'award' | 'rain' | 'children' | 'garden' | 'clock' | 'wind'
PRP -> 'they' | 'we' | 'you'
V -> 'opened' | 'received' | 'played' | 'wind' | 'chase' | 'stopped'
IN -> 'near' | 'after' | 'in' | 'while'
ADV -> 'happily' | 'back'
CONJ -> 'while'
""")

parser = ChartParser(grammar)

parse_sentences = [
    "They wind back the clock",
    "The curious child opened the old box",
    "The scientist received an international award",
    "After the heavy rain stopped the children played happily in the garden"
]

for sentence in parse_sentences:
    print("\nSentence:", sentence)
    tokens = word_tokenize(sentence.lower())   # 🔥 fixed here
    trees = list(parser.parse(tokens))

    if trees:
        for tree in trees:
            print(tree)
            tree.pretty_print()
    else:
        print("No parse tree generated.")