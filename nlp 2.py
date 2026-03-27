import nltk
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    ps = PorterStemmer()
except Exception:
    import nltk

    nltk.download('punkt')
    ps = PorterStemmer()


class MorphologicalLab:
    def __init__(self):
        self.ps = PorterStemmer()
        # Suffix lists for Rule-Based Analysis
        self.inflectional = ['s', 'es', 'ed', 'ing', 'er', 'est']
        self.derivational = ['ly', 'ment', 'ness', 'tion', 'able', 'ful', 'ize']

    def task_1_2_6(self, word):
        """Rule-based analyzer: Separates stem+suffix and checks Inflected vs Derived"""
        found_suffix = ""
        category = "Base Form"

        # Check Derivational (Task 6: Base form detection)
        for s in self.derivational:
            if word.endswith(s):
                return word[:-len(s)], s, "Derived"

        # Check Inflectional (Task 6: Removing inflectional suffixes)
        for s in self.inflectional:
            if word.endswith(s):
                return word[:-len(s)], s, "Inflected"

        return word, "", "Base Form"

    def task_3_fsa(self, noun):
        """Task 3 & 8: FSA for Plural Noun Formation (cat, box, baby)"""
        if noun.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return noun + "es"
        elif noun.endswith('y') and noun[-2] not in 'aeiou':
            return noun[:-1] + "ies"
        else:
            return noun + "s"

    def task_4_verbs(self, verb):
        """Task 4 & 9-12: Verb Inflections (Present, Past, Participle)"""
        # Handling basic ending 'e' for regular verbs
        if verb.endswith('e'):
            return {
                "Present": verb + "s",
                "Past": verb + "d",
                "Present Participle": verb[:-1] + "ing"
            }
        return {
            "Present": verb + "s",
            "Past": verb + "ed",
            "Present Participle": verb + "ing"
        }

    def task_5_porter(self, word_list):
        """Task 14: Porter Stemmer implementation"""
        return {word: self.ps.stem(word) for word in word_list}


# --- MAIN EXECUTION ---
lab = MorphologicalLab()

print("=" * 60)
print("NATURAL LANGUAGE PROCESSING: PRACTICAL - 02 OUTPUT")
print("=" * 60)

# 1. Rule-based Analysis & Base Form (Tasks 5, 6, 15, 16, 17)
print("\n[TASKS 1, 2, 6]: RULE-BASED ANALYSIS & INFLECTION VS DERIVATION")
test_words = ["enjoyment", "happily", "playing", "faster", "books"]
print(f"{'Word':<15} | {'Stem':<10} | {'Suffix':<8} | {'Type':<12}")
print("-" * 55)
for w in test_words:
    stem, suffix, cat = lab.task_1_2_6(w)
    print(f"{w:<15} | {stem:<10} | {suffix:<8} | {cat:<12}")

# 2. FSA for Plurals (Tasks 7, 8, 18)
print("\n[TASK 3]: FSA FOR PLURAL NOUN FORMATION")
plurals = ["cat", "box", "baby"]
for p in plurals:
    print(f"Noun: {p:6} ---> Plural: {lab.task_3_fsa(p)}")

# 3. Verb Inflections (Tasks 9, 10, 11, 12, 18)
print("\n[TASK 4]: VERB INFLECTION ANALYSIS")
verbs = ["walk", "dance"]
for v in verbs:
    forms = lab.task_4_verbs(v)
    print(f"Verb: {v}")
    for tense, form in forms.items():
        print(f"  - {tense:18}: {form}")

# 4. Porter Stemmer (Task 14)
print("\n[TASK 5]: PORTER STEMMER OUTPUT")
stem_list = ["waiting", "waited", "waits", "stemming", "stems"]
stem_results = lab.task_5_porter(stem_list)
for original, stemmed in stem_results.items():
    print(f"Original: {original:10} ---> Stemmed: {stemmed}")

print("\n" + "=" * 60)