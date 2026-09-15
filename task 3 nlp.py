# SMART NEXT-WORD PREDICTOR USING WIKITEXT-2

import re
import urllib.request
from collections import Counter

# 1. Load WikiText-2 corpus
url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"

print("Loading WikiText-2 corpus...")
text = urllib.request.urlopen(url).read().decode("utf-8")

# 2. Clean and tokenize the text
text = text.lower()
text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
words = text.split()

print("Total words:", len(words))

# 3. Build unigram, bigram and trigram frequency tables

unigram = Counter(words)

bigram = Counter()
trigram = Counter()

for i in range(len(words) - 1):
    bigram[(words[i], words[i + 1])] += 1

for i in range(len(words) - 2):
    trigram[(words[i], words[i + 1], words[i + 2])] += 1

# 4. Calculate word probabilities
total_words = sum(unigram.values())

unigram_probability = {
    word: count / total_words
    for word, count in unigram.items()
}

# Function to predict next words
def predict_next(sentence, top_n=5):

    # Clean and tokenize user input
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    input_words = sentence.split()

    if len(input_words) == 0:
        return []

    candidates = {}

    # 5 & 6. Find candidate words using previous words

    # Try trigram prediction first
    if len(input_words) >= 2:
        w1 = input_words[-2]
        w2 = input_words[-1]

        for (a, b, c), count in trigram.items():
            if a == w1 and b == w2:
                probability = count / bigram[(w1, w2)]
                candidates[c] = probability

    # If no trigram candidates, use bigram
    if not candidates:
        w1 = input_words[-1]

        for (a, b), count in bigram.items():
            if a == w1:
                probability = count / unigram[w1]
                candidates[b] = probability

    # If still no candidates, use unigram
    if not candidates:
        for word, probability in unigram_probability.items():
            candidates[word] = probability

    # 7. Rank candidates based on probability
    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 8. Display Top-N predictions
    return ranked[:top_n]


# 9. Test predictor with user sentences

print("\n----------------------------------------")
print("       SMART NEXT-WORD PREDICTOR")
print("----------------------------------------")
print("Enter a sentence or partial sentence.")
print("Type 'exit' to stop.\n")

while True:

    sentence = input("Enter sentence: ")

    if sentence.lower() == "exit":
        print("Program ended.")
        break

    predictions = predict_next(sentence, 5)

    print("\nTop predicted next words:")

    if predictions:
        for i, (word, probability) in enumerate(predictions, 1):
            print(f"{i}. {word} - {probability:.3f}")
    else:
        print("No prediction found.")

    print()
