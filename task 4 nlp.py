# ASSIGNMENT SIMILARITY / PLAGIARISM DETECTOR
# Using TF-IDF and Cosine Similarity

import re
import math
from collections import Counter

# ------------------------------------------
# 1. Load assignment documents
# ------------------------------------------

documents = {}

n = int(input("Enter number of assignments: "))

for i in range(n):
    name = input(f"\nEnter name of Assignment {i+1}: ")
    text = input(f"Enter text for {name}: ")
    documents[name] = text


# ------------------------------------------
# 2 & 3. Clean, normalize and tokenize
# ------------------------------------------

def clean_and_tokenize(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize
    words = text.split()

    return words


tokenized_documents = {}

for name, text in documents.items():
    tokenized_documents[name] = clean_and_tokenize(text)


# ------------------------------------------
# 4. Calculate TF-IDF
# ------------------------------------------

# Create vocabulary
vocabulary = set()

for words in tokenized_documents.values():
    vocabulary.update(words)

vocabulary = list(vocabulary)


# Term Frequency (TF)
def calculate_tf(words):

    count = Counter(words)
    total = len(words)

    tf = {}

    for word in vocabulary:
        tf[word] = count[word] / total if total > 0 else 0

    return tf


# Document Frequency (DF)
df = {}

for word in vocabulary:

    df[word] = 0

    for words in tokenized_documents.values():

        if word in words:
            df[word] += 1


# Inverse Document Frequency (IDF)
idf = {}

total_documents = len(tokenized_documents)

for word in vocabulary:

    idf[word] = math.log(
        total_documents / (1 + df[word])
    ) + 1


# TF-IDF vectors
tfidf_vectors = {}

for name, words in tokenized_documents.items():

    tf = calculate_tf(words)

    vector = {}

    for word in vocabulary:
        vector[word] = tf[word] * idf[word]

    tfidf_vectors[name] = vector


# ------------------------------------------
# 5. Calculate Cosine Similarity
# ------------------------------------------

def cosine_similarity(vector1, vector2):

    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0

    for word in vocabulary:

        dot_product += vector1[word] * vector2[word]

        magnitude1 += vector1[word] ** 2

        magnitude2 += vector2[word] ** 2

    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# ------------------------------------------
# 6 & 7. Compare similarity with threshold
# ------------------------------------------

threshold = 0.30

similarity_results = []

names = list(documents.keys())

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        name1 = names[i]
        name2 = names[j]

        similarity = cosine_similarity(
            tfidf_vectors[name1],
            tfidf_vectors[name2]
        )

        percentage = similarity * 100

        if similarity >= threshold:
            status = "Possible Plagiarism"
        else:
            status = "Low Similarity"

        similarity_results.append(
            (name1, name2, percentage, status)
        )


# ------------------------------------------
# 8. Generate Similarity Report
# ------------------------------------------

print("\n")
print("=" * 60)
print("        ASSIGNMENT SIMILARITY REPORT")
print("=" * 60)

for name1, name2, percentage, status in similarity_results:

    print(f"\n{name1} vs {name2}")
    print(f"Similarity: {percentage:.2f}%")
    print(f"Status: {status}")


# ------------------------------------------
# Similarity Matrix
# ------------------------------------------

print("\n")
print("=" * 60)
print("             SIMILARITY MATRIX")
print("=" * 60)

print("\n" + " " * 15, end="")

for name in names:
    print(f"{name[:10]:>12}", end="")

print()

for name1 in names:

    print(f"{name1[:10]:<15}", end="")

    for name2 in names:

        if name1 == name2:
            similarity = 100

        else:
            similarity = cosine_similarity(
                tfidf_vectors[name1],
                tfidf_vectors[name2]
            ) * 100

        print(f"{similarity:>11.2f}%", end="")

    print()


# ------------------------------------------
# 9. Display ranked suspicious pairs
# ------------------------------------------

print("\n")
print("=" * 60)
print("       RANKED SUSPICIOUS DOCUMENT PAIRS")
print("=" * 60)

# Sort from highest similarity to lowest
similarity_results.sort(
    key=lambda x: x[2],
    reverse=True
)

for rank, result in enumerate(similarity_results, 1):

    name1, name2, percentage, status = result

    print(
        f"{rank}. {name1} vs {name2} "
        f"-> {percentage:.2f}% -> {status}"
    )

print("\nThreshold used:", threshold * 100, "%")
