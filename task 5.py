import nltk
from collections import defaultdict, Counter
import math

# Download dataset
nltk.download('udhr')
nltk.download('universal_tagset')
nltk.download('brown')

# Load Brown corpus as training/test data
from nltk.corpus import brown

# Get tagged sentences
data = brown.tagged_sents(tagset='universal')

# Split data into training and testing
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Count transitions and emissions
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

for sentence in train_data:
    previous_tag = "<START>"

    for word, tag in sentence:
        word = word.lower()

        transition_counts[previous_tag][tag] += 1
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1

        previous_tag = tag

    transition_counts[previous_tag]["<END>"] += 1


# Transition probability
def transition_probability(previous_tag, current_tag):
    total = sum(transition_counts[previous_tag].values())
    return (transition_counts[previous_tag][current_tag] + 1) / (total + len(tag_counts) + 1)


# Emission probability
def emission_probability(tag, word):
    total = tag_counts[tag]
    return (emission_counts[tag][word] + 1) / (total + len(emission_counts[tag]) + 1)


# Get all POS tags
tags = list(tag_counts.keys())


# Viterbi Algorithm
def viterbi(sentence):
    words = sentence.lower().split()

    V = [{}]
    path = {}

    # First word
    for tag in tags:
        V[0][tag] = (
            math.log(transition_probability("<START>", tag)) +
            math.log(emission_probability(tag, words[0]))
        )
        path[tag] = [tag]

    # Remaining words
    for i in range(1, len(words)):
        V.append({})
        new_path = {}

        for current_tag in tags:
            best_score = float("-inf")
            best_previous = None

            for previous_tag in tags:
                score = (
                    V[i - 1][previous_tag]
                    + math.log(transition_probability(previous_tag, current_tag))
                    + math.log(emission_probability(current_tag, words[i]))
                )

                if score > best_score:
                    best_score = score
                    best_previous = previous_tag

            V[i][current_tag] = best_score
            new_path[current_tag] = path[best_previous] + [current_tag]

        path = new_path

    # Find best final tag
    best_tag = max(
        tags,
        key=lambda tag: V[-1][tag] +
        math.log(transition_probability(tag, "<END>"))
    )

    return list(zip(words, path[best_tag]))


# User input
sentence = input("Enter a sentence: ")

result = viterbi(sentence)

print("\nPOS Tagged Output:")
for word, tag in result:
    print(word, "→", tag)


# Test dataset accuracy
correct = 0
total = 0

for sentence in test_data:
    words = " ".join(word for word, tag in sentence)

    predicted = viterbi(words)

    for (word, predicted_tag), (_, actual_tag) in zip(predicted, sentence):
        if predicted_tag == actual_tag:
            correct += 1
        total += 1

accuracy = (correct / total) * 100

print("\nEvaluation Report")
print("-----------------")
print("Correct tags :", correct)
print("Total tags   :", total)
print("Accuracy     :", round(accuracy, 2), "%")
