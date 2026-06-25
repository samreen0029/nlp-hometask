import nltk
import string
from collections import Counter
import matplotlib.pyplot as plt

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample News Articles
articles = [
    "Artificial intelligence is transforming healthcare and education systems",
    "AI and machine learning are the future of technology and innovation",
    "Healthcare systems are improving with artificial intelligence solutions",
    "Education technology is growing rapidly with AI based tools"
]

# Step 1: Convert text to lowercase
text = " ".join(articles).lower()
print(text)

# Step 2: Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Step 3: Tokenization
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Step 4: Word Frequency Distribution
word_freq = Counter(filtered_tokens)

# Step 5: Vocabulary (Unique Words)
vocabulary = set(filtered_tokens)

# Step 6: Top 5 Most Frequent Words
top5 = word_freq.most_common(5)

# Step 7: Type-Token Ratio (TTR)
ttr = len(vocabulary) / len(filtered_tokens)

# Step 8: Dominant Topic Analysis
topics = {
    "Artificial Intelligence": ["artificial", "intelligence", "ai"],
    "Healthcare": ["healthcare"],
    "Education": ["education"],
    "Technology": ["technology", "machine", "learning", "innovation", "tools"]
}

topic_scores = {}

for topic, keywords in topics.items():
    score = sum(word_freq[word] for word in keywords if word in word_freq)
    topic_scores[topic] = score

dominant_topic = max(topic_scores, key=topic_scores.get)

# Display Results
print("Tokens:")
print(filtered_tokens)

print("\nWord Frequency Distribution:")
print(word_freq)

print("\nVocabulary:")
print(vocabulary)

print("\nVocabulary Size:", len(vocabulary))

print("\nTop 5 Most Frequent Words:")
for word, freq in top5:
    print(word, ":", freq)

print("\nType-Token Ratio (TTR):", round(ttr, 2))

print("\nTopic Scores:")
for topic, score in topic_scores.items():
    print(topic, ":", score)

print("\nDominant Topic:", dominant_topic)

# Step 9: Bar Chart Visualization
words = list(word_freq.keys())
frequencies = list(word_freq.values())

plt.figure(figsize=(10, 5))
plt.bar(words, frequencies)
plt.title("Word Frequency Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
