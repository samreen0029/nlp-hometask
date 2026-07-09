from collections import defaultdict

class KneserNey5GramLM:
    def __init__(self, n=5, discount=0.75):
        self.n = n
        self.discount = discount

        # Sparse n-gram counts
        self.ngram_counts = [defaultdict(int) for _ in range(n)]
        self.context_counts = [defaultdict(int) for _ in range(n)]

        # Continuation statistics
        self.continuation = defaultdict(set)
        self.preceding = defaultdict(set)

        self.vocab = set()

    # -----------------------------
    # Tokenizer
    # -----------------------------
    def tokenize(self, text):
        return text.lower().split()

    # -----------------------------
    # Train Model
    # -----------------------------
    def train(self, corpus):

        for sentence in corpus:

            words = ["<s>"]*(self.n-1) + self.tokenize(sentence) + ["</s>"]

            self.vocab.update(words)

            for k in range(1, self.n+1):

                for i in range(len(words)-k+1):

                    gram = tuple(words[i:i+k])

                    self.ngram_counts[k-1][gram] += 1

                    if k > 1:

                        context = gram[:-1]

                        self.context_counts[k-1][context] += 1

                        self.continuation[context].add(gram[-1])

                        self.preceding[gram[-1]].add(context)

    # -----------------------------
    # Recursive Kneser-Ney
    # -----------------------------
    def probability(self, ngram):

        if len(ngram) == 1:

            word = ngram[0]

            numerator = len(self.preceding[word])

            denominator = sum(len(v) for v in self.preceding.values())

            if denominator == 0:
                return 1 / max(len(self.vocab),1)

            return numerator / denominator

        context = ngram[:-1]

        word = ngram[-1]

        order = len(ngram)

        full_count = self.ngram_counts[order-1].get(ngram,0)

        context_count = self.context_counts[order-1].get(context,0)

        lower_prob = self.probability(ngram[1:])

        if context_count == 0:
            return lower_prob

        unique_followers = len(self.continuation.get(context,set()))

        lambda_weight = (self.discount * unique_followers) / context_count

        first = max(full_count-self.discount,0)/context_count

        return first + lambda_weight*lower_prob

    # -----------------------------
    # Predict Next Word
    # -----------------------------
    def predict(self, text, top_k=5):

        tokens = self.tokenize(text)

        tokens = ["<s>"]*(self.n-1) + tokens

        context = tuple(tokens[-(self.n-1):])

        scores = []

        for word in self.vocab:

            if word in ("<s>","</s>"):
                continue

            gram = context + (word,)

            prob = self.probability(gram)

            scores.append((word,prob))

        scores.sort(key=lambda x:x[1], reverse=True)

        return scores[:top_k]



# -------------------------------------------------
# Sample Search Query Dataset
# -------------------------------------------------

queries = [

"best places to visit in india",
"best places to visit in chennai",
"best places to visit near me",
"best places to visit during summer",
"best places to visit in kerala",
"best restaurants near me",
"best restaurants in chennai",
"weather today in chennai",
"weather today near me",
"iphone 17 launch event",
"iphone 17 price in india",
"best colleges in tamil nadu",
"places to visit in india",
"places to visit in chennai",
"best tourist places in india"

]



# -------------------------------------------------
# Train Model
# -------------------------------------------------

lm = KneserNey5GramLM()

lm.train(queries)



# -------------------------------------------------
# Prediction
# -------------------------------------------------

query = "best places to visit"

print("Input :", query)

print("\nSuggestions:\n")

for word,score in lm.predict(query):

    print(word,"->",round(score,5))
