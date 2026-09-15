import pandas as pd
import re
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# 1. Load IMDb 50K Dataset
# -----------------------------
data = pd.read_csv("IMDB Dataset.csv")

# Convert sentiment into numbers
data["sentiment"] = data["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# -----------------------------
# 2. Clean and Tokenize
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub("<.*?>", "", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    return text.split()

data["tokens"] = data["review"].apply(clean_text)

# -----------------------------
# 3. Create Vocabulary
# -----------------------------
counter = Counter()

for tokens in data["tokens"]:
    counter.update(tokens)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.items():
    if count >= 2:
        vocab[word] = len(vocab)

print("Vocabulary size:", len(vocab))


# -----------------------------
# 4. Convert Words to Numbers
# -----------------------------
def encode(tokens):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokens]

data["encoded"] = data["tokens"].apply(encode)


# -----------------------------
# 5. Pad Sequences
# -----------------------------
MAX_LEN = 200

def pad_sequence(sequence):
    if len(sequence) < MAX_LEN:
        sequence = sequence + [0] * (MAX_LEN - len(sequence))
    else:
        sequence = sequence[:MAX_LEN]

    return sequence

data["padded"] = data["encoded"].apply(pad_sequence)


# -----------------------------
# 6. Train-Test Split
# -----------------------------
X = torch.tensor(data["padded"].tolist(), dtype=torch.long)
y = torch.tensor(data["sentiment"].values, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Further split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# -----------------------------
# 7. LSTM Model
# -----------------------------
class LSTMModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            128,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            128,
            64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc(x)

        return x.squeeze()


# Create model
model = LSTMModel(len(vocab))

# -----------------------------
# 8. Loss and Optimizer
# -----------------------------
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 9. Training
# -----------------------------
EPOCHS = 5
BATCH_SIZE = 64

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for i in range(0, len(X_train), BATCH_SIZE):

        batch_x = X_train[i:i+BATCH_SIZE]
        batch_y = y_train[i:i+BATCH_SIZE]

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / (len(X_train) // BATCH_SIZE + 1)

    # Validation
    model.eval()

    with torch.no_grad():

        val_output = model(X_val)

        val_loss = criterion(
            val_output,
            y_val
        ).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        "Epoch",
        epoch + 1,
        "Training Loss:",
        round(train_loss, 4),
        "Validation Loss:",
        round(val_loss, 4)
    )


# -----------------------------
# 10. Test Model
# -----------------------------
model.eval()

with torch.no_grad():

    test_output = model(X_test)

    probabilities = torch.sigmoid(test_output)

    predictions = (
        probabilities >= 0.5
    ).int().numpy()

actual = y_test.numpy()


# -----------------------------
# 11. Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(
    actual,
    predictions
)

precision = precision_score(
    actual,
    predictions
)

recall = recall_score(
    actual,
    predictions
)

f1 = f1_score(
    actual,
    predictions
)

print("\nEvaluation Report")
print("-------------------------")
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1-Score  :", round(f1 * 100, 2), "%")


# -----------------------------
# 12. Predict New Review
# -----------------------------
def predict_sentiment(review):

    tokens = clean_text(review)

    encoded = encode(tokens)

    padded = pad_sequence(encoded)

    input_tensor = torch.tensor(
        [padded],
        dtype=torch.long
    )

    model.eval()

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output).item()

    if probability >= 0.5:

        sentiment = "Positive"
        confidence = probability * 100

    else:

        sentiment = "Negative"
        confidence = (1 - probability) * 100

    print("\nReview:", review)
    print("Predicted Sentiment:", sentiment)
    print("Confidence:", round(confidence, 2), "%")


# -----------------------------
# 13. User Input
# -----------------------------
review = input("\nEnter a movie review: ")

predict_sentiment(review)
