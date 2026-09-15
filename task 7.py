import pandas as pd
import re
import torch
import torch.nn as nn
import random

# -----------------------------------
# 1. Load English-Tamil Dataset
# -----------------------------------
data = pd.read_csv("english_tamil.csv")

# Example columns:
# english,tamil
# How are you?,நீங்கள் எப்படி இருக்கிறீர்கள்?


# -----------------------------------
# 2. Clean and Tokenize
# -----------------------------------
def clean_english(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z? ]", "", text)
    return text.strip().split()


def clean_tamil(text):
    return text.strip().split()


data["english_tokens"] = data["english"].apply(clean_english)
data["tamil_tokens"] = data["tamil"].apply(clean_tamil)


# -----------------------------------
# 3. Build Vocabularies
# -----------------------------------
eng_vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3
}

tam_vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3
}

for sentence in data["english_tokens"]:
    for word in sentence:
        if word not in eng_vocab:
            eng_vocab[word] = len(eng_vocab)

for sentence in data["tamil_tokens"]:
    for word in sentence:
        if word not in tam_vocab:
            tam_vocab[word] = len(tam_vocab)


eng_reverse = {v: k for k, v in eng_vocab.items()}
tam_reverse = {v: k for k, v in tam_vocab.items()}


# -----------------------------------
# 4. Convert Sentences to Numbers
# -----------------------------------
def encode_english(tokens):
    return [eng_vocab["<START>"]] + \
           [eng_vocab.get(w, eng_vocab["<UNK>"]) for w in tokens] + \
           [eng_vocab["<END>"]]


def encode_tamil(tokens):
    return [tam_vocab["<START>"]] + \
           [tam_vocab.get(w, tam_vocab["<UNK>"]) for w in tokens] + \
           [tam_vocab["<END>"]]


data["english_encoded"] = data["english_tokens"].apply(
    encode_english
)

data["tamil_encoded"] = data["tamil_tokens"].apply(
    encode_tamil
)


# -----------------------------------
# 5. Encoder
# -----------------------------------
class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            input_size,
            embedding_size,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, (hidden, cell) = self.lstm(
            embedded
        )

        return outputs, hidden, cell


# -----------------------------------
# 6. Attention
# -----------------------------------
class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.score = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, hidden, encoder_outputs):

        # hidden:
        # [1, batch, hidden]

        hidden = hidden[-1].unsqueeze(1)

        # Repeat hidden for every source word
        hidden = hidden.repeat(
            1,
            encoder_outputs.size(1),
            1
        )

        energy = torch.tanh(
            self.attention(
                torch.cat(
                    (hidden, encoder_outputs),
                    dim=2
                )
            )
        )

        scores = self.score(energy).squeeze(2)

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        )

        return context, attention_weights


# -----------------------------------
# 7. Decoder with Attention
# -----------------------------------
class Decoder(nn.Module):

    def __init__(
        self,
        output_size,
        embedding_size,
        hidden_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            output_size,
            embedding_size,
            padding_idx=0
        )

        self.attention = Attention(hidden_size)

        self.lstm = nn.LSTM(
            embedding_size + hidden_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs
    ):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)

        context, attention_weights = self.attention(
            hidden,
            encoder_outputs
        )

        lstm_input = torch.cat(
            (embedded, context),
            dim=2
        )

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(
            torch.cat(
                (output, context),
                dim=1
            )
        )

        return prediction, hidden, cell, attention_weights


# -----------------------------------
# 8. Seq2Seq Model
# -----------------------------------
class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        encoder_outputs, hidden, cell = self.encoder(
            source
        )

        outputs = []

        input_token = target[:, 0]

        for t in range(1, target.size(1)):

            prediction, hidden, cell, attention = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

            outputs.append(prediction.unsqueeze(1))

            best_word = prediction.argmax(1)

            if random.random() < teacher_forcing_ratio:
                input_token = target[:, t]
            else:
                input_token = best_word

        return torch.cat(outputs, dim=1)


# -----------------------------------
# 9. Prepare Training Data
# -----------------------------------
max_len = 20

def pad_sequence(sequence):

    sequence = sequence[:max_len]

    while len(sequence) < max_len:
        sequence.append(0)

    return sequence


X = [
    pad_sequence(x)
    for x in data["english_encoded"]
]

Y = [
    pad_sequence(y)
    for y in data["tamil_encoded"]
]

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)


# -----------------------------------
# Train-Test Split
# -----------------------------------
split = int(0.8 * len(X))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]


# -----------------------------------
# Create Model
# -----------------------------------
encoder = Encoder(
    len(eng_vocab),
    128,
    256
)

decoder = Decoder(
    len(tam_vocab),
    128,
    256
)

model = Seq2Seq(
    encoder,
    decoder
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# -----------------------------------
# 10. Train Model
# -----------------------------------
EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    output = model(
        X_train,
        Y_train
    )

    output = output.reshape(
        -1,
        output.size(-1)
    )

    target = Y_train[:, 1:].reshape(-1)

    loss = criterion(
        output,
        target
    )

    loss.backward()

    optimizer.step()

    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        round(loss.item(), 4)
    )


# -----------------------------------
# 11. Translate New Sentence
# -----------------------------------
def translate(sentence):

    model.eval()

    tokens = clean_english(sentence)

    encoded = encode_english(tokens)

    encoded = pad_sequence(encoded)

    source = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    with torch.no_grad():

        encoder_outputs, hidden, cell = model.encoder(
            source
        )

    input_token = torch.tensor(
        [tam_vocab["<START>"]]
    )

    translated_words = []
    attention_values = []

    for i in range(max_len):

        with torch.no_grad():

            prediction, hidden, cell, attention = model.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

        best_word = prediction.argmax(1).item()

        if best_word == tam_vocab["<END>"]:
            break

        if best_word != tam_vocab["<PAD>"]:
            translated_words.append(
                tam_reverse.get(
                    best_word,
                    "<UNK>"
                )
            )

        attention_values.append(
            attention.squeeze().numpy()
        )

        input_token = torch.tensor(
            [best_word]
        )

    print("\nEnglish:", sentence)

    print(
        "Tamil:",
        " ".join(translated_words)
    )

    # -----------------------------------
    # 12. Display Attention
    # -----------------------------------
    print("\nAttention over source words:")

    source_words = tokens

    for i, word in enumerate(source_words):

        if i < len(attention_values):

            value = attention_values[i][i]

            print(
                word,
                "→",
                round(float(value), 3)
            )


# -----------------------------------
# User Input
# -----------------------------------
sentence = input(
    "\nEnter an English sentence: "
)

translate(sentence)
