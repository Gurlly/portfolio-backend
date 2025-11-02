import torch
import torch.nn as nn
from torch.nn import LSTM
from tokenizers import Tokenizer

# Model version for observability
MODEL_VERSION = "lstm-spam-ham-1.0.0"

# LSTM Model
class LSTMSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        dense_output = self.fc(hidden)
        return dense_output

# Tokenizer and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer.from_file("model/tokenizer.json")

# Hyperparameters
vocab_size = tokenizer.get_vocab_size()
embedding_dim, hidden_dim, output_dim = 100, 256, 1
num_layers, bidirectional, dropout = 3, True, 0.3

model = LSTMSpamClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout)
state = torch.load("model/spam-ham-detection-best-model.pt", map_location=device)

# Loading the model
model.load_state_dict(state)
model.to(device)
model.eval()

# Model Inference
def predict_spam(text: str, max_len: int = 128, threshold: float = 0.7):
    with torch.no_grad():
        ids = tokenizer.encode(text).ids
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        padded = ids[:max_len] + [pad_id] * (max_len - len(ids))
        x = torch.tensor(padded).unsqueeze(0).to(device)
        logits = model(x)                # shape [1, 1]
        prob = torch.sigmoid(logits).item()
        label = "Spam" if prob > threshold else "Ham"
        return prob, label
