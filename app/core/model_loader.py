import torch
import torch.nn as nn
from tokenizers import Tokenizer
from pathlib import Path
from app.core.config import settings

# LSTM Definition
class LSTMSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        dense_output = self.fc(hidden)
        return dense_output

def load_model_and_tokenizer():
    """
    Loads the trained LSTM model and tokenizer from specified paths.
    Returns:
        model (LSTMSpamClassifier): The loaded model.
        tokenizer (tokenizers.Tokenizer): The loaded tokenizer.
        device (torch.device): The device the model is on.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model and tokenizer... Device: {device}")

    # Load Tokenizer
    tokenizer_path = Path(settings.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Tokenizer loaded successfully.")

    # Define model hyperparameters (must match training)
    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    num_layers = 3
    bidirectional = True
    dropout = 0.3

    # Initialize Model
    model = LSTMSpamClassifier(
        vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout
    )
    model.to(device)

    # Load Model State Dict
    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Ensure map_location matches the target device
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")

    return model, tokenizer, device