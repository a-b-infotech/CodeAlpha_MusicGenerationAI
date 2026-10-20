import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    """
    Recurrent Neural Network (LSTM) for Polyphonic Music Generation.
    Takes note sequence token indices as input and predicts the probability distribution
    over the next musical pitch/chord in the vocabulary.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super(MusicLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits, hidden
