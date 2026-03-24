import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len=1000, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded + self.pos_encoding[:, :x.size(1), :]
        transformer_out = self.transformer(embedded)
        pooled = transformer_out.mean(dim=1)
        out = self.dropout(pooled)
        out = self.fc(out)
        return out


def create_transformer_model(vocab_size, num_classes, embed_dim=128, num_heads=4, num_layers=2, max_len=1000, dropout=0.3):
    return TransformerClassifier(vocab_size, embed_dim, num_heads, num_layers, num_classes, max_len, dropout)