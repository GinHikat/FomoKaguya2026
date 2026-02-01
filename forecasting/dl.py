import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

class LSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden=128, mlp_hidden1=64, mlp_hidden2=32,
        output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden,
        batch_first=True, num_layers=2, dropout=dropout)
        self.mlp = nn.Sequential(
        nn.Linear(lstm_hidden, mlp_hidden1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden1, mlp_hidden2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden2, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        last_out = lstm_out[:, -1, :] 
        return self.mlp(last_out)

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head additive (Bahdanau-style) temporal self-attention
    for LSTM outputs.

    Input:
        lstm_outputs: (B, T, H)

    Output:
        context_vector: (B, H)
        attention_weights: (B, num_heads, T)
    """

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.attn_weight = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim, bias=True)
            for _ in range(num_heads)
        ])

        self.attn_score = nn.ModuleList([
            nn.Linear(self.head_dim, 1, bias=False)
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lstm_outputs):
        
        head_contexts = []
        head_attn_weights = []

        for i in range(self.num_heads):
            energy = torch.tanh(self.attn_weight[i](lstm_outputs))

            scores = self.attn_score[i](energy).squeeze(-1)

            attn_weights = F.softmax(scores, dim=1)

            head_attn_weights.append(attn_weights)

            attn_weights_expanded = attn_weights.unsqueeze(-1)

            weighted = lstm_outputs * attn_weights_expanded

            context = weighted.sum(dim=1)[:, :self.head_dim]

            head_contexts.append(context)

        # Concatenate all heads
        context_vector = torch.cat(head_contexts, dim=-1)

        context_vector = self.out_proj(context_vector)

        attention_weights = torch.stack(head_attn_weights, dim=1)

        return context_vector, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class Transformer(nn.Module):
    def __init__(
        self,
        input_size=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, T, input_size)
        """
        x = self.input_proj(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)  # (B, T, d_model)

        x = self.norm(x)

        context = x.mean(dim=1)  # (B, d_model)

        output = self.fc(context)  # (B, 1)
        return output

class LSTMSelfAttention(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads=4)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, input_dim)

        lstm_out, _ = self.lstm(x)          # (B, T, H)

        context, attn_weights = self.attention(lstm_out)
        # context: (B, H)

        output = self.fc(context)            # (B, 1)
        return output

class DL():
    def __init__(self, model_name, input_dim):
        self.model_name = model_name
        self.input_dim = input_dim
    
    def load_statedict(self):
        if self.model_name == 'lstm':
            model = LSTM(input_size=self.input_dim, output_size=1, dropout=0.5).to(device)
        if self.model_name == 'bilstm_attention':
            model = LSTMSelfAttention(input_size = self.input_dim).to(device)

        state_dict = torch.load(f'artifact/{self.model_name}.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        return model

    def predict(self, X):

        model = self.load_statedict()
        
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred_tensor = model(X_test_tensor).cpu()

        y_pred = y_pred_tensor.numpy().ravel()

        return y_pred
