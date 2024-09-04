import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import attention
import CTEN.encoder.feedforward as feedforward

class EncoderModule(nn.Module):
    def __init__(self, model, num_heads, ff, dropout):
        super(EncoderModule, self).__init__()

        # Implementing Attention Mechanism
        self.attention = attention.EncoderAttention(model, num_heads)

        # Implementing Feed-Forward Network
        self.feed_forward_network = feedforward.EncoderFeedForwardNetwork(model, ff)

        # Using Layer Normalization
        self.normalization1 = nn.LayerNorm(model)
        self.normalization2 = nn.LayerNorm(model)

        # Implementing Dropout Normalization
        self.dropout = nn.Dropout(dropout)
