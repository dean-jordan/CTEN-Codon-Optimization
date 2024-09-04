import torch.nn as nn
import torch
import math
from additional import activation

class PositionalEncoding(nn.Module):
    # Creates Class to Input Position Information into Each Token
    def __init__(self, model, max_sequence_length):
        super(PositionalEncoding, self).__init__()

        # Creates Tensors for Positional Encodings
        encoding = torch.zeros(max_sequence_length, model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        divided_term = torch.exp(torch.arange(0, model, 2).float() * -(math.log(10000.0) / model))

        # Uses Sine and Cosine Functions for Position Encoding
        encoding[:, 0::2] = torch.sin(position * divided_term)
        encoding[:, 1::2] = torch.cos(position * divided_term)

        # Creates Register Buffers for Unsqueezing Tensors
        self.register_buffer('encoding', encoding.unsqueeze(0))

    def forward(self, x):
        # Injects Positional Encodings
        return x + self.encoding[:, :x.size(1)]