import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from additional import activation

class EncoderAttention(nn.Module):
    def __init__(self, model, num_heads):
        super(EncoderAttention, self).__init__()
        # Model Dimension Must be Divisible By Number of Attention Heads
        assert model % num_heads == 0

        # Create Dimensions for Model and Attention Heads
        self.model = model
        self.num_heads = num_heads
        self.d_k = model // num_heads

        # Initial set of Recurrent Layers for Input Transformation
        self.query = nn.Linear(model, model)
        self.key = nn.Linear(model, model)
        self.value = nn.Linear(model, model)
        self.output = nn.Linear(model, model)

    def dot_product_attention(self, Q, K, V, mask=None):
        # Find Attention Values
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Masking to Prevent Attention to Non-Trainable Layers
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax is Used at End of Attention Mechanism
        attention_probabilities = activation.Softmax(x=attention_scores, dim=-1)

        # Normalization of Softmax Values
        output = torch.matmul(attention_probabilities, V)

        return output

    def combine_heads(self, x):
        # Re-Combines Attention to Original Shape
        batch_size, sequence_length, d_k = x.size()
        return x.view(batch_size, sequence_length, d_k, self.num_heads).transpose(1, 2)
    
    def split_heads(self, x):
        # Changing Inputs to num_heads for Multi-Head Attention
        batch_size, _, sequence_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model)