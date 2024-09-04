import torch.nn as nn
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

    def forward(self, x, mask):
        # Finalizing Attention Output and Applying Normalization
        attention_output = self.attention(x, x, x, mask)
        x = self.normalization1(x + self.dropout(attention_output))

        # Finalizing Feed-Forward Network Output and Applying Normalization
        ff_output = self.feed_forward_network(x)
        x = self.normalization2(x + self.dropout(ff_output))
