import torch.nn as nn
import attention_d
import ff_d

class DecoderModule(nn.Module):
    # Initializing the Decoder Module
    def __init__(self, model, num_heads, ff, dropout):
        super(DecoderModule, self).__init__()
        
        # Utilizing Decoder Attention Mechanism with Multi-Head and Cross Attention
        self.attention1 = attention_d.DecoderAttention(model, num_heads)
        self.attention2 = attention_d.DecoderAttention(model, num_heads)

        # Implementing Feed-Forward Network
        self.feed_forward = ff_d.DecoderFeedForwardNetwork(model, ff)

        # Implementing Layer Normalization-Based Regularization
        self.normalization1 = nn.LayerNorm(model)
        self.normalization2 = nn.LayerNorm(model)
        self.normalization3 = nn.LayerNorm(model)

        # Implementing Dropout-Based Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # Multi-Head Attention Output
        attention_output = self.attention1(x, x, x, target_mask)
        x = self.normalization1(x + self.dropout(attention_output))

        # Cross-Attention Output
        attention_output = self.attention2(x, encoder_output, encoder_output, source_mask)
        x = self.normalization2(x + self.dropout(attention_output))

        # Feed-Forward Network Output
        feed_forward_output = self.feed_forward(x)
        x = self.normalization3(x + self.dropout(feed_forward_output))

        # Outputting Regularized Decoder Values
        return x