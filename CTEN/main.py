import torch
import torch.nn as nn
from additional import positional_encoding
from encoder import encoder
from decoder import decoder

# Defining Full Transformer Network
class TransformerNetwork(nn.Module):

    # Defining Full Transformer Network from Encoder and Decoder
    def __init__(self, source_vocab_size, target_vocab_size, model, num_heads, num_layers, ff, max_sequence_length, dropout):
        super(TransformerNetwork, self).__init__()

        # Input Embeddings Using nn.Embedding Layer
        self.encoder_embeddings = nn.Embedding(source_vocab_size, model)
        self.decoder_embeddings = nn.Embedding(target_vocab_size, model)
        self.positional_encoding = positional_encoding.PositionalEncoding(model, max_sequence_length)

        # Create Ensemble of Layers for Each Transformer Module
        self.encoder_layers = nn.ModuleList([encoder.EncoderModule(model, num_heads, ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([decoder.DecoderModule(model, num_heads, ff, dropout) for _ in range(num_layers)])

        # Create Recurrent Layers for Full Model Passthrough
        self.recurrent = nn.Linear(model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # Creates Masks for Attention
    def generate_mask(self, source, target):
        # Creating Masked Multi-Head Attention Mechanism
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (source != 0).unsqueeze(1).unsqueeze(3)

        # Defining Sequence Lengths for Masks and Creating Masks
        sequence_length = target.size(1)
        no_peak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length), diagonal=1)).bool()
        target_mask = target_mask & no_peak_mask

        # Creates Source and Return Mask
        return source_mask, target_mask
    
    # Defines Data Passthrough Within Network
    def forward(self, source, target):

        # Embedding the Source and Target Masks
        source_mask, target_mask = self.generate_mask(source, target)
        source_embedded = self.dropout(positional_encoding.PositionalEncoding(self.encoder_embeddings(source)))
        target_embedded = self.dropout(positional_encoding.PositionalEncoding(self.decoder_embeddings(target)))

        # Applying Embeddings for Source
        encoder_output = source_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        # Applying Embeddings for Target
        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)
        
        # Transforming Final Output of Transformer
        output = self.recurrent(decoder_output)
        return output