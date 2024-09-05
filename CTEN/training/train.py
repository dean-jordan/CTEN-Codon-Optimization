import torch.optim as optim
from additional import loss
import main

# Vocabulary Sizes for Source and Target Sequences
source_vocab_size = 5000
target_vocab_size = 5000

# Dimensionality of Model Embeddings
model = 512

# Number of Attention Heads in Multi-Head Attention
num_heads = 8

# Layers in Both Encoder and Decoder
num_layers = 6

# Dimensionality of Feed-Forward Network
ff = 2048

# Maximum Sequence Length for Positional Encoding
max_sequence_length = 100

# Dropout Regularization Rate
dropout = 0.1

codon_optimization_transformer = main.TransformerNetwork(source_vocab_size, target_vocab_size, model, num_heads, num_layers, ff, max_sequence_length, dropout)

# Hyperparameters
lr = 0.0001
betas = (0.9, 0.98)
eps = 1e-9
epochs = 100

criterion = loss.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(codon_optimization_transformer.parameters(), lr, betas, eps)

for epoch in range(epochs):
    print("Finishing Later")