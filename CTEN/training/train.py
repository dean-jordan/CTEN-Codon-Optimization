import torch.optim as optim
from additional import loss
import main
import data_pipeline

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

# Loss Function and Optimizer Setup
criterion = loss.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(codon_optimization_transformer.parameters(), lr, betas, eps)

codon_optimization_transformer.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    output = codon_optimization_transformer(data_pipeline.source_data, data_pipeline.target_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, target_vocab_size), data_pipeline.target_data[:, 1:].contiguous.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")