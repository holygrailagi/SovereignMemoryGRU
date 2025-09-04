import torch
import torch.nn as nn
import xxhash
import math

class HashGateGRU(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.W_ir = nn.Linear(inp_dim, hid_dim, bias=False)
        self.W_hr = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_ih = nn.Linear(inp_dim, hid_dim, bias=False)
        self.W_hh = nn.Linear(hid_dim, hid_dim, bias=True)

    def hash_gate(self, h):
        h_bytes = h.detach().cpu().numpy().tobytes()
        hash_val = xxhash.xxh64(h_bytes).intdigest() / 2**64
        return torch.sigmoid(torch.tensor(hash_val, device=h.device))

    def forward(self, x, h):
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = self.hash_gate(h)  # The key trick
        h_tilde = torch.tanh(self.W_ih(x) + r * self.W_hh(h))
        h_out = (1 - z) * h + z * h_tilde
        return h_out

# Example usage: A simple training loop for demonstration
# Assuming you have some input data and a loss function

# Hyperparameters
input_dim = 128  # Example input dimension (e.g., embedding size)
hidden_dim = 8192
batch_size = 1
seq_len = 2048
learning_rate = 1e-4

# Initialize model
model = HashGateGRU(input_dim, hidden_dim)

# Optimizer (only optimize the GRU parameters)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Dummy data for example (replace with real text data tokenized and embedded)
# Here, assuming inputs are random for illustration
inputs = torch.randn(seq_len, batch_size, input_dim)  # Sequence of inputs
targets = torch.randn(seq_len, batch_size, hidden_dim)  # Dummy targets (e.g., next hidden states or predictions)

# Training loop example
num_steps = 20000  # Train for 20k steps as mentioned
h = torch.zeros(batch_size, hidden_dim)  # Initial hidden state

for step in range(num_steps):
    optimizer.zero_grad()
    
    # Forward pass through the sequence
    outputs = []
    for t in range(seq_len):
        h = model(inputs[t], h)
        outputs.append(h)
    
    outputs = torch.stack(outputs)
    
    # Example loss: Mean squared error for simplicity (replace with cross-entropy on actual task)
    loss = nn.MSELoss()(outputs, targets)
    
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# Note: In a real setup, use a proper dataset, tokenizer, embeddings, and cross-entropy loss for language modeling.
# The self-improvement happens as the hidden states learn to compress history via the hash gate.
