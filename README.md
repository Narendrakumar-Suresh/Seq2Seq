# Seq2Seq Model: Basic Example

This repository contains a simple implementation of a Sequence-to-Sequence (Seq2Seq) model using PyTorch, found in `main.py`. The code is designed to help you get a basic idea of how Seq2Seq models work, making it suitable for beginners who want to understand the core concepts.

---

## What is a Seq2Seq Model?
A Seq2Seq (Sequence-to-Sequence) model is a type of neural network architecture commonly used for tasks where an input sequence is transformed into an output sequence. Examples include machine translation (e.g., English to French), text summarization, and more. The model consists of two main parts:
- **Encoder:** Processes the input sequence and summarizes its information into a context (hidden state).
- **Decoder:** Uses this context to generate the output sequence.

---

## Code Walkthrough

### 1. Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
```
We use PyTorch for building and training the neural network.

### 2. Encoder
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
    def forward(self, x):
        x = self.embed(x)
        _, hidden = self.rnn(x)
        return hidden
```
- **Embedding Layer:** Converts word indices into dense vectors.
- **GRU Layer:** Processes the embedded sequence and returns the final hidden state, which summarizes the input.

### 3. Decoder
```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden):
        x = self.embed(x)
        output, _ = self.rnn(x, hidden)
        logits = self.out(output)
        return logits
```
- **Embedding Layer:** Same as in the encoder.
- **GRU Layer:** Takes the previous hidden state (from the encoder) and the target sequence so far.
- **Linear Layer:** Maps the GRU output to vocabulary size for prediction.

### 4. Dummy Data
```python
vocab_size = 10
in_seq = torch.tensor([[1, 2, 3]])
out_seq = torch.tensor([[3, 2, 1]])
```
- **vocab_size:** Number of unique tokens in the vocabulary.
- **in_seq:** Example input sequence (batch size 1).
- **out_seq:** Example output sequence (batch size 1).

### 5. Model Initialization
```python
enc = Encoder(vocab_size, 16, 32)
dec = Decoder(vocab_size, 16, 32)
```
- **enc:** The encoder model.
- **dec:** The decoder model.
- `16` is the embedding size, `32` is the hidden size.

### 6. Training Setup
```python
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.01)
```
- **Loss Function:** Measures how well the model predicts the target sequence.
- **Optimizer:** Updates model parameters to minimize the loss.

### 7. Training Loop
```python
for i in range(300):
    hidden = enc(in_seq)
    output_logits = dec(out_seq[:, :-1], hidden)
    loss = loss_fn(output_logits.reshape(-1, vocab_size), out_seq[:, 1:].reshape(-1))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if i % 50 == 0:
        print(f'Epoch {i}, Loss: {loss.item():.4f}')
```
- **hidden = enc(in_seq):** Get the context from the encoder.
- **output_logits = dec(out_seq[:, :-1], hidden):** Decoder predicts the next token for each position in the output sequence, given the previous tokens and the encoder's hidden state.
- **loss:** Compares the decoder's predictions to the actual target sequence (shifted by one position).
- **optimiser.zero_grad():** Clears previous gradients.
- **loss.backward():** Computes gradients.
- **optimiser.step():** Updates model parameters.
- **Prints loss every 50 epochs.**

---

## Notes
- This is a minimal example for learning purposes. Real Seq2Seq models for tasks like translation use much larger datasets, more complex tokenization, and often attention mechanisms.
- The code uses dummy data for demonstration.

---

## How to Run
1. Make sure you have PyTorch installed: `pip install torch`
2. Run the script:
   ```bash
   python main.py
   ```

---

## License
This code is provided for educational purposes.
