import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,hidden_size,batch_first=True)
    
    def forward(self, x):
        x=self.embed(x)
        _,hidden=self.rnn(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size,hidden_size ):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,hidden_size,batch_first=True)
        self.out=nn.Linear(hidden_size,vocab_size)

    def forward(self, x, hidden):
        x=self.embed(x)
        output,_=self.rnn(x, hidden)
        logits=self.out(output)
        return logits

'''Dummy data'''
vocab_size=10
in_seq=torch.tensor([[1,2,3]])
out_seq=torch.tensor([[3,2,1]])


#Init

enc=Encoder(vocab_size, 16,32)
dec=Decoder(vocab_size,16,32)

#Training
loss_fn=nn.CrossEntropyLoss()
optimiser=optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=0.01)


#Loop

for i in range(300):
    hidden=enc(in_seq)
    output_logits=dec(out_seq[:,:-1],hidden)

    loss=loss_fn(output_logits.reshape(-1, vocab_size),out_seq[:,1:].reshape(-1))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if i%50==0:
        print(f'Epoch {i}, Loss: {loss.item():.4f}')

