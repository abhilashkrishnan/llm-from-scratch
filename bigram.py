import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

block_size = 8
batch_size = 4

with open('shakespeare_macbeth.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data[:100])

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
