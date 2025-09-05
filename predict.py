import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda")

with open('./intro_transformer.txt','r',encoding='utf-8') as f:
        text = f.read()

chars = sorted(list(set(text)))
char_size = len(chars)
char2int = { ch : i for i, ch in enumerate(chars) }
int2char = { i : ch for i, ch in enumerate(chars) }
encode = lambda a: [char2int[b] for b in a ]
decode = lambda a: ''.join([int2char[b] for b in a ])

class SelfAttention_Head(nn.Module):

    def __init__(self, n_mbed, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_mbed, head_size, bias=False)
        self.query = nn.Linear(n_mbed, head_size, bias=False)
        self.value = nn.Linear(n_mbed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)* C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out
    
class SelfAttention_MultiHeads(nn.Module):

    def __init__(self, n_mbed, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList((SelfAttention_Head(n_mbed, head_size, block_size) for _ in range(num_heads)))

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)

class FeedForward(nn.Module):

    def __init__(self, n_mbed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_mbed, n_mbed), nn.ReLU())

    def forward(self, x):
        return self.net(x)
    
class Model(nn.Module):
    def __init__(self, n_mbed, char_size, block_size, number_of_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(char_size, n_mbed)
        self.position_embedding = nn.Embedding(block_size, n_mbed)
        self.selfattention_multiheads = SelfAttention_MultiHeads(n_mbed, number_of_heads, n_mbed//number_of_heads, block_size)
        self.feedforward = FeedForward(n_mbed)
        self.linear = nn.Linear(n_mbed , char_size)

    def forward(self, idx, targets=None):
        B, T= idx.shape
        token_mbed = self.token_embedding(idx)
        position_mbed = self.position_embedding(torch.arange(T))
        x = token_mbed + position_mbed
        x = self.selfattention_multiheads(x)
        x = self.feedforward(x)
        logits = self.linear(x)

        loss = None
        if targets is not None:
            B, T, C =logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
#number_of_heads = 32 # 同時に実行されるself-attentionの数
#block_size = 512 # 一度に処理できる最大の文字数
#n_mbed = 256 # トークンの埋め込むベクトルの次元数
#batch_size = 512 # 同時に処理できる配列の数
number_of_heads = 32 # 同時に実行されるself-attentionの数
block_size = 512 # 一度に処理できる最大の文字数
n_mbed = 256 # トークンの埋め込むベクトルの次元数
batch_size = 512 # 同時に処理できる配列の数
temperature = 1.0

model = Model(n_mbed, char_size, block_size, number_of_heads)

model.load_state_dict(torch.load('model_weights.pth'))

#idx = torch.zeros((1,1), dtype = torch.long)
while True:
    input_text = input("Input test:\n")
    idx = torch.tensor([encode(input_text)], dtype=torch.long)
    for _ in range(400):
        idx_pred = idx[:, -block_size:]
        logits , loss = model(idx_pred)
        logits = logits[:,-1,:]
        probs = F.softmax(logits/temperature, dim=1)
        idx_next_pred = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next_pred),dim = 1)

    predict = decode(idx[0].tolist())
    print("予測結果 : ", predict)

