
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from transformer import Transformer

# 训练和目标语言
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Special symbols
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# token转换函数
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en')

def yield_tokens(data_iter):
    for data_sample in data_iter:
        yield token_transform[SRC_LANGUAGE](data_sample[0])
        yield token_transform[TGT_LANGUAGE](data_sample[1])

vocab_transform = {}
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_symbols)
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_symbols)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

def preprocess(src_sentence, tgt_sentence):
    src_indexes = [vocab_transform[SRC_LANGUAGE][token] for token in token_transform[SRC_LANGUAGE](src_sentence)]
    tgt_indexes = [vocab_transform[TGT_LANGUAGE][token] for token in token_transform[TGT_LANGUAGE](tgt_sentence)]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(1).to(device)

    return src_tensor, tgt_tensor

# Model Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = Transformer(num_layers=6, d_model=512, nhead=8, d_ff=2048, 
                    src_vocab_size=len(vocab_transform[SRC_LANGUAGE]), 
                    tgt_vocab_size=len(vocab_transform[TGT_LANGUAGE]),
                    src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX).to(device)

#定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

#定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
for epoch in range(5):
    total_loss = 0
    for i, (src_sentence, tgt_sentence) in enumerate(train_iter):
        optimizer.zero_grad()
        src, tgt = preprocess(src_sentence, tgt_sentence)
        output = model(src, tgt) 
        
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        tgt = tgt.view(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:  # 每100个batch打印一次训练信息
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {total_loss / (i+1)}')

# Save model parameters
torch.save(model.state_dict(), 'model.pth')
