import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(train_iter), specials=special_symbols,)

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

# Creating the model
model = Transformer(src_pad_idx=PAD_IDX, 
                    trg_pad_idx=PAD_IDX, 
                    trg_sos_idx=BOS_IDX,
                    enc_voc_size=len(vocab_transform[SRC_LANGUAGE]), 
                    dec_voc_size=len(vocab_transform[TGT_LANGUAGE]),
                    d_model=512, n_head=8, max_len=100, 
                    ffn_hidden=2048, n_layers=6, 
                    drop_prob=0.1, 
                    device=device).to(device)

# Initialize model weights
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Define learningrate scheduler
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

# Training loop
for epoch in range(50):  # 50 epochs
    model.train()
    total_loss = 0
    total_batches = 0

    for i, (src_sentence, tgt_sentence) in enumerate(train_iter):
        optimizer.zero_grad()
        src, tgt = preprocess(src_sentence, tgt_sentence)
        print("Size of source tensor: ", src.shape)
        print("Size of target tensor: ", tgt.shape)
        
        # tgt[:-1] is used to remove the last element in each sentence in the batch, since the decoder should not receive it
        output = model(src, tgt[:-1])
        print("Size of source tensor after model: ", src.shape)
        print("Size of target tensor after model: ", tgt[:-1].shape)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        
        # tgt[1:] is used to remove <sos> in each sentence in the batch, as the model is not supposed to generate it
        tgt = tgt[1:].contiguous().view(-1)

        print("Size of target tensor after reshaping: ", tgt.shape)
        
        loss = criterion(output, tgt)
        loss.backward()
    
        # Clipping the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1

        # Logging
        if total_batches % 1000 == 0:
            avg_loss = total_loss/1000
            print('Epoch:', epoch, 'Batch:', total_batches, 'Avg Batch Loss:', avg_loss)
            total_loss = 0
            
    # Evaluation loop with(validation data), early stopping, model checkpoint saving, etc.
    # TODO

    # Decay learning rate
    scheduler.step(avg_loss)

# Save model parameters
torch.save(model.state_dict(), 'model.pth')
