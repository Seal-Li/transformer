import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.nhead = nhead
        self.d_k = d_model // nhead
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(nhead * self.d_k, d_model)  # Updated output_projection

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        
        query = query.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        scaled_attention = torch.matmul(attention_weights, value)
        scaled_attention = scaled_attention.transpose(1, 2)  # swap num_heads and seq_len
        scaled_attention = scaled_attention.contiguous().view(batch_size, -1, self.nhead * self.d_k)  # combine num_heads and d_k

        output = self.output_projection(scaled_attention)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 加入batch的维度
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 使pe变为不学习的参数，并说用 .detach() 来防止自动梯度计算
        self.register_buffer('pe', pe.detach())

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear2(output)
        output = self.dropout(output)

        output += x

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):

        src2 = self.self_attention(src, src, src, mask)

        src = src + self.dropout(src2)
        src = self.layer_norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.layer_norm2(src)

        return src
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        attended_tgt = self.self_attention(tgt, tgt, tgt, tgt_mask)

        tgt = tgt + self.dropout(attended_tgt)
        tgt = self.norm1(tgt)

        attended_memory, _ = self.encoder_attention(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout(attended_memory)
        tgt = self.norm2(tgt)

        linear_out = self.linear2(F.relu(self.linear1(tgt)))
        tgt = tgt + self.dropout(linear_out)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        output = src

        for layer in self.layers:
            output = layer(output, mask)

        output = self.dropout(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, src_mask, tgt_mask)

        output = self.dropout(output)

        return output


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, d_model, nhead, d_ff, src_pad_idx, tgt_pad_idx, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.src_position_encoding = PositionalEncoding(d_model)
        self.tgt_position_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = TransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)
        self.transformer_decoder = TransformerDecoder(num_layers, d_model, nhead, d_ff, dropout)
        self.src_pad_idx = src_pad_idx  # padding index for source sequences
        self.tgt_pad_idx = tgt_pad_idx  # padding index for target sequences

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_mask, tgt_mask = self.generate_masks(src, tgt)
        src_embedding = self.src_position_encoding(self.src_word_embedding(src))
        tgt_embedding = self.tgt_position_encoding(self.tgt_word_embedding(tgt))
        src_output = self.transformer_encoder(src_embedding, src_mask)
        tgt_output = self.transformer_decoder(tgt_embedding, src_output, tgt_mask)
        return tgt_output

    def generate_masks(self, src, tgt): 
        src_padding_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        tgt_padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)

        tgt_no_look_ahead_mask = (1 - torch.triu(torch.ones((1, 1, tgt.size(1), tgt.size(1))), diagonal=1)).type_as(tgt_padding_mask)

        src_mask = src_padding_mask
        tgt_mask = tgt_padding_mask & tgt_no_look_ahead_mask  # apply no look ahead mask and padding mask

        return src_mask, tgt_mask
