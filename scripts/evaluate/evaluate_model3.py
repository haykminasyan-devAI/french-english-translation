"""
Evaluate Model 3: Transformer
Show sample translations and compute BLEU score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}\n")

# Load vocabularies
processed_path = Path('./data/processed')
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_word2idx = vocab_data['en_word2idx']
    fr_idx2word = vocab_data['fr_idx2word']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

# Load embeddings
with open(processed_path / 'embeddings.pkl', 'rb') as f:
    embeddings_data = pickle.load(f)
    fr_embedding_tensor = torch.FloatTensor(embeddings_data['fr_embedding_matrix'])
    en_embedding_tensor = torch.FloatTensor(embeddings_data['en_embedding_matrix'])

print(f"✅ Loaded vocabularies and embeddings\n")

# Load test data
df = pd.read_csv(processed_path / 'questions.csv')
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_df = df[train_size + val_size:]

print(f"Test samples: {len(test_df):,}\n")

# ============================================================================
# TRANSFORMER MODEL ARCHITECTURE (same as train__model_3.py)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(output)
        
        return output, attention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        
        attn_output, attention = self.cross_attn(tgt, enc_output, enc_output, src_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt, attention


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != d_model:
                self.emb_projection = nn.Linear(pretrained_embeddings.size(1), d_model, bias=False)
                temp_emb = nn.Embedding(vocab_size, pretrained_embeddings.size(1), padding_idx=0)
                temp_emb.weight.data.copy_(pretrained_embeddings)
                self.temp_emb = temp_emb
            else:
                self.embedding.weight.data.copy_(pretrained_embeddings)
                self.temp_emb = None
                self.emb_projection = None
        else:
            self.temp_emb = None
            self.emb_projection = None
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        if self.temp_emb is not None:
            x = self.temp_emb(src)
            x = self.emb_projection(x)
        else:
            x = self.embedding(src)
        
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != d_model:
                self.emb_projection = nn.Linear(pretrained_embeddings.size(1), d_model, bias=False)
                temp_emb = nn.Embedding(vocab_size, pretrained_embeddings.size(1), padding_idx=0)
                temp_emb.weight.data.copy_(pretrained_embeddings)
                self.temp_emb = temp_emb
            else:
                self.embedding.weight.data.copy_(pretrained_embeddings)
                self.temp_emb = None
                self.emb_projection = None
        else:
            self.temp_emb = None
            self.emb_projection = None
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        if self.temp_emb is not None:
            x = self.temp_emb(tgt)
            x = self.emb_projection(x)
        else:
            x = self.embedding(tgt)
        
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, enc_output, tgt_mask, src_mask)
            attentions.append(attention)
        
        output = self.fc_out(x)
        return output, attentions


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encoder(src, src_mask)
        output, attentions = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return output


def translate_sentence(model, sentence, src_word2idx, tgt_idx2word, max_len=50):
    """Translate using Transformer with greedy decoding"""
    model.eval()
    
    tokens = sentence.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_output = model.encoder(src_tensor, src_mask)
        
        tgt_indices = [2]  # Start with <sos>
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            output, _ = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
            next_token = output[0, -1].argmax().item()
            
            if next_token == 3:  # <eos>
                break
            
            tgt_indices.append(next_token)
        
        translation = ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in tgt_indices[1:]])
    
    return translation


def compute_bleu(reference, candidate):
    """Compute simple BLEU score"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    ref_1grams = set(ref_tokens)
    cand_1grams = set(cand_tokens)
    overlap = len(ref_1grams & cand_1grams)
    precision = overlap / len(cand_tokens) * 100 if len(cand_tokens) > 0 else 0
    
    return precision


# Load Model 3
print("📥 Loading Model 3 (Transformer)...")

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_LEN = 200

encoder = TransformerEncoder(
    vocab_size=len(fr_vocab),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    pretrained_embeddings=fr_embedding_tensor
)

decoder = TransformerDecoder(
    vocab_size=len(en_vocab),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    pretrained_embeddings=en_embedding_tensor
)

model3 = Transformer(encoder, decoder, src_pad_idx=0, tgt_pad_idx=0).to(device)
model3.load_state_dict(torch.load('best_model3.pt', map_location=device, weights_only=False))
model3.eval()

print(f"✅ Model 3 loaded!\n")

# Evaluate
print("="*80)
print("MODEL 3 (TRANSFORMER): SAMPLE TRANSLATIONS")
print("="*80)

# Show 20 random examples
samples = test_df.sample(20, random_state=42)

for i, (_, row) in enumerate(samples.iterrows(), 1):
    fr = row['fr']
    en_true = row['en']
    en_pred = translate_sentence(model3, fr, fr_word2idx, en_idx2word)
    
    print(f"\n{i}.")
    print(f"  FR:        {fr}")
    print(f"  EN (true): {en_true}")
    print(f"  EN (pred): {en_pred}")

# Compute BLEU on full test set
print("\n\n" + "="*80)
print("COMPUTING BLEU SCORE ON FULL TEST SET...")
print("="*80)

bleu_scores = []
for idx, row in test_df.iterrows():
    pred = translate_sentence(model3, row['fr'], fr_word2idx, en_idx2word)
    bleu = compute_bleu(row['en'], pred)
    bleu_scores.append(bleu)

avg_bleu = np.mean(bleu_scores)

print(f"\n📊 Model 3 BLEU Score: {avg_bleu:.2f}")
print(f"\n   Distribution:")
print(f"     Min:    {min(bleu_scores):.2f}")
print(f"     Median: {np.median(bleu_scores):.2f}")
print(f"     Max:    {max(bleu_scores):.2f}")

print("\n" + "="*80)
print("✅ Evaluation Complete!")
print("="*80)
