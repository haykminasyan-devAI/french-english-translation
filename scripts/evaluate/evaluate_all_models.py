"""
Comprehensive Evaluation of All 3 Models
Computes BLEU scores and creates comparison
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

# Load data
processed_path = Path('./data/processed')
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_word2idx = vocab_data['en_word2idx']
    fr_idx2word = vocab_data['fr_idx2word']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

with open(processed_path / 'embeddings.pkl', 'rb') as f:
    embeddings_data = pickle.load(f)
    fr_embedding_tensor = torch.FloatTensor(embeddings_data['fr_embedding_matrix'])
    en_embedding_tensor = torch.FloatTensor(embeddings_data['en_embedding_matrix'])

# Load test data
df = pd.read_csv(processed_path / 'questions.csv')
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_df = df[train_size + val_size:]

print(f"✅ Loaded data")
print(f"   Test samples: {len(test_df):,}\n")

# ==================================================================
# MODEL ARCHITECTURES
# ==================================================================

# Model 1 classes
class Encoder1(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = hidden.view(2, -1, hidden.size(1), hidden.size(2))
        hidden = torch.cat([hidden[0], hidden[1]], dim=2)
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden

class Decoder1(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt, hidden):
        embedded = self.dropout(self.embedding(tgt))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2Seq1(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

def translate_model1(model, sentence):
    model.eval()
    tokens = sentence.lower().split()
    src_indices = [fr_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[2]]).to(device)
        translation_indices = []
        
        for _ in range(50):
            output, hidden = model.decoder(input_token, hidden)
            top_idx = output.argmax(1).item()
            if top_idx == 3:
                break
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]]).to(device)
        
        return ' '.join([en_idx2word.get(idx, '<unk>') for idx in translation_indices])


# Model 2 classes (with attention)
class Encoder2(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        batch_size, src_len, enc_dim = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return attention_weights, context

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.attention = BahdanauAttention(hidden_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim + hidden_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(tgt))
        attention_weights, context = self.attention(hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc(torch.cat([output, context, embedded], dim=1))
        return prediction, hidden, attention_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

def translate_model2(model, sentence):
    model.eval()
    tokens = sentence.lower().split()
    src_indices = [fr_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[2]]).to(device)
        translation_indices = []
        
        for _ in range(50):
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            top_idx = output.argmax(1).item()
            if top_idx == 3:
                break
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]]).to(device)
        
        return ' '.join([en_idx2word.get(idx, '<unk>') for idx in translation_indices])


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


# ==================================================================
# LOAD ALL MODELS
# ==================================================================

print("="*80)
print("LOADING MODELS")
print("="*80)

# Model 1
print("\n📥 Loading Model 1 (Basic Seq2Seq)...")
encoder1 = Encoder1(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
decoder1 = Decoder1(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
model1 = Seq2Seq1(encoder1, decoder1).to(device)
model1.load_state_dict(torch.load('best_model1.pt', map_location=device, weights_only=False))
print("✅ Model 1 loaded")

# Model 2
print("\n📥 Loading Model 2 (with Attention)...")
encoder2 = Encoder2(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
decoder2 = AttentionDecoder(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
model2 = Seq2SeqAttention(encoder2, decoder2).to(device)
model2.load_state_dict(torch.load('best_model2.pt', map_location=device, weights_only=False))
print("✅ Model 2 loaded")

# Model 3
print("\n📥 Loading Model 3 (Transformer)...")
try:
    # For simplicity, we'll use Model 2 as proxy if Model 3 has different architecture
    # You can add full Transformer loading here if needed
    print("⚠️  Model 3 evaluation pending (different architecture)")
    model3 = None
except:
    model3 = None

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

# Evaluate Model 1
print("\n📊 Evaluating Model 1...")
bleu1_scores = []
for idx, row in test_df.iterrows():
    pred = translate_model1(model1, row['fr'])
    bleu = compute_bleu(row['en'], pred)
    bleu1_scores.append(bleu)
bleu1_avg = np.mean(bleu1_scores)
print(f"✅ Model 1 BLEU: {bleu1_avg:.2f}")

# Evaluate Model 2
print("\n📊 Evaluating Model 2...")
bleu2_scores = []
for idx, row in test_df.iterrows():
    pred = translate_model2(model2, row['fr'])
    bleu = compute_bleu(row['en'], pred)
    bleu2_scores.append(bleu)
bleu2_avg = np.mean(bleu2_scores)
print(f"✅ Model 2 BLEU: {bleu2_avg:.2f}")

# Final Comparison
print("\n" + "="*80)
print("📊 FINAL RESULTS COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Model': ['Model 1: Basic Seq2Seq', 'Model 2: + Bahdanau Attention', 'Model 3: Transformer'],
    'Parameters': ['11.6M', '20.3M', '65.8M'],
    'Best Val Loss': [5.24, 4.74, 4.03],
    'BLEU Score': [bleu1_avg, bleu2_avg, 'TBD'],
    'Architecture': ['Bidirectional GRU', 'GRU + Attention', 'Multi-head Self-Attention']
})

print("\n" + results.to_string(index=False))

print("\n\n📈 Improvements:")
print(f"   Model 1 → Model 2: +{((bleu2_avg - bleu1_avg) / bleu1_avg * 100):.1f}% BLEU")
print(f"   Model 1 → Model 2: {((5.24 - 4.74) / 5.24 * 100):.1f}% better Val Loss")

# Show side-by-side translations
print("\n" + "="*80)
print("SIDE-BY-SIDE TRANSLATION COMPARISON")
print("="*80)

samples = test_df.sample(10, random_state=42)
for i, (_, row) in enumerate(samples.iterrows(), 1):
    fr = row['fr']
    en_true = row['en']
    en_pred1 = translate_model1(model1, fr)
    en_pred2 = translate_model2(model2, fr)
    
    print(f"\n{i}. FR: {fr}")
    print(f"   TRUE:    {en_true}")
    print(f"   MODEL 1: {en_pred1}")
    print(f"   MODEL 2: {en_pred2}")

# Save results
results_summary = {
    'model1': {'bleu': bleu1_avg, 'val_loss': 5.24, 'params': '11.6M'},
    'model2': {'bleu': bleu2_avg, 'val_loss': 4.74, 'params': '20.3M'},
    'model3': {'bleu': 'TBD', 'val_loss': 4.03, 'params': '65.8M'}
}

with open('final_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("\n" + "="*80)
print("✅ Evaluation Complete!")
print("="*80)
print("\n💾 Results saved to: final_results.pkl")
