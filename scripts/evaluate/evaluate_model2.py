"""
Evaluate Model 2: Seq2Seq with Bahdanau Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

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

print(f"✅ Loaded data and embeddings\n")

# Model Architecture (Model 2 with Attention)
class Encoder(nn.Module):
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
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
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


def translate_sentence(model, sentence, src_word2idx, tgt_idx2word, max_len=50):
    """Translate with attention"""
    model.eval()
    
    tokens = sentence.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[2]]).to(device)
        translation_indices = []
        
        for _ in range(max_len):
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            top_idx = output.argmax(1).item()
            
            if top_idx == 3:
                break
            
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]]).to(device)
        
        translation = ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in translation_indices])
    
    return translation


def compute_bleu(reference, candidate):
    """Simple BLEU score"""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # 1-gram precision
    ref_1grams = set(ref_tokens)
    cand_1grams = set(cand_tokens)
    overlap = len(ref_1grams & cand_1grams)
    precision = overlap / len(cand_tokens) * 100 if len(cand_tokens) > 0 else 0
    
    return precision


# Load Model 2
print("📥 Loading Model 2 (with Bahdanau Attention)...")
encoder = Encoder(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
decoder = AttentionDecoder(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
model = Seq2SeqAttention(encoder, decoder).to(device)
model.load_state_dict(torch.load('best_model2.pt', map_location=device, weights_only=False))
model.eval()

print(f"✅ Model loaded successfully!\n")

# Load test data (last 10%)
df = pd.read_csv(processed_path / 'questions.csv')
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_df = df[train_size + val_size:]

print("="*80)
print("MODEL 2: TEST RESULTS (Attention)")
print("="*80)
print(f"Test samples: {len(test_df):,}\n")

# Evaluate on test set
bleu_scores = []
print("Computing metrics...")

for idx, row in test_df.iterrows():
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    bleu = compute_bleu(english_true, english_pred)
    bleu_scores.append(bleu)

avg_bleu = np.mean(bleu_scores)

print("\n" + "="*80)
print("📊 OVERALL METRICS")
print("="*80)
print(f"\n   BLEU Score: {avg_bleu:.2f}")
print(f"\n   Distribution:")
print(f"     Min:    {min(bleu_scores):.2f}")
print(f"     Median: {np.median(bleu_scores):.2f}")
print(f"     Max:    {max(bleu_scores):.2f}")

# Show random 20 examples
print("\n" + "="*80)
print("SAMPLE TRANSLATIONS")
print("="*80)

samples = test_df.sample(20, random_state=42)
for i, (_, row) in enumerate(samples.iterrows(), 1):
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    print(f"\n{i}.")
    print(f"  FR:        {french}")
    print(f"  EN (true): {english_true}")
    print(f"  EN (pred): {english_pred}")

# Compare with Model 1
print("\n" + "="*80)
print("📊 COMPARISON: Model 1 vs Model 2")
print("="*80)

# Load Model 1 results if available
try:
    with open('model1_test_results.pkl', 'rb') as f:
        model1_results = pickle.load(f)
        model1_bleu = model1_results['bleu']
    
    print(f"\n   Model 1 (Baseline):     BLEU = {model1_bleu:.2f}")
    print(f"   Model 2 (+ Attention):  BLEU = {avg_bleu:.2f}")
    print(f"\n   Improvement: {((avg_bleu - model1_bleu) / model1_bleu * 100):.1f}%")
except:
    print(f"\n   Model 2 BLEU: {avg_bleu:.2f}")
    print(f"   (Model 1 results not found for comparison)")

print("\n" + "="*80)
print("✅ Evaluation complete!")
print("="*80)
