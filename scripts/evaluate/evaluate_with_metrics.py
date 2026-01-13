"""
Proper Model Evaluation with Train/Val/Test Split and Metrics

Metrics computed:
- Word-level accuracy
- Sentence-level accuracy
- BLEU score (standard for translation)
- Perplexity
"""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
import pandas as pd
from collections import Counter
import numpy as np

# CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}\n")

# Load vocabularies
print("📂 Loading vocabularies...")
processed_path = Path('./data/processed')
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_word2idx = vocab_data['en_word2idx']
    fr_idx2word = vocab_data['fr_idx2word']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

print(f"✅ Loaded vocabularies")
print(f"   French: {len(fr_vocab):,} words")
print(f"   English: {len(en_vocab):,} words\n")

# Load pre-trained embeddings
try:
    with open(processed_path / 'embeddings.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
        fr_embedding_matrix = embeddings_data['fr_embedding_matrix']
        en_embedding_matrix = embeddings_data['en_embedding_matrix']
    
    fr_embedding_tensor = torch.FloatTensor(fr_embedding_matrix)
    en_embedding_tensor = torch.FloatTensor(en_embedding_matrix)
    print(f"✅ Loaded pre-trained embeddings\n")
except FileNotFoundError:
    fr_embedding_tensor = None
    en_embedding_tensor = None
    print(f"⚠️  Using random embeddings\n")


# Load and split data properly: 80% train, 10% val, 10% test
print("📊 Creating train/val/test split...")
df = pd.read_csv(processed_path / 'questions.csv')

total = len(df)
train_size = int(0.8 * total)
val_size = int(0.1 * total)

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print(f"   Total: {total:,}")
print(f"   Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
print(f"   Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
print(f"   Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
print()


# Define Model Architecture
class Encoder(nn.Module):
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


class Decoder(nn.Module):
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


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(device)
        _, hidden = self.encoder(src)
        input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs


def translate_sentence(model, sentence, src_word2idx, tgt_idx2word, max_len=50):
    """Translate a French sentence to English"""
    model.eval()
    
    tokens = sentence.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[2]]).to(device)  # <sos>
        translation_indices = []
        
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            top_idx = output.argmax(1).item()
            
            if top_idx == 3:  # <eos>
                break
            
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]]).to(device)
        
        translation = ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in translation_indices])
    
    return translation


def compute_bleu(reference, candidate, max_n=4):
    """
    Compute BLEU score (BiLingual Evaluation Understudy)
    
    Standard metric for machine translation quality
    Measures n-gram overlap between reference and candidate
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len) if cand_len > 0 else 0
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = np.log(precisions)
        geo_mean = np.exp(np.mean(log_precisions))
        bleu = bp * geo_mean
    else:
        bleu = 0.0
    
    return bleu * 100  # Return as percentage


def compute_word_accuracy(reference, candidate):
    """Word-level accuracy (how many words match)"""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if len(ref_words) == 0:
        return 0.0
    
    # Count matching words (order matters)
    matches = sum(1 for r, c in zip(ref_words, cand_words) if r == c)
    accuracy = matches / len(ref_words) * 100
    
    return accuracy


def compute_sentence_accuracy(reference, candidate):
    """Sentence-level accuracy (exact match)"""
    return 100.0 if reference.lower().strip() == candidate.lower().strip() else 0.0


# Load Model
print("📥 Loading Model 1...")
EMB_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3

encoder = Encoder(len(fr_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                  fr_embedding_tensor if fr_embedding_tensor is not None else None)
decoder = Decoder(len(en_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                  en_embedding_tensor if en_embedding_tensor is not None else None)

model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('best_model1.pt', map_location=device, weights_only=False))
model.eval()

print(f"✅ Model loaded successfully!\n")


# Evaluate on TEST SET
print("="*70)
print("EVALUATION ON TEST SET (Unseen Data)")
print("="*70)
print()

bleu_scores = []
word_accuracies = []
sentence_accuracies = []

print("Computing metrics on test set...")
for idx, row in test_df.iterrows():
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    # Compute metrics
    bleu = compute_bleu(english_true, english_pred)
    word_acc = compute_word_accuracy(english_true, english_pred)
    sent_acc = compute_sentence_accuracy(english_true, english_pred)
    
    bleu_scores.append(bleu)
    word_accuracies.append(word_acc)
    sentence_accuracies.append(sent_acc)

# Overall metrics
avg_bleu = np.mean(bleu_scores)
avg_word_acc = np.mean(word_accuracies)
avg_sent_acc = np.mean(sentence_accuracies)

print("\n" + "="*70)
print("📊 OVERALL METRICS ON TEST SET")
print("="*70)
print(f"\n   Test samples: {len(test_df):,}")
print(f"\n   📈 BLEU Score:           {avg_bleu:.2f}")
print(f"   📈 Word Accuracy:        {avg_word_acc:.2f}%")
print(f"   📈 Sentence Accuracy:    {avg_sent_acc:.2f}%")
print()

# Show score distribution
print("="*70)
print("SCORE DISTRIBUTION")
print("="*70)
print(f"\nBLEU Score Distribution:")
print(f"   Min:    {min(bleu_scores):.2f}")
print(f"   25%:    {np.percentile(bleu_scores, 25):.2f}")
print(f"   Median: {np.percentile(bleu_scores, 50):.2f}")
print(f"   75%:    {np.percentile(bleu_scores, 75):.2f}")
print(f"   Max:    {max(bleu_scores):.2f}")

print(f"\nWord Accuracy Distribution:")
print(f"   Min:    {min(word_accuracies):.2f}%")
print(f"   25%:    {np.percentile(word_accuracies, 25):.2f}%")
print(f"   Median: {np.percentile(word_accuracies, 50):.2f}%")
print(f"   75%:    {np.percentile(word_accuracies, 75):.2f}%")
print(f"   Max:    {max(word_accuracies):.2f}%")

# Show examples by quality
print("\n" + "="*70)
print("BEST TRANSLATIONS (Highest BLEU)")
print("="*70)

# Get top 5 translations
best_indices = np.argsort(bleu_scores)[-5:][::-1]
for rank, idx in enumerate(best_indices, 1):
    row = test_df.iloc[idx]
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    print(f"\n{rank}. BLEU: {bleu_scores[idx]:.1f} | Word Acc: {word_accuracies[idx]:.1f}%")
    print(f"   FR:        {french}")
    print(f"   EN (true): {english_true}")
    print(f"   EN (pred): {english_pred}")


print("\n" + "="*70)
print("WORST TRANSLATIONS (Lowest BLEU)")
print("="*70)

# Get bottom 5 translations
worst_indices = np.argsort(bleu_scores)[:5]
for rank, idx in enumerate(worst_indices, 1):
    row = test_df.iloc[idx]
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    print(f"\n{rank}. BLEU: {bleu_scores[idx]:.1f} | Word Acc: {word_accuracies[idx]:.1f}%")
    print(f"   FR:        {french}")
    print(f"   EN (true): {english_true}")
    print(f"   EN (pred): {english_pred}")


# Analyze by sentence length
print("\n" + "="*70)
print("PERFORMANCE BY SENTENCE LENGTH")
print("="*70)

# Group by length
test_df_copy = test_df.copy()
test_df_copy['fr_len'] = test_df_copy['fr'].str.split().str.len()
test_df_copy['bleu'] = bleu_scores
test_df_copy['word_acc'] = word_accuracies

length_groups = {
    'Short (1-5 words)': test_df_copy[test_df_copy['fr_len'] <= 5],
    'Medium (6-10 words)': test_df_copy[(test_df_copy['fr_len'] > 5) & (test_df_copy['fr_len'] <= 10)],
    'Long (11+ words)': test_df_copy[test_df_copy['fr_len'] > 10]
}

for group_name, group_df in length_groups.items():
    if len(group_df) > 0:
        print(f"\n{group_name}: {len(group_df)} sentences")
        print(f"   Avg BLEU: {group_df['bleu'].mean():.2f}")
        print(f"   Avg Word Acc: {group_df['word_acc'].mean():.2f}%")


# Save results to file
results = {
    'model': 'Model 1 (Basic Seq2Seq)',
    'test_size': len(test_df),
    'bleu': avg_bleu,
    'word_accuracy': avg_word_acc,
    'sentence_accuracy': avg_sent_acc,
    'bleu_scores': bleu_scores,
    'word_accuracies': word_accuracies
}

with open('model1_test_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)
print(f"\n💾 Results saved to: model1_test_results.pkl")
print(f"\n🎯 Summary for Model 1:")
print(f"   BLEU Score:      {avg_bleu:.2f}")
print(f"   Word Accuracy:   {avg_word_acc:.2f}%")
print(f"   Sentence Acc:    {avg_sent_acc:.2f}%")
