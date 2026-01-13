"""
Simple script to show 30 test translations
No interaction needed - just shows results
"""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
import pandas as pd

device = torch.device('cpu')
processed_path = Path('./data/processed')

# Load vocabularies
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

# Load embeddings
with open(processed_path / 'embeddings.pkl', 'rb') as f:
    embeddings_data = pickle.load(f)
    fr_embedding_tensor = torch.FloatTensor(embeddings_data['fr_embedding_matrix'])
    en_embedding_tensor = torch.FloatTensor(embeddings_data['en_embedding_matrix'])

# Model classes
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

def translate(sentence, model, src_word2idx, tgt_idx2word):
    model.eval()
    tokens = sentence.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0)
    
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[2]])
        translation_indices = []
        
        for _ in range(50):
            output, hidden = model.decoder(input_token, hidden)
            top_idx = output.argmax(1).item()
            if top_idx == 3:
                break
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]])
        
        return ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in translation_indices])

# Load model
print("Loading Model 1...")
encoder = Encoder(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
decoder = Decoder(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load('best_model1.pt', map_location=device, weights_only=False))
print("✅ Model loaded!\n")

# Load test data
df = pd.read_csv(processed_path / 'questions.csv')
test_df = df[int(0.8 * len(df)) + int(0.1 * len(df)):]

# Show 30 random examples
print("="*80)
print("MODEL 1: SAMPLE TRANSLATIONS FROM TEST SET")
print("="*80)

samples = test_df.sample(30, random_state=42)
for i, (_, row) in enumerate(samples.iterrows(), 1):
    fr = row['fr']
    en_true = row['en']
    en_pred = translate(fr, model, fr_word2idx, en_idx2word)
    
    print(f"\n{i}.")
    print(f"  FR:        {fr}")
    print(f"  EN (true): {en_true}")
    print(f"  EN (pred): {en_pred}")

print("\n" + "="*80)
print("✅ Done! Run again to see different random samples.")
print("="*80)
