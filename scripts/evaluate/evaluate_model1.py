"""
Evaluate Model 1: Test translations and compute metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
import pandas as pd

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


# Define Model Architecture (same as training)
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
    """
    Translate a French sentence to English
    
    Args:
        model: Trained Seq2Seq model
        sentence: French sentence (string)
        src_word2idx: French vocabulary (word to index)
        tgt_idx2word: English vocabulary (index to word)
        max_len: Maximum output length
        
    Returns:
        translation: English translation (string)
    """
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in tokens]  # 1 = <unk>
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    
    with torch.no_grad():
        # Encode
        _, hidden = model.encoder(src_tensor)
        
        # Start decoding with <sos> token
        input_token = torch.LongTensor([[2]]).to(device)  # 2 = <sos>
        
        translation_indices = []
        
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            
            # Get most likely next word
            top_idx = output.argmax(1).item()
            
            # Stop if <eos>
            if top_idx == 3:  # 3 = <eos>
                break
            
            translation_indices.append(top_idx)
            input_token = torch.LongTensor([[top_idx]]).to(device)
        
        # Convert indices to words
        translation = ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in translation_indices])
    
    return translation


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

# Load trained weights
model.load_state_dict(torch.load('best_model1.pt', map_location=device))
model.eval()

print(f"✅ Model loaded successfully!\n")

# Test translations
print("="*70)
print("SAMPLE TRANSLATIONS")
print("="*70)

test_sentences = [
    "où sont les étoiles ?",
    "qu'est-ce que la lumière ?",
    "pourquoi le ciel est-il bleu ?",
    "quand est-ce que le soleil se lève ?",
    "comment fonctionne la gravité ?",
    "quelle est la température du soleil ?",
    "où se trouve mars ?",
    "qu'est-ce qu'un trou noir ?",
    "quand a été découvert neptune ?",
    "pourquoi la lune brille-t-elle ?"
]

for i, french in enumerate(test_sentences, 1):
    translation = translate_sentence(model, french, fr_word2idx, en_idx2word)
    print(f"\n{i}. FR: {french}")
    print(f"   EN: {translation}")

# Load validation data for comparison
print("\n\n" + "="*70)
print("VALIDATION SAMPLES (with ground truth)")
print("="*70)

df = pd.read_csv(processed_path / 'questions.csv')
val_df = df[int(0.9 * len(df)):]  # Last 10%

# Show 10 random examples
import random
random.seed(42)
samples = val_df.sample(10)

for idx, row in samples.iterrows():
    french = row['fr']
    english_true = row['en']
    english_pred = translate_sentence(model, french, fr_word2idx, en_idx2word)
    
    print(f"\nFR:        {french}")
    print(f"EN (true): {english_true}")
    print(f"EN (pred): {english_pred}")
    
    # Simple word overlap metric
    true_words = set(english_true.split())
    pred_words = set(english_pred.split())
    overlap = len(true_words & pred_words)
    total = len(true_words)
    accuracy = overlap / total * 100 if total > 0 else 0
    print(f"Word overlap: {overlap}/{total} ({accuracy:.1f}%)")

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)
