import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import time
import math

# CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True


# DATA LOADING
print("\n📂 Loading data...")
processed_path = Path('./data/processed')
df = pd.read_csv(processed_path / 'questions.csv')

# Load vocabularies
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_word2idx = vocab_data['en_word2idx']
    fr_idx2word = vocab_data['fr_idx2word']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

# Load pre-trained embeddings
try:
    with open(processed_path / 'embeddings.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
        fr_embedding_matrix = embeddings_data['fr_embedding_matrix']
        en_embedding_matrix = embeddings_data['en_embedding_matrix']
    
    fr_embedding_tensor = torch.FloatTensor(fr_embedding_matrix)
    en_embedding_tensor = torch.FloatTensor(en_embedding_matrix)
    USE_PRETRAINED = True
    print(f"✅ Loaded pre-trained embeddings")
except FileNotFoundError:
    fr_embedding_tensor = None
    en_embedding_tensor = None
    USE_PRETRAINED = False
    print(f"⚠️  Using random embeddings")

print(f"   Dataset: {len(df):,} pairs")
print(f"   French vocab: {len(fr_vocab):,} | English vocab: {len(en_vocab):,}")

# DATASET
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_word2idx, tgt_word2idx, max_len = 50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_word2idx = src_word2idx
        self.tgt_word2idx = tgt_word2idx

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx].split()
        tgt = self.tgt_sentences[idx].split()

        src_indices = [self.src_word2idx.get(w, 1) for w in src]
        tgt_indices = [2] + [self.tgt_word2idx.get(w, 1) for w in tgt] + [3]

        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# Collate function (outside class)
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


# Create datasets
train_size = int(0.9 * len(df))
train_df, val_df = df[:train_size], df[train_size:]

train_dataset = TranslationDataset(train_df['fr'].tolist(), train_df['en'].tolist(),
                                   fr_word2idx, en_word2idx)
val_dataset = TranslationDataset(val_df['fr'].tolist(), val_df['en'].tolist(),
                                 fr_word2idx, en_word2idx)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=4, pin_memory=True)

print(f"\n📊 Training: {len(train_dataset):,} | Validation: {len(val_dataset):,}")


#Basic Model seq2seq

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers = 2, dropout = 0.3, pretrained_embeddings = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = 0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout = dropout, batch_first = True, bidirectional = True)
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
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers = 2, dropout = 0.3, pretrained_embeddings = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx = 0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        

        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout = dropout, batch_first = True)
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

    def forward(self, src, tgt, teacher_forcing_ratio = 0.5):
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

if __name__ == "__main__":
    EMB_DIM = 300
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    N_EPOCHS = 15
    LEARNING_RATE = 0.001

    print(f"\n🎯 Configuration:")
    print(f"   Pre-trained: {USE_PRETRAINED}")
    print(f"   Epochs: {N_EPOCHS}")
    
    # Experiment 1: Basic Seq2Seq
    print("\n\n" + "="*70)
    print("EXPERIMENT 1: Basic Seq2Seq (Baseline)")
    print("="*70)

    encoder = Encoder(len(fr_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                      fr_embedding_tensor if USE_PRETRAINED else None)
    decoder1 = Decoder(len(en_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                       en_embedding_tensor if USE_PRETRAINED else None)

    model1 = Seq2Seq(encoder, decoder1).to(device)
    
    # Training setup
    optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    print(f"\n📝 Model Summary:")
    print(f"   Total parameters: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model1.parameters() if p.requires_grad):,}")
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    best_val_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        model1.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}')
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model1(src, tgt)
            
            # Reshape for loss: [batch * seq_len, vocab_size]
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model1.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model1(src, tgt, teacher_forcing_ratio=0)  # No teacher forcing
                
                output = output[:, 1:].reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model1.state_dict(), 'best_model1.pt')
            print(f'✅ Saved best model (val_loss: {val_loss:.4f})')
    
    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    








    

