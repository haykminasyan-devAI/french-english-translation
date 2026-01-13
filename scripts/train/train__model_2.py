"""
Model 2: Seq2Seq with Bahdanau Attention

Reference:
- Bahdanau et al. (2015): "Neural Machine Translation by Jointly Learning to Align and Translate"
  https://arxiv.org/abs/1409.0473

Key improvement over Model 1:
- Attention mechanism allows decoder to "focus" on different parts of source sentence
- Solves the information bottleneck problem of basic seq2seq
"""

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
    def __init__(self, src_sentences, tgt_sentences, src_word2idx, tgt_word2idx, max_len=50):
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


# Create datasets with 80/10/10 split (train/val/test)
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
# test_df = df[train_size + val_size:]  # Reserved for final evaluation

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


# ============================================================================
# MODEL 2: Seq2Seq with Bahdanau Attention
# ============================================================================

class Encoder(nn.Module):
    """Same encoder as Model 1 - Bidirectional GRU"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, 
                         batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch, src_len, hidden_dim*2]
        # hidden: [n_layers*2, batch, hidden_dim]
        
        # Combine bidirectional hidden states
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)  # [n_layers, 2, batch, hidden_dim]
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)      # [n_layers, batch, hidden_dim*2]
        hidden = torch.tanh(self.fc(hidden))                          # [n_layers, batch, hidden_dim]
        
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    
    Computes attention weights between decoder hidden state and all encoder outputs
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Attention weights
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)  # [dec_hidden + enc_output]
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: [batch, hidden_dim] - current decoder hidden state
            encoder_outputs: [batch, src_len, hidden_dim*2] - all encoder outputs
            
        Returns:
            attention_weights: [batch, src_len] - attention distribution
            context: [batch, hidden_dim*2] - weighted sum of encoder outputs
        """
        batch_size, src_len, enc_dim = encoder_outputs.size()
        
        # Repeat hidden state for each source position
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden_dim]
        
        # Concatenate hidden state with each encoder output
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # energy: [batch, src_len, hidden_dim]
        
        # Calculate attention scores
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        
        # Normalize to get attention weights (probabilities)
        attention_weights = F.softmax(attention, dim=1)  # [batch, src_len]
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context: [batch, 1, hidden_dim*2]
        context = context.squeeze(1)  # [batch, hidden_dim*2]
        
        return attention_weights, context


class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau Attention
    
    At each time step:
    1. Compute attention over encoder outputs
    2. Combine context with current input
    3. Generate output
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.attention = BahdanauAttention(hidden_dim)
        
        # RNN input = embedding + context vector
        self.rnn = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, n_layers, 
                         dropout=dropout, batch_first=True)
        
        # Output layer combines RNN output, context, and embedding
        self.fc = nn.Linear(hidden_dim + hidden_dim * 2 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, encoder_outputs):
        """
        Args:
            tgt: [batch, 1] - current target token
            hidden: [n_layers, batch, hidden_dim] - previous hidden state
            encoder_outputs: [batch, src_len, hidden_dim*2] - all encoder outputs
            
        Returns:
            prediction: [batch, vocab_size] - output distribution
            hidden: [n_layers, batch, hidden_dim] - new hidden state
            attention_weights: [batch, src_len] - attention distribution
        """
        # Embed current input
        embedded = self.dropout(self.embedding(tgt))  # [batch, 1, emb_dim]
        
        # Get attention context using last layer's hidden state
        # hidden[-1]: [batch, hidden_dim]
        attention_weights, context = self.attention(hidden[-1], encoder_outputs)
        # context: [batch, hidden_dim*2]
        
        # Concatenate embedding with context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # rnn_input: [batch, 1, emb_dim + hidden_dim*2]
        
        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch, 1, hidden_dim]
        
        # Prepare for output layer
        output = output.squeeze(1)      # [batch, hidden_dim]
        embedded = embedded.squeeze(1)   # [batch, emb_dim]
        
        # Combine output, context, and embedding for prediction
        prediction = self.fc(torch.cat([output, context, embedded], dim=1))
        # prediction: [batch, vocab_size]
        
        return prediction, hidden, attention_weights


class Seq2SeqAttention(nn.Module):
    """
    Seq2Seq model with Bahdanau Attention
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            teacher_forcing_ratio: probability of using ground truth as next input
            
        Returns:
            outputs: [batch, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(device)
        
        # Encode source sentence
        encoder_outputs, hidden = self.encoder(src)
        # encoder_outputs: [batch, src_len, hidden_dim*2]
        # hidden: [n_layers, batch, hidden_dim]
        
        # First input to decoder is <sos> token
        input_token = tgt[:, 0].unsqueeze(1)  # [batch, 1]
        
        # Decode step by step
        for t in range(1, tgt_len):
            # Get prediction with attention
            output, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs)
            # output: [batch, vocab_size]
            
            # Store output
            outputs[:, t] = output
            
            # Decide next input (teacher forcing or prediction)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)  # [batch, 1]
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
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
    print(f"   Hidden dim: {HIDDEN_DIM}")
    
    # Experiment 2: Seq2Seq with Bahdanau Attention
    print("\n\n" + "="*70)
    print("EXPERIMENT 2: Seq2Seq with Bahdanau Attention")
    print("="*70)
    print("\n📖 Key Innovation:")
    print("   - Attention mechanism allows decoder to focus on relevant source words")
    print("   - Solves information bottleneck of fixed-length context vector")
    print("   - Expected improvement: Better handling of long sentences\n")

    encoder = Encoder(len(fr_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                      fr_embedding_tensor if USE_PRETRAINED else None)
    decoder2 = AttentionDecoder(len(en_vocab), EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                                en_embedding_tensor if USE_PRETRAINED else None)

    model2 = Seq2SeqAttention(encoder, decoder2).to(device)
    
    # Training setup
    optimizer = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    print(f"\n📝 Model Summary:")
    total_params = sum(p.numel() for p in model2.parameters())
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Additional params vs Model 1: +{total_params - 10234567:,} (attention)")
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    best_val_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        model2.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}')
        for src, tgt in progress_bar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model2(src, tgt)
            
            # Reshape for loss: [batch * seq_len, vocab_size]
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model2.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model2(src, tgt, teacher_forcing_ratio=0)  # No teacher forcing
                
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
            torch.save(model2.state_dict(), 'best_model2.pt')
            print(f'✅ Saved best model (val_loss: {val_loss:.4f})')
    
    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"\n💡 Model saved to: best_model2.pt")
