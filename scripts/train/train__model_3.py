"""
Model 3: Transformer for Neural Machine Translation

Reference:
- Vaswani et al. (2017): "Attention Is All You Need"
  https://arxiv.org/abs/1706.03762

Key improvements over Model 2:
- Self-attention mechanism (no recurrence!)
- Parallel computation (much faster training)
- Multi-head attention (attends to different aspects)
- Positional encoding (captures word order without RNN)
- State-of-the-art performance on translation tasks
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
c = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=4, pin_memory=True)

print(f"\n📊 Training: {len(train_dataset):,} | Validation: {len(val_dataset):,}")


# ============================================================================
# MODEL 3: Transformer Architecture
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional Encoding from "Attention is All You Need"
    
    Adds position information to embeddings using sine/cosine functions.
    This tells the model the order of words (since there's no recurrence).
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism
    
    Allows model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, q_len, d_model]
            key: [batch, k_len, d_model]
            value: [batch, v_len, d_model]
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            output: [batch, q_len, d_model]
            attention: [batch, n_heads, q_len, k_len]
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch, n_heads, q_len, k_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        # output: [batch, n_heads, q_len, d_k]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # output: [batch, q_len, d_model]
        
        # Final linear projection
        output = self.fc(output)
        
        return output, attention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Applied to each position separately and identically.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # [batch, seq_len, d_model] -> [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer Encoder
    
    Contains:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization and residual connections
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)  # Residual connection
        src = self.norm1(src)  # Layer norm
        
        # Feed-forward
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)  # Residual connection
        src = self.norm2(src)  # Layer norm
        
        return src


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer Decoder
    
    Contains:
    1. Masked multi-head self-attention (prevents looking ahead)
    2. Multi-head cross-attention (attends to encoder output)
    3. Feed-forward network
    4. Layer normalization and residual connections
    """
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
        # Masked self-attention (can't look ahead in target)
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)
        
        # Cross-attention (attend to encoder output)
        attn_output, attention = self.cross_attn(tgt, enc_output, enc_output, src_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)
        
        return tgt, attention


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder: Stack of N encoder layers
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            # Resize or interpolate if dimensions don't match
            if pretrained_embeddings.size(1) != d_model:
                print(f"⚠️  Embedding dim mismatch: {pretrained_embeddings.size(1)} -> {d_model}")
                print(f"   Using linear projection...")
                self.emb_projection = nn.Linear(pretrained_embeddings.size(1), d_model, bias=False)
                temp_emb = nn.Embedding(vocab_size, pretrained_embeddings.size(1), padding_idx=0)
                temp_emb.weight.data.copy_(pretrained_embeddings)
                # We'll project in forward pass
                self.temp_emb = temp_emb
            else:
                self.embedding.weight.data.copy_(pretrained_embeddings)
                self.temp_emb = None
                self.emb_projection = None
        else:
            self.temp_emb = None
            self.emb_projection = None
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch, src_len]
            src_mask: [batch, 1, 1, src_len]
            
        Returns:
            output: [batch, src_len, d_model]
        """
        # Embedding and scaling
        if self.temp_emb is not None:
            x = self.temp_emb(src)
            x = self.emb_projection(x)
        else:
            x = self.embedding(src)
        
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: Stack of N decoder layers
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pretrained_embeddings=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != d_model:
                print(f"⚠️  Embedding dim mismatch: {pretrained_embeddings.size(1)} -> {d_model}")
                print(f"   Using linear projection...")
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
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """
        Args:
            tgt: [batch, tgt_len]
            enc_output: [batch, src_len, d_model]
            tgt_mask: [batch, 1, tgt_len, tgt_len]
            src_mask: [batch, 1, 1, src_len]
            
        Returns:
            output: [batch, tgt_len, vocab_size]
        """
        # Embedding and scaling
        if self.temp_emb is not None:
            x = self.temp_emb(tgt)
            x = self.emb_projection(x)
        else:
            x = self.embedding(tgt)
        
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, enc_output, tgt_mask, src_mask)
            attentions.append(attention)
        
        # Project to vocabulary
        output = self.fc_out(x)
        
        return output, attentions


class Transformer(nn.Module):
    """
    Complete Transformer Model for Translation
    """
    def __init__(self, encoder, decoder, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
    
    def make_src_mask(self, src):
        """Create mask for source padding"""
        # src: [batch, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, src_len]
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """Create mask for target (padding + future positions)"""
        # tgt: [batch, tgt_len]
        batch_size, tgt_len = tgt.size()
        
        # Padding mask
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch, 1, 1, tgt_len]
        
        # Future mask (lower triangular matrix)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        # [tgt_len, tgt_len]
        
        # Combine masks
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # [batch, 1, tgt_len, tgt_len]
        
        return tgt_mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: [batch, src_len]
            tgt: [batch, tgt_len]
            
        Returns:
            output: [batch, tgt_len, vocab_size]
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encode source
        enc_output = self.encoder(src, src_mask)
        
        # Decode to target
        output, attentions = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        return output


def generate_translation(model, src, src_word2idx, tgt_idx2word, max_len=50):
    """
    Generate translation using greedy decoding
    
    Args:
        model: Trained Transformer
        src: Source sentence (string)
        src_word2idx: Source vocabulary
        tgt_idx2word: Target index-to-word mapping
        max_len: Maximum translation length
        
    Returns:
        translation: Translated sentence (string)
    """
    model.eval()
    
    # Tokenize and convert to indices
    src_tokens = src.lower().split()
    src_indices = [src_word2idx.get(w, 1) for w in src_tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    
    # Start with <sos> token
    tgt_indices = [2]  # <sos>
    
    with torch.no_grad():
        # Encode source
        src_mask = model.make_src_mask(src_tensor)
        enc_output = model.encoder(src_tensor, src_mask)
        
        # Decode step by step
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            # Get predictions
            output, _ = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
            
            # Get next token
            next_token = output[0, -1].argmax().item()
            
            # Stop if <eos>
            if next_token == 3:
                break
            
            tgt_indices.append(next_token)
        
    # Convert indices to words
    translation = ' '.join([tgt_idx2word.get(idx, '<unk>') for idx in tgt_indices[1:]])
    
    return translation


if __name__ == "__main__":
    # Hyperparameters
    D_MODEL = 512        # Model dimension
    N_HEADS = 8          # Number of attention heads
    N_LAYERS = 6         # Number of encoder/decoder layers
    D_FF = 2048          # Feed-forward dimension
    DROPOUT = 0.1        # Dropout rate
    MAX_LEN = 200        # Maximum sequence length (increased to handle long sentences)
    
    N_EPOCHS = 15
    LEARNING_RATE = 0.0001  # Lower learning rate for Transformer
    WARMUP_STEPS = 4000     # Learning rate warmup

    print(f"\n🎯 Configuration:")
    print(f"   Pre-trained: {USE_PRETRAINED}")
    print(f"   Epochs: {N_EPOCHS}")
    print(f"   Model dim: {D_MODEL}")
    print(f"   Attention heads: {N_HEADS}")
    print(f"   Layers: {N_LAYERS}")
    
    # Experiment 3: Transformer
    print("\n\n" + "="*70)
    print("EXPERIMENT 3: Transformer (State-of-the-Art)")
    print("="*70)
    print("\n📖 Key Innovations:")
    print("   - Self-attention: Captures long-range dependencies")
    print("   - Multi-head attention: Attends to different aspects")
    print("   - Parallel computation: Much faster than RNN")
    print("   - Positional encoding: Captures word order without recurrence")
    print("   - Expected improvement: 20-30% better than RNN models\n")

    # Create encoder and decoder
    encoder = TransformerEncoder(
        vocab_size=len(fr_vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pretrained_embeddings=fr_embedding_tensor if USE_PRETRAINED else None
    )
    
    decoder = TransformerDecoder(
        vocab_size=len(en_vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        pretrained_embeddings=en_embedding_tensor if USE_PRETRAINED else None
    )
    
    model3 = Transformer(encoder, decoder, src_pad_idx=0, tgt_pad_idx=0).to(device)
    
    # Training setup
    # Use Adam with learning rate warmup (standard for Transformers)
    optimizer = optim.Adam(model3.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    
    # Learning rate scheduler with warmup (from "Attention Is All You Need" paper)
    # Formula: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    def get_lr(step):
        """
        Learning rate schedule from Vaswani et al. (2017)
        - Linearly increases during warmup
        - Then decays proportional to inverse square root of step number
        """
        step = max(step, 1)  # Avoid division by zero
        d_model = D_MODEL
        warmup = WARMUP_STEPS
        
        arg1 = step ** (-0.5)
        arg2 = step * (warmup ** (-1.5))
        
        return (d_model ** (-0.5)) * min(arg1, arg2)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    
    # Verify learning rate schedule
    print(f"\n📈 Learning Rate Schedule:")
    print(f"   Step 1:    {get_lr(1):.6f}")
    print(f"   Step 100:  {get_lr(100):.6f}")
    print(f"   Step 1000: {get_lr(1000):.6f}")
    print(f"   Step 4000: {get_lr(4000):.6f} (peak)")
    print(f"   Step 8000: {get_lr(8000):.6f}\n")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # Label smoothing for better generalization
    
    print(f"\n📝 Model Summary:")
    total_params = sum(p.numel() for p in model3.parameters())
    trainable_params = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    best_val_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        model3.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}')
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass (tgt input excludes last token)
            tgt_input = tgt[:, :-1]
            
            optimizer.zero_grad()
            output = model3(src, tgt_input)
            
            # Reshape for loss computation
            # output: [batch, tgt_len-1, vocab_size]
            # tgt: [batch, tgt_len], we want to predict from position 1 onwards
            output = output.reshape(-1, output.shape[-1])  # [batch * (tgt_len-1), vocab_size]
            tgt_output = tgt[:, 1:].reshape(-1)  # [batch * (tgt_len-1)]
            
            loss = criterion(output, tgt_output)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model3.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            train_loss += loss.item()
            
            # Show current learning rate in progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Validation
        model3.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                
                output = model3(src, tgt_input)
                
                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output, tgt_output)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | LR = {current_lr:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model3.state_dict(), 'best_model3.pt')
            print(f'✅ Saved best model (val_loss: {val_loss:.4f})')
        
        # Test translation on sample
        if (epoch + 1) % 5 == 0:
            print("\n🔍 Sample translations:")
            test_sentences = [
                "où sont les étoiles ?",
                "qu'est-ce que la lumière ?",
                "quand est-ce que le soleil se lève ?"
            ]
            for test_src in test_sentences:
                translation = generate_translation(model3, test_src, fr_word2idx, en_idx2word)
                print(f"   FR: {test_src}")
                print(f"   EN: {translation}\n")
    
    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"\n💡 Model saved to: best_model3.pt")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Load best model
    model3.load_state_dict(torch.load('best_model3.pt'))
    model3.eval()
    
    print("\n🔍 Translation examples:")
    test_examples = [
        "où sont les étoiles ?",
        "qu'est-ce que la lumière ?",
        "pourquoi le ciel est-il bleu ?",
        "quand est-ce que le soleil se lève ?",
        "comment fonctionne la gravité ?",
        "quelle est la distance entre la terre et la lune ?",
        "où se trouve mars ?",
        "qu'est-ce qu'un trou noir ?"
    ]
    
    for test_src in test_examples:
        translation = generate_translation(model3, test_src, fr_word2idx, en_idx2word)
        print(f"\nFR: {test_src}")
        print(f"EN: {translation}")
