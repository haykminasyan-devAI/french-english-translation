"""
Interactive French-to-English Translator
Choose your model (1, 2, or 3) and translate sentences!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import math

# CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocabularies
processed_path = Path('./data/processed')
with open(processed_path / 'vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    fr_word2idx = vocab_data['fr_word2idx']
    en_idx2word = vocab_data['en_idx2word']
    fr_vocab = vocab_data['fr_vocab']
    en_vocab = vocab_data['en_vocab']

# Load embeddings
try:
    with open(processed_path / 'embeddings.pkl', 'rb') as f:
        embeddings_data = pickle.load(f)
        fr_embedding_tensor = torch.FloatTensor(embeddings_data['fr_embedding_matrix'])
        en_embedding_tensor = torch.FloatTensor(embeddings_data['en_embedding_matrix'])
except:
    fr_embedding_tensor = None
    en_embedding_tensor = None


# ============================================================================
# MODEL 1: Basic Seq2Seq
# ============================================================================

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


# ============================================================================
# MODEL 2: Seq2Seq with Attention
# ============================================================================

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



# ============================================================================
# MODEL 3: Transformer (simplified for loading)
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
        if pretrained_embeddings is not None and pretrained_embeddings.size(1) != d_model:
            self.emb_projection = nn.Linear(pretrained_embeddings.size(1), d_model, bias=False)
            temp_emb = nn.Embedding(vocab_size, pretrained_embeddings.size(1), padding_idx=0)
            temp_emb.weight.data.copy_(pretrained_embeddings)
            self.temp_emb = temp_emb
        elif pretrained_embeddings is not None:
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
        if pretrained_embeddings is not None and pretrained_embeddings.size(1) != d_model:
            self.emb_projection = nn.Linear(pretrained_embeddings.size(1), d_model, bias=False)
            temp_emb = nn.Embedding(vocab_size, pretrained_embeddings.size(1), padding_idx=0)
            temp_emb.weight.data.copy_(pretrained_embeddings)
            self.temp_emb = temp_emb
        elif pretrained_embeddings is not None:
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

def translate_model3(model, sentence):
    """Translate using Transformer"""
    model.eval()
    tokens = sentence.lower().split()
    src_indices = [fr_word2idx.get(w, 1) for w in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_output = model.encoder(src_tensor, src_mask)
        
        tgt_indices = [2]  # <sos>
        
        for _ in range(50):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            output, _ = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
            next_token = output[0, -1].argmax().item()
            
            if next_token == 3:  # <eos>
                break
            
            tgt_indices.append(next_token)
        
        return ' '.join([en_idx2word.get(idx, '<unk>') for idx in tgt_indices[1:]])

# ============================================================================
# LOAD ALL AVAILABLE MODELS
# ============================================================================

models = {}

print("Loading models...")

# Try to load Model 1
try:
    encoder1 = Encoder1(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
    decoder1 = Decoder1(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
    model1 = Seq2Seq1(encoder1, decoder1).to(device)
    model1.load_state_dict(torch.load('best_model1.pt', map_location=device, weights_only=False))
    model1.eval()
    models['1'] = ('Model 1: Basic Seq2Seq', model1, translate_model1)
    print("✅ Model 1 loaded (Basic Seq2Seq)")
except Exception as e:
    print(f"⚠️  Model 1 not available: {e}")

# Try to load Model 2
try:
    encoder2 = Encoder2(len(fr_vocab), 300, 256, 2, 0.3, fr_embedding_tensor)
    decoder2 = AttentionDecoder(len(en_vocab), 300, 256, 2, 0.3, en_embedding_tensor)
    model2 = Seq2SeqAttention(encoder2, decoder2).to(device)
    model2.load_state_dict(torch.load('best_model2.pt', map_location=device, weights_only=False))
    model2.eval()
    models['2'] = ('Model 2: + Bahdanau Attention', model2, translate_model2)
    print("✅ Model 2 loaded (with Attention)")
except Exception as e:
    print(f"⚠️  Model 2 not available: {e}")

# Try to load Model 3
try:
    encoder3 = TransformerEncoder(
        vocab_size=len(fr_vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=200,
        dropout=0.1,
        pretrained_embeddings=fr_embedding_tensor
    )
    decoder3 = TransformerDecoder(
        vocab_size=len(en_vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=200,
        dropout=0.1,
        pretrained_embeddings=en_embedding_tensor
    )
    model3 = Transformer(encoder3, decoder3, src_pad_idx=0, tgt_pad_idx=0).to(device)
    model3.load_state_dict(torch.load('best_model3.pt', map_location=device, weights_only=False))
    model3.eval()
    models['3'] = ('Model 3: Transformer (SOTA)', model3, translate_model3)
    print("✅ Model 3 loaded (Transformer)")
except Exception as e:
    print(f"⚠️  Model 3 not available: {e}")

# Check if any models loaded
if not models:
    print("\n❌ No models loaded! Make sure model files exist.")
    exit(1)

print()

# ============================================================================
# INTERACTIVE TRANSLATION
# ============================================================================

sample_sentences = [
    "où sont les étoiles ?",
    "qu'est-ce que la lumière ?",
    "pourquoi le ciel est-il bleu ?",
    "quand est-ce que le soleil se lève ?",
    "comment fonctionne la gravité ?",
    "quelle est la température du soleil ?",
    "où se trouve mars ?",
    "qu'est-ce qu'un trou noir ?",
]

print("="*80)
print("🇫🇷 → 🇬🇧  INTERACTIVE FRENCH-TO-ENGLISH TRANSLATOR")
print("="*80)

# Select model
print("\n📦 Available Models:")
for key, (name, _, _) in models.items():
    print(f"   {key}. {name}")

while True:
    available = ', '.join(sorted(models.keys()))
    model_choice = input(f"\nSelect model ({available}): ").strip()
    if model_choice in models:
        selected_name, selected_model, translate_func = models[model_choice]
        print(f"✅ Using: {selected_name}\n")
        break
    else:
        print(f"❌ Invalid choice. Please enter {available}.")

# Translation loop
print("="*80)
print(f"🤖 Translator ready!")
print("="*80)
print("\nCommands:")
print("  • Type French sentence to translate")
print("  • 'switch' - Change model")
print("  • 'examples' - Show sample sentences")
print("  • 'quit' or 'q' - Exit\n")

while True:
    try:
        french_input = input(f"[{selected_name.split(':')[0]}] FR: ").strip()
        
        if not french_input:
            continue
            
        if french_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir!")
            break
        
        if french_input.lower() == 'switch':
            print("\n📦 Available Models:")
            for key, (name, _, _) in sorted(models.items()):
                print(f"   {key}. {name}")
            available = ', '.join(sorted(models.keys()))
            model_choice = input(f"\nSelect model ({available}): ").strip()
            if model_choice in models:
                selected_name, selected_model, translate_func = models[model_choice]
                print(f"✅ Switched to: {selected_name}\n")
            else:
                print(f"❌ Invalid choice. Keeping current model.\n")
            continue
            
        if french_input.lower() == 'examples':
            print("\n📝 Sample sentences you can try:")
            for i, sent in enumerate(sample_sentences, 1):
                print(f"   {i}. {sent}")
            print()
            continue
        
        # Translate
        english = translate_func(selected_model, french_input)
        print(f"                   EN: {english}\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir!")
        break
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()