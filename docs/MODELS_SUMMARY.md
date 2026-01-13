# French-to-English Translation Models

This project implements three progressively advanced neural machine translation models, trained on the giga-fren dataset (52,331 French-English question pairs).

---

## 📊 Dataset

- **Source**: [giga-fren corpus](https://s3.amazonaws.com/fast-ai-nlp/giga-fren.tgz)
- **Size**: 22 million parallel sentences (filtered to 52,331 questions)
- **Languages**: French (source) → English (target)
- **Vocabulary**: 
  - French: 10,004 unique words
  - English: 10,004 unique words
- **Pre-trained Embeddings**: fastText (300-dimensional, trained on Common Crawl)
  - French coverage: 61.2%
  - English coverage: 61.6%

---

## 🤖 Model Architectures

### Model 1: Basic Seq2Seq with Bidirectional GRU

**File**: `train__model_1.py`

**Architecture**:
- **Encoder**: 2-layer bidirectional GRU
- **Decoder**: 2-layer unidirectional GRU
- **Hidden dim**: 256
- **Embeddings**: 300-dimensional (pre-trained fastText)

**Key Features**:
- Bidirectional encoding (captures context from both directions)
- Fixed-length context vector (information bottleneck)
- Teacher forcing during training

**Parameters**: ~11.6 million

**Expected Performance**: Baseline

**Limitations**:
- Information bottleneck: entire source sentence compressed into single vector
- Struggles with long sentences
- No explicit attention to source words

---

### Model 2: Seq2Seq with Bahdanau Attention

**File**: `train__model_2.py`

**Architecture**:
- **Encoder**: 2-layer bidirectional GRU (same as Model 1)
- **Decoder**: 2-layer GRU with additive attention
- **Hidden dim**: 256
- **Embeddings**: 300-dimensional (pre-trained fastText)

**Key Features**:
- **Bahdanau (additive) attention mechanism**
- Decoder can "focus" on different source words at each step
- Solves information bottleneck problem
- Attention weights are interpretable (can visualize what the model focuses on)

**Reference**: 
> Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015*. 
> https://arxiv.org/abs/1409.0473

**Parameters**: ~12-13 million (slightly more due to attention layers)

**Expected Performance**: 10-15% improvement over Model 1

**Innovation**:
- First to introduce attention mechanism in NMT
- Alignment and translation learned jointly
- Better handling of long sentences

---

### Model 3: Transformer (State-of-the-Art)

**File**: `train__model_3.py`

**Architecture**:
- **Encoder**: 6-layer Transformer encoder
- **Decoder**: 6-layer Transformer decoder
- **Model dimension**: 512
- **Attention heads**: 8
- **Feed-forward dim**: 2048
- **Embeddings**: 300-dimensional (projected to 512)

**Key Features**:
- **Multi-head self-attention** (no recurrence!)
- **Parallel computation** (much faster training than RNNs)
- **Positional encoding** (captures word order)
- **Layer normalization** and **residual connections**
- **Label smoothing** for better generalization
- **Learning rate warmup** (4000 steps)

**Reference**:
> Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
> https://arxiv.org/abs/1706.03762

**Parameters**: ~25-30 million

**Expected Performance**: 20-30% improvement over Model 2

**Innovations**:
- Eliminates recurrence entirely (enables parallelization)
- Multi-head attention attends to different aspects simultaneously
- Self-attention captures long-range dependencies efficiently
- Current state-of-the-art for translation (basis for GPT, BERT, etc.)

---

## 🚀 Training

### Hardware
- **Cluster**: SLURM with scalar6000q partition
- **GPU**: NVIDIA RTX A6000 (48 GB)
- **Framework**: PyTorch 2.4.1 with CUDA 11.8

### Training Scripts

| Model | Script | GPU Script | Expected Time (15 epochs) |
|-------|--------|------------|---------------------------|
| Model 1 | `train__model_1.py` | `run_training_model1.sh` | ~12-15 minutes |
| Model 2 | `train__model_2.py` | `run_training_model2.sh` | ~15-20 minutes |
| Model 3 | `train__model_3.py` | `run_training_model3.sh` | ~20-25 minutes |

### To Submit Jobs:

```bash
# Model 1: Basic Seq2Seq
sbatch run_training_model1.sh

# Model 2: With Attention
sbatch run_training_model2.sh

# Model 3: Transformer
sbatch run_training_model3.sh
```

### Monitor Progress:

```bash
# Check job status
squeue -u hayk.minasyan

# View logs (real-time)
tail -f logs/model1_*.log
tail -f logs/model2_*.log
tail -f logs/model3_*.log
```

---

## 📈 Expected Results

### Performance Hierarchy:

```
Model 1 (Baseline)
    ↓ +10-15%
Model 2 (+ Attention)
    ↓ +20-30%
Model 3 (Transformer) ← Best
```

### Validation Loss (Expected):

- **Model 1**: ~4.5-5.0
- **Model 2**: ~4.0-4.5
- **Model 3**: ~3.0-3.5

### BLEU Score (Expected):

- **Model 1**: ~15-20
- **Model 2**: ~20-25
- **Model 3**: ~25-35

---

## 🔬 Technical Comparison

| Feature | Model 1 | Model 2 | Model 3 |
|---------|---------|---------|---------|
| **Architecture** | GRU | GRU + Attention | Transformer |
| **Recurrence** | Yes (sequential) | Yes (sequential) | No (parallel) |
| **Attention** | None | Additive (single) | Multi-head self-attention |
| **Context** | Fixed vector | Dynamic (attention) | Full self-attention |
| **Parallelization** | Limited | Limited | Full |
| **Training Speed** | Baseline | Similar | 2-3x faster |
| **Long Sequences** | Poor | Good | Excellent |
| **Interpretability** | Low | Medium (attention) | High (multi-head) |

---

## 📝 Files Structure

```
translation/
├── fr2eng.ipynb                    # Data preparation notebook
├── train__model_1.py               # Model 1: Basic Seq2Seq
├── train__model_2.py               # Model 2: + Bahdanau Attention
├── train__model_3.py               # Model 3: Transformer
├── run_training_model1.sh          # SLURM script for Model 1
├── run_training_model2.sh          # SLURM script for Model 2
├── run_training_model3.sh          # SLURM script for Model 3
├── data/
│   ├── raw/giga-fren/              # Raw parallel corpus
│   │   ├── giga-fren.release2.fixed.en
│   │   └── giga-fren.release2.fixed.fr
│   └── processed/                  # Processed data
│       ├── questions.csv           # Filtered questions
│       ├── vocab.pkl               # Vocabularies
│       └── embeddings.pkl          # Pre-trained embeddings
├── logs/                           # Training logs
├── cc.fr.300.vec                   # French embeddings (4.3 GB)
├── cc.en.300.vec                   # English embeddings (4.3 GB)
└── venv/                           # Virtual environment
```

---

## 🎓 Learning Objectives Demonstrated

1. ✅ **Data Processing**: Web-scraped data → filtered dataset
2. ✅ **Vocabulary Building**: Word tokenization and indexing
3. ✅ **Embeddings**: Pre-trained word vectors integration
4. ✅ **RNN Architectures**: Understanding sequential models
5. ✅ **Attention Mechanisms**: From fixed context to dynamic attention
6. ✅ **Transformer Architecture**: Modern state-of-the-art NLP
7. ✅ **GPU Computing**: SLURM cluster with CUDA
8. ✅ **PyTorch Best Practices**: DataLoaders, checkpointing, evaluation

---

## 📚 References

1. **Sutskever et al. (2014)**: "Sequence to Sequence Learning with Neural Networks"
   - https://arxiv.org/abs/1409.3215

2. **Bahdanau et al. (2015)**: "Neural Machine Translation by Jointly Learning to Align and Translate"
   - https://arxiv.org/abs/1409.0473

3. **Vaswani et al. (2017)**: "Attention Is All You Need"
   - https://arxiv.org/abs/1706.03762

4. **Bojanowski et al. (2017)**: "Enriching Word Vectors with Subword Information" (fastText)
   - https://arxiv.org/abs/1607.04606

---

## 🎯 Next Steps (After Training)

1. **Compare models**: Analyze validation loss curves
2. **BLEU evaluation**: Compute translation quality metrics
3. **Qualitative analysis**: Manual inspection of translations
4. **Attention visualization**: See what Model 2 and 3 focus on
5. **Error analysis**: Identify common failure patterns
6. **Hyperparameter tuning**: Experiment with different settings

---

**Author**: Hayk Minasyan  
**Date**: January 2026  
**Based on**: FastAI NLP Course - Seq2Seq Translation Notebook
