# French-to-English Neural Machine Translation

A comprehensive implementation of three generations of neural machine translation architectures, from basic RNNs to state-of-the-art Transformers.

## 🎯 Project Overview

This project implements and compares three progressively advanced neural machine translation models trained on the Giga-Fren French-English parallel corpus.

## 📊 Results

| Model | Architecture | BLEU Score | Val Loss | Parameters |
|-------|--------------|------------|----------|------------|
| Model 1 | Bidirectional GRU Seq2Seq | 27.69 | 5.24 | 11.6M |
| Model 2 | + Bahdanau Attention | 27.35 | 4.74 | 20.3M |
| Model 3 | Transformer (Multi-head) | **42.46** ⭐ | **4.03** ⭐ | 65.8M |

**Best Model**: Model 3 achieves 42.46 BLEU score with 53% improvement over baseline.

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.4+ with CUDA support
Virtual environment (venv)
```

### Installation

```bash
# Clone/navigate to project
cd translation

# Create virtual environment
python3 -m virtualenv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy matplotlib tqdm jupyter
```

### Train Models

```bash
# On SLURM cluster with GPU:
sbatch scripts/slurm/run_training_model1.sh
sbatch scripts/slurm/run_training_model2.sh
sbatch scripts/slurm/run_training_model3.sh

# On local machine:
python scripts/train/train__model_1.py
```

### Evaluate Models

```bash
# Evaluate specific model
python scripts/evaluate/evaluate_model3.py

# Compare all models
python scripts/evaluate/evaluate_all_models.py

# Interactive translator
python scripts/evaluate/interactive_translator.py
```

### Visualize Results

```bash
python scripts/evaluate/visualize_results.py
# Generates plots in visualizations/
```

## 📁 Project Structure

```
translation/
├── README.md                    # This file
├── data/
│   ├── raw/                     # Original Giga-Fren corpus (22M sentences)
│   ├── processed/               # Filtered questions (52K pairs)
│   │   ├── questions.csv
│   │   ├── vocab.pkl
│   │   └── embeddings.pkl
│   ├── giga-fren.tgz           # Downloaded dataset
│   └── questions.csv           # Processed data
├── models/                      # Trained model checkpoints
│   ├── model1/
│   │   ├── best_model1.pt
│   │   └── model1_test_results.pkl
│   ├── model2/
│   │   └── best_model2.pt
│   └── model3/
│       └── best_model3.pt
├── scripts/
│   ├── train/                   # Training scripts
│   │   ├── train__model_1.py
│   │   ├── train__model_2.py
│   │   └── train__model_3.py
│   ├── evaluate/                # Evaluation scripts
│   │   ├── evaluate_model1.py
│   │   ├── evaluate_model2.py
│   │   ├── evaluate_model3.py
│   │   ├── evaluate_all_models.py
│   │   ├── evaluate_with_metrics.py
│   │   ├── interactive_translator.py
│   │   ├── show_translations.py
│   │   ├── compare_all_translations.py
│   │   └── visualize_results.py
│   └── slurm/                   # SLURM job submission scripts
│       ├── run_training_model1.sh
│       ├── run_training_model2.sh
│       └── run_training_model3.sh
├── logs/                        # Training logs
│   ├── model1_*.log
│   ├── model2_*.log
│   └── model3_*.log
├── visualizations/              # Generated plots
│   ├── training_comparison.png
│   ├── Model_1_Basic_Seq2Seq_training.png
│   ├── Model_2_+_Attention_training.png
│   └── Model_3_Transformer_training.png
├── notebooks/                   # Jupyter notebooks
│   └── fr2eng.ipynb
├── docs/                        # Documentation
│   ├── MODELS_SUMMARY.md
│   ├── FINAL_RESULTS.md
│   └── PROJECT_SUMMARY.md
├── cc.fr.300.vec               # Pre-trained embeddings
├── cc.en.300.vec
└── venv/                       # Virtual environment
```

## 📖 Documentation

- **[MODELS_SUMMARY.md](docs/MODELS_SUMMARY.md)**: Technical specifications of all models
- **[FINAL_RESULTS.md](docs/FINAL_RESULTS.md)**: Detailed results and analysis
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)**: Complete project overview

## 🤖 Models

### Model 1: Basic Seq2Seq
- **Architecture**: Bidirectional GRU encoder + GRU decoder
- **Reference**: Sutskever et al. (2014)
- **Training**: `scripts/train/train__model_1.py`

### Model 2: Seq2Seq with Attention
- **Architecture**: GRU + Bahdanau additive attention
- **Reference**: Bahdanau et al. (2015)
- **Training**: `scripts/train/train__model_2.py`

### Model 3: Transformer
- **Architecture**: Multi-head self-attention (6 layers, 8 heads)
- **Reference**: Vaswani et al. (2017) "Attention Is All You Need"
- **Training**: `scripts/train/train__model_3.py`

## 📚 References

1. Bahdanau et al. (2015): https://arxiv.org/abs/1409.0473
2. Vaswani et al. (2017): https://arxiv.org/abs/1706.03762
3. FastText embeddings: https://fasttext.cc/

## 🎓 Author

Hayk Minasyan  
January 2026

## 📝 License

Educational/Research purposes
