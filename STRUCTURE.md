# Project Structure

## Overview

Professional, production-ready organization for a neural machine translation project.

## Directory Structure

```
translation/
├── README.md                       # Project overview and quick start
├── requirements.txt                # Python dependencies
├── best_model*.pt                  # Symlinks to trained models
│
├── data/                           # All datasets
│   ├── raw/                        # Original corpus
│   │   └── giga-fren/             # 22M sentence pairs
│   ├── processed/                  # Preprocessed data
│   │   ├── questions.csv          # 52K filtered questions
│   │   ├── vocab.pkl              # Vocabularies (word2idx, idx2word)
│   │   └── embeddings.pkl         # Pre-trained fastText embeddings
│   └── giga-fren.tgz              # Downloaded archive
│
├── models/                         # Trained model checkpoints
│   ├── model1/
│   │   ├── best_model1.pt         # Best Model 1 checkpoint
│   │   └── model1_test_results.pkl
│   ├── model2/
│   │   └── best_model2.pt         # Best Model 2 checkpoint
│   └── model3/
│       └── best_model3.pt         # Best Model 3 checkpoint
│
├── scripts/                        # All executable scripts
│   ├── train/                      # Training scripts
│   │   ├── train__model_1.py      # Model 1: Basic Seq2Seq
│   │   ├── train__model_2.py      # Model 2: + Bahdanau Attention
│   │   └── train__model_3.py      # Model 3: Transformer
│   ├── evaluate/                   # Evaluation & utilities
│   │   ├── evaluate_model1.py     # Model-specific evaluation
│   │   ├── evaluate_model2.py
│   │   ├── evaluate_model3.py
│   │   ├── evaluate_all_models.py # Compare all models
│   │   ├── evaluate_with_metrics.py  # Detailed metrics
│   │   ├── interactive_translator.py # Interactive CLI tool
│   │   ├── show_translations.py   # Quick translation viewer
│   │   ├── compare_all_translations.py
│   │   └── visualize_results.py   # Plot training curves
│   └── slurm/                      # HPC job scripts
│       ├── run_training_model1.sh
│       ├── run_training_model2.sh
│       └── run_training_model3.sh
│
├── logs/                           # Training logs
│   ├── model1_*.log
│   ├── model2_*.log
│   └── model3_*.log
│
├── visualizations/                 # Generated plots
│   ├── training_comparison.png    # 4-panel comparison
│   ├── Model_1_Basic_Seq2Seq_training.png
│   ├── Model_2_+_Attention_training.png
│   └── Model_3_Transformer_training.png
│
├── notebooks/                      # Jupyter notebooks
│   └── fr2eng.ipynb               # Data exploration & preprocessing
│
├── docs/                           # Documentation
│   ├── MODELS_SUMMARY.md          # Technical model details
│   ├── FINAL_RESULTS.md           # Results analysis
│   └── PROJECT_SUMMARY.md         # Complete project overview
│
├── cc.fr.300.vec                   # Pre-trained French embeddings
├── cc.en.300.vec                   # Pre-trained English embeddings
└── venv/                           # Virtual environment
```

## 🎯 Design Principles

1. **Separation of Concerns**: Code, data, models, docs in separate directories
2. **Clear Naming**: Descriptive folder and file names
3. **Hierarchical**: Logical grouping (scripts → train/evaluate/slurm)
4. **Accessibility**: Symlinks for backward compatibility
5. **Scalability**: Easy to add new models or experiments

## 📦 Key Directories

### `/data`
- **raw/**: Original, unmodified datasets
- **processed/**: Cleaned, filtered, ready-to-use data

### `/models`
- **model1/, model2/, model3/**: Separate checkpoints per model
- Prevents accidental overwrites
- Clear versioning

### `/scripts`
- **train/**: All training code
- **evaluate/**: All evaluation and inference code
- **slurm/**: HPC-specific job submission scripts

### `/logs`
- Training logs from SLURM jobs
- Timestamped for tracking experiments

### `/visualizations`
- Generated plots and figures
- Separate from code for clarity

### `/notebooks`
- Jupyter notebooks for exploration
- Data preprocessing workflow

### `/docs`
- Technical documentation
- Results analysis
- Project summaries

## 🚀 Usage Patterns

### Training
```bash
cd scripts/train
python train__model_3.py
```

### Evaluation
```bash
cd scripts/evaluate
python evaluate_model3.py
```

### Interactive Use
```bash
python scripts/evaluate/interactive_translator.py
```

### Documentation
```bash
cat docs/FINAL_RESULTS.md
```

## ✅ Benefits of This Structure

- ✅ **Professional**: Industry-standard organization
- ✅ **Maintainable**: Easy to find and modify files
- ✅ **Scalable**: Simple to add Model 4, Model 5, etc.
- ✅ **Clear**: Anyone can understand the layout
- ✅ **Git-friendly**: Logical .gitignore boundaries
- ✅ **Reproducible**: Clear separation of artifacts

## 🔄 Migration Guide

Files were reorganized from flat structure to hierarchical:

- Training scripts: `./` → `scripts/train/`
- Evaluation scripts: `./` → `scripts/evaluate/`
- Model checkpoints: `./` → `models/model*/`
- Documentation: `./` → `docs/`
- Visualizations: `./` → `visualizations/`
- Notebooks: `./` → `notebooks/`

Symlinks maintain backward compatibility for model files.
