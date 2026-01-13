"""
Visualize Training Results - Loss Curves and Comparisons
Creates plots comparing all 3 models
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def parse_log_file(log_file):
    """Extract training metrics from log file"""
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'Epoch (\d+): Train Loss = ([\d.]+) \| Val Loss = ([\d.]+)', line)
            if match:
                epochs.append({
                    'epoch': int(match.group(1)),
                    'train_loss': float(match.group(2)),
                    'val_loss': float(match.group(3))
                })
    
    return pd.DataFrame(epochs)

# Parse log files
print("📊 Parsing training logs...")

models_data = {}

if Path('logs/model1_1307032.log').exists():
    models_data['Model 1: Basic Seq2Seq'] = parse_log_file('logs/model1_1307032.log')
    print(f"✅ Model 1: {len(models_data['Model 1: Basic Seq2Seq'])} epochs")

if Path('logs/model2_1307036.log').exists():
    models_data['Model 2: + Attention'] = parse_log_file('logs/model2_1307036.log')
    print(f"✅ Model 2: {len(models_data['Model 2: + Attention'])} epochs")

if Path('logs/model3_1307054.log').exists():
    models_data['Model 3: Transformer'] = parse_log_file('logs/model3_1307054.log')
    print(f"✅ Model 3: {len(models_data['Model 3: Transformer'])} epochs")

if not models_data:
    print("❌ No log files found!")
    exit(1)

print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Neural Machine Translation: Model Comparison', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
ax1 = axes[0, 0]
for model_name, data in models_data.items():
    ax1.plot(data['epoch'], data['train_loss'], marker='o', label=model_name, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Loss
ax2 = axes[0, 1]
for model_name, data in models_data.items():
    ax2.plot(data['epoch'], data['val_loss'], marker='s', label=model_name, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Loss Convergence (log scale)
ax3 = axes[1, 0]
for model_name, data in models_data.items():
    ax3.plot(data['epoch'], data['train_loss'], marker='o', label=f'{model_name} (Train)', linewidth=2)
    ax3.plot(data['epoch'], data['val_loss'], marker='s', linestyle='--', label=f'{model_name} (Val)', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (log scale)')
ax3.set_title('Training vs Validation Loss')
ax3.set_yscale('log')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Best Performance Comparison
ax4 = axes[1, 1]

model_names = []
best_val_losses = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model_name, data) in enumerate(models_data.items()):
    short_name = model_name.split(':')[0]
    model_names.append(short_name)
    best_val_losses.append(data['val_loss'].min())

bars = ax4.bar(model_names, best_val_losses, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Best Validation Loss')
ax4.set_title('Best Performance Comparison')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, best_val_losses):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: training_comparison.png")

# Create individual plots for each model
for model_name, data in models_data.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data['epoch'], data['train_loss'], marker='o', label='Training Loss', linewidth=2, markersize=8)
    ax.plot(data['epoch'], data['val_loss'], marker='s', label='Validation Loss', linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} - Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotate best epoch
    best_epoch = data['val_loss'].idxmin()
    best_val = data.loc[best_epoch, 'val_loss']
    ax.annotate(f'Best: {best_val:.2f}', 
                xy=(data.loc[best_epoch, 'epoch'], best_val),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    safe_name = model_name.replace(':', '').replace(' ', '_')
    plt.savefig(f'{safe_name}_training.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {safe_name}_training.png")

plt.show()

print("\n" + "="*70)
print("✅ All visualizations created!")
print("="*70)
print("\nGenerated files:")
print("  - training_comparison.png (all models)")
print("  - Model_1_Basic_Seq2Seq_training.png")
print("  - Model_2_+_Bahdanau_Attention_training.png")
print("  - Model_3_Transformer_training.png")
