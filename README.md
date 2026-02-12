# Temporal Graph Attention Networks for Cryptocurrency Fraud Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Abstract

This research presents a **Temporal Graph Attention Network (TemporalGAT)** for detecting fraudulent transactions across multiple blockchain platforms. The model combines multi-head graph attention mechanisms with Gated Recurrent Units (GRU) to capture both structural and temporal patterns in transaction graphs.

### Key Results

| Blockchain | Dataset | PR-AUC | ROC-AUC | Recall | Precision |
|------------|---------|--------|---------|--------|-----------|
| **Bitcoin** | Elliptic | 0.476 | 0.904 | 92.9% | 35.1% |
| **Ethereum** | XBlock Phishing | 0.560 | 0.947 | 96.8% | 35.1% |

**Highlights:**
- ✅ **92.9% recall on Bitcoin** - missed only 4 of 56 fraudulent transactions
- ✅ **96.8% recall on Ethereum** - state-of-the-art phishing detection
- ✅ **No SMOTE required** - handles extreme class imbalance (0.04%-2%) through Focal Loss
- ✅ **Cross-chain generalization** - same architecture works on both blockchains
- ✅ **Zero degradation transfer learning** - validated on Elliptic++ dataset

---

## 🗂️ Project Structure

```
thesis/
├── config/                     # Training configurations
│   └── config.yaml
├── data/                       # Dataset loaders
│   ├── elliptic_loader.py      # Bitcoin Elliptic dataset
│   └── ethereum_phishing_loader.py  # Ethereum XBlock dataset
├── models/                     # Neural network architectures
│   └── temporal_gnn.py         # TemporalGAT model
├── losses/                     # Loss functions
│   └── imbalanced_losses.py    # Focal Loss with class weighting
├── training/                   # Training loops
│   └── trainer.py
├── evaluation/                 # Metrics
│   └── metrics.py
├── explainability/             # XAI modules
│   └── gnn_explainer.py
├── outputs/                    # Experiment results
│   ├── improved_temporal_gat_pw25.0/  # Best Bitcoin model
│   ├── ethereum_phishing/      # Ethereum results
│   ├── transfer_test/          # Transfer learning
│   ├── ablation_study/         # Ablation experiments
│   └── visualizations/         # Figures for thesis
├── thesis/                     # LaTeX thesis document
├── main.py                     # Main training script
├── train_enhanced.py           # Enhanced training with features
├── train_ethereum_vertex.py    # Ethereum training (Vertex AI)
├── transfer_test.py            # Transfer learning evaluation
├── ablation_study.py           # Ablation experiments
├── baseline_comparison.py      # Baseline comparisons
├── multi_seed_analysis.py      # Stability analysis
├── crosschain_comparison.py    # Bitcoin vs Ethereum comparison
├── explainability.py           # XAI analysis
├── generate_visualizations.py  # Plot generation
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

> **⚠️ Note:** Data files, model checkpoints (.pt), and outputs are excluded from this repository due to GitHub file size limits. You'll need to download the datasets separately.

**Bitcoin Elliptic Dataset:**
```bash
# Auto-downloads from Kaggle
python main.py --download-data
```

**Ethereum XBlock Dataset:**
- Download `MulDiGraph.pkl` from [XBlock-ETH Repository](https://github.com/xblock-eth)
- Place in `data/EthereumPhishingTransactionNetwork/`

**Elliptic++ Dataset** (for transfer learning):
- Download from [Elliptic++ GitHub](https://github.com/Elliptic-Coin/Elliptic-Plus-Plus)
- Extract to `data/Elliptic++ Dataset/raw/`

---

## 📊 Reproducing Research Results

### Step 1: Train Bitcoin Model (Elliptic Dataset)

```bash
python train_enhanced.py \
    --model temporal_gat \
    --pos-weight 25 \
    --focal-alpha 0.92 \
    --focal-gamma 3.5 \
    --hidden-channels "256,128,64" \
    --epochs 100 \
    --output-dir outputs/bitcoin_final
```

**Expected Results:**
- PR-AUC: ~0.476
- ROC-AUC: ~0.904
- Recall: ~92.9% (at threshold 0.614)

### Step 2: Multi-Seed Stability Analysis

```bash
python multi_seed_analysis.py --num-seeds 5 --output-dir outputs/multi_seed_5runs
```

**Output:** Statistical analysis of model stability across random seeds.

### Step 3: Ablation Study

```bash
python ablation_study.py --output-dir outputs/ablation_study
```

**Ablations tested:**
- Without enhanced features
- GConvGRU instead of GAT
- Without Focal Loss
- Without class weighting
- Without threshold optimization

### Step 4: Baseline Comparison

```bash
python baseline_comparison.py --output-dir outputs/baseline_comparison
```

**Baselines:**
- Logistic Regression
- Random Forest
- Multi-Layer Perceptron

### Step 5: Transfer Learning (Elliptic++)

```bash
python transfer_test.py \
    --checkpoint outputs/bitcoin_final/checkpoints/best_model.pt \
    --output-dir outputs/transfer_test
```

**Expected:** Zero performance degradation (PR-AUC 0.476 → 0.482)

### Step 6: Train Ethereum Model (XBlock)

```bash
python train_ethereum_vertex.py \
    --sample-ratio 0.05 \
    --max-nodes-per-snapshot 30000 \
    --num-timesteps 10 \
    --hidden-channels "128,64,32" \
    --pos-weight 50 \
    --focal-alpha 0.95 \
    --epochs 30 \
    --output-dir outputs/ethereum_phishing
```

**Expected Results:**
- PR-AUC: ~0.560
- ROC-AUC: ~0.947
- Recall: ~96.8%

### Step 7: Cross-Chain Comparison

```bash
python crosschain_comparison.py
```

**Output:** LaTeX table and markdown report comparing Bitcoin vs Ethereum results.

### Step 8: Generate Visualizations

```bash
python generate_visualizations.py --output-dir outputs/visualizations
```

**Generates:**
- PR/ROC curves
- Confusion matrix
- Threshold analysis
- t-SNE embeddings
- Feature importance

### Step 9: Explainability Analysis

```bash
python explainability.py \
    --checkpoint outputs/bitcoin_final/checkpoints/best_model.pt \
    --output-dir outputs/explainability
```

**Output:** Gradient-based feature importance and attention weight analysis.

---

## 📈 Key Findings

### 1. Model Performance Comparison

| Method | ROC-AUC | PR-AUC | Recall | SMOTE |
|--------|---------|--------|--------|-------|
| Ertam (2025) XGBoost-XAI | 0.96* | -- | -- | No |
| Maulana (2026) RF+SMOTE | 0.95* | -- | -- | Yes |
| ATGAT (2025) | 0.913 | ~0.42 | ~85% | No |
| FraudGNN-RL (2026) | 0.891 | 0.44 | 88.2% | No |
| **TemporalGAT (Ours)** | **0.904** | **0.476** | **92.9%** | **No** |

*Accuracy reported on balanced datasets

### 2. Ablation Results

| Configuration | PR-AUC | Recall |
|--------------|--------|--------|
| Full Model | **0.296** | **37.5%** |
| w/o Enhanced Features | 0.278 | 28.6% |
| GConvGRU (no attention) | 0.184 | 1.8% |
| w/o Focal Loss | 0.287 | 10.7% |
| w/o Class Weighting | 0.262 | 3.6% |

**Key Insight:** Attention mechanism provides +61% PR-AUC improvement over GConvGRU.

### 3. Cross-Chain Generalization

| Metric | Bitcoin | Ethereum | Δ |
|--------|---------|----------|---|
| PR-AUC | 0.476 | 0.560 | +17.6% |
| ROC-AUC | 0.904 | 0.947 | +4.8% |
| Recall | 92.9% | 96.8% | +4.2% |

**Key Insight:** Same architecture generalizes across blockchain platforms.

### 4. Feature Enrichment Experiment

We attempted to enrich the Ethereum graph with raw transaction data from XBlock (33GB of internal transactions, blocks 23M-24.25M):

| Metric | Value |
|--------|-------|
| Nodes enriched | 30,718 (1.03%) |
| Phishing nodes enriched | 4 (0.34%) |

**Key Finding:** Temporal mismatch - phishing addresses were active in different block ranges than the available raw transaction data.

---

## 🔬 Model Architecture

```
Input: Temporal Graph Sequence [G₁, G₂, ..., Gₜ]
       Each Gₜ = (Nodes, Edges, Features)
                    ↓
┌─────────────────────────────────────────────────┐
│  Input Projection (Linear: 166 → 256)           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  GAT-GRU Block 1                                │
│  ├─ Graph Attention (256 → 256, 4 heads)        │
│  └─ GRU Cell (temporal memory)                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  GAT-GRU Block 2                                │
│  ├─ Graph Attention (256 → 128, 4 heads)        │
│  └─ GRU Cell                                    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  GAT-GRU Block 3                                │
│  ├─ Graph Attention (128 → 64, 4 heads)         │
│  └─ GRU Cell                                    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Classification Head                            │
│  ├─ Linear (64 → 32)                            │
│  ├─ ReLU + Dropout (0.4)                        │
│  └─ Linear (32 → 1) → Sigmoid                   │
└─────────────────────────────────────────────────┘
                    ↓
           Output: P(fraud) ∈ [0, 1]
```

### Loss Function

**Focal Loss with Class Weighting:**
```
L = -α(1 - p)^γ · log(p)
```
- α = 0.92 (Bitcoin) / 0.95 (Ethereum)
- γ = 3.5 (focus on hard examples)
- pos_weight = 25 (Bitcoin) / 50 (Ethereum)

---

## 📚 Datasets

### Bitcoin: Elliptic Dataset
- **Nodes:** 203,769 transactions
- **Features:** 166 per transaction (94 local + 72 aggregated)
- **Labels:** 4,545 illicit (2%), 42,019 licit
- **Temporal:** 49 timesteps
- **Source:** [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set)

### Bitcoin: Elliptic++ Dataset
- Extended version with 182 features
- Used for transfer learning validation
- **Source:** [GitHub](https://github.com/git-disl/EllipticPlusPlus)

### Ethereum: XBlock Phishing Transaction Network
- **Nodes:** 2,973,489 addresses
- **Edges:** 13,551,303 transactions
- **Labels:** 1,165 phishing addresses (0.04%)
- **Source:** [XBlock-ETH](https://xblock.pro/xblock-eth/)

---

## 📊 Output Files

After running all experiments, you'll have:

```
outputs/
├── improved_temporal_gat_pw25.0/
│   ├── checkpoints/best_model.pt    # Trained model
│   ├── results.json                 # Metrics
│   └── plots/                       # Training curves
├── ethereum_phishing/
│   ├── results.json                 # Ethereum metrics
│   └── best_model.pt
├── transfer_test/
│   └── transfer_results.json        # Transfer learning metrics
├── ablation_study/
│   └── ablation_results.json        # Ablation comparison
├── baseline_comparison/
│   └── baseline_results.json        # Baseline metrics
├── comparison/
│   ├── crosschain_table.tex         # LaTeX table
│   └── crosschain_comparison.md     # Markdown report
└── visualizations/
    ├── pr_roc_curves.png
    ├── confusion_matrix.png
    ├── threshold_analysis.png
    ├── feature_importance.png
    └── tsne_embeddings.png
```

---

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 (6GB) | RTX 3080 (10GB+) |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |

**Note:** Ethereum training with memory-optimized settings (hidden_channels=[128,64,32]) works on RTX 3050 Ti (4GB).

---

## 📝 Citation

```bibtex
@thesis{alkuraimi2026,
  title={Temporal Graph Neural Networks for Cryptocurrency Fraud Detection},
  author={Ahmed Ameen Hassan Alkuraimi},
  year={2026},
  school={State University of Malang},
  department={Department of Electrical and Informatics Engineering}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Elliptic for providing the Bitcoin transaction dataset
- XBlock team for the Ethereum phishing dataset
- PyTorch Geometric team for the excellent GNN library
- Google Cloud for Vertex AI training resources
