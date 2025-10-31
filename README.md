# ADWC-DFS Ensemble

**Adaptive Density-Weighted Cascade with Dynamic Feature Synthesis**

Ensemble voting system for fraud detection achieving **84-91% recall**.

---

## ⚡ Quick Start

```bash
# 1. Test ensemble (10% data, ~5 min)
python test_ensemble.py

# 2. Train production ensemble (100% data, ~30-60 min)
python ensemble_voting.py --n_models 5

# 3. Use in code
from ensemble_voting import VotingEnsemble
ensemble = VotingEnsemble.load('results/ensemble_model.pkl')
predictions = ensemble.predict(X_new, threshold=0.10)
```

---

## 📊 Performance

| Strategy | Recall | Detected Frauds | Use Case |
|----------|--------|-----------------|----------|
| Soft (0.13) | 83.9% | 1800/2145 | Balanced |
| Soft (0.10) | 85.8% | 1840/2145 | Production |
| Aggressive (2/5) | 88.3% | 1893/2145 | High-value |
| Aggressive (1/5) | 90.2% | 1934/2145 | Critical |
| **ULTRA** | **91.4%** | **1960/2145** | **Maximum detection** |

---

## 🎓 How It Works

### ADWC-DFS Base Algorithm
1. **Density Profiling**: Find easy/medium/hard samples
2. **Cascade Training**: 3 specialist models
3. **Feature Synthesis**: Meta-features from disagreement
4. **Meta-Classifier**: Adaptive weighting

### Ensemble Enhancement
- **5 models** with different configs (random seeds, scale_pos_weight)
- **Voting strategies**: Soft (average), Aggressive (min votes), ULTRA (any)
- **Weighted** by validation recall

---

## 📚 Documentation

- **START_HERE.md** - Quick start guide
- **ENSEMBLE_USAGE_GUIDE.md** - Detailed ensemble guide
- **HUONG_DAN_TIENG_VIET.md** - Vietnamese guide
- **ALGORITHM.md** - Mathematical details
- **PROJECT_SUMMARY.md** - Project overview

---

## 🚀 Installation

```bash
cd /home/shyn/Dev/ADWC-DFS-
uv sync
```

---

## 🎮 Main Scripts

- **test_ensemble.py** - Quick test with 10% data
- **ensemble_voting.py** - Full ensemble training
- **main.py** - Main entry point
- **train_ensemble.sh** - Background training script
- **monitor_training.sh** - Monitor training progress
- **view_logs.py** - View training logs

---

## 💻 Usage Examples

### Basic Usage
```python
from ensemble_voting import VotingEnsemble

# Train ensemble
ensemble = VotingEnsemble(n_models=5)
ensemble.fit(X_train, y_train)

# Predict
predictions = ensemble.predict(X_test, threshold=0.10)
```

### Advanced Strategies
```python
# Soft voting (balanced)
pred_soft = ensemble.predict(X_test, threshold=0.10)

# Aggressive (2/5 models must agree)
pred_agg = ensemble.predict_aggressive(X_test, min_votes=2)

# ULTRA aggressive (any model detects → fraud)
pred_ultra = ensemble.predict_ultra_aggressive(X_test)
```

### Evaluate Multiple Strategies
```python
results = ensemble.evaluate(X_test, y_test)
print(results)
```

---

## 🎯 Key Features

- ✅ **High Recall**: 84-91% fraud detection
- ✅ **No SMOTE**: No synthetic data needed
- ✅ **Multiple Strategies**: Choose based on use case
- ✅ **Fast Training**: ~30-60 min for 5 models
- ✅ **Production-Ready**: Save/load, batch inference

---

## 📁 Project Structure
```
ADWC-DFS/
├── adwc_dfs/                       # Main package
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── adwc_dfs.py            # Main ADWC-DFS model
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage1_density_profiling.py
│   │   ├── stage2_cascade_training.py
│   │   ├── stage3_feature_synthesis.py
│   │   └── stage4_meta_classifier.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging utilities
│       ├── metrics.py             # Evaluation metrics
│       └── visualization.py       # Plotting utilities
├── data/
│   ├── train.csv
│   └── test.csv
├── results/                       # Training outputs
│   └── plots/                     # Generated plots
├── training_logs/                 # Training logs
├── ensemble_voting.py             # ⭐ Ensemble implementation
├── test_ensemble.py               # ⭐ Quick ensemble test
├── main.py                        # Main entry point
├── train_ensemble.sh              # Background training script
├── monitor_training.sh            # Monitor training
├── watch_training.sh              # Watch training realtime
├── stop_training.sh               # Stop background training
├── view_logs.py                   # View training logs
├── ENSEMBLE_USAGE_GUIDE.md        # ⭐ Ensemble documentation
└── README.md
```

---

