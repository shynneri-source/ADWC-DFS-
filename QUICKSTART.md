# Quick Start Guide - ADWC-DFS

## Installation

```bash
# Clone or navigate to the project directory
cd ADWC-DFS

# Install dependencies using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Running the Demo

```bash
# Quick demo with 10% of data
uv run demo.py
```

This will:
- Load a sample of the training data
- Train the ADWC-DFS model
- Evaluate performance
- Show feature importance
- Display sample predictions

Expected output: F1 score ~0.53, PR AUC ~0.63 (on 10% sample)

## Training on Full Data

```bash
# Train with full data
uv run train.py --train_path data/train.csv --test_path data/test.csv
# 📝 Log file: logs/training_YYYYMMDD_HHMMSS.log

# Quick test with 10% sample
uv run train.py --sample_frac 0.1

# Custom configuration
uv run train.py --k_neighbors 50 --output_dir experiments/run1
```

Output files:
- `results/adwc_dfs_model.pkl` - Trained model
- `results/metrics.csv` - Performance metrics
- `results/feature_importance.csv` - Feature importance
- `results/plots/` - Visualization plots
- `logs/training_*.log` - **Complete training log** 📝

## Training Ensemble (Higher Recall) ⭐

```bash
# Quick ensemble test with 10% data (~5-10 minutes)
python test_ensemble.py

# Train full ensemble with 5 models (~30-60 minutes)
python ensemble_voting.py --n_models 5

# Quick test with 10% data
python ensemble_voting.py --n_models 5 --sample_frac 0.1
```

**Ensemble Results:**
- Recall improvement: **+2-5%** (87% → 90%+)
- Multiple voting strategies available
- See `ENSEMBLE_USAGE_GUIDE.md` for details

## Comparing with Baselines

```bash
# Compare ADWC-DFS with XGBoost, SMOTE, and Random Forest
uv run evaluate.py --sample_frac 0.1 --output_csv results/comparison.csv
# 📝 Log file: logs/evaluation_YYYYMMDD_HHMMSS.log
```

This will train and evaluate:
- XGBoost with class weights
- SMOTE + XGBoost
- Random Forest
- ADWC-DFS

Expected results (on 10% sample):
| Method | Precision | Recall | F1 | PR AUC |
|--------|-----------|--------|----|----|
| XGBoost | ~0.70 | ~0.40 | ~0.51 | ~0.55 |
| SMOTE+XGBoost | ~0.65 | ~0.50 | ~0.56 | ~0.58 |
| Random Forest | ~0.68 | ~0.42 | ~0.52 | ~0.56 |
| **ADWC-DFS** | **~0.69** | **~0.44** | **~0.54** | **~0.63** |

## Viewing Training Logs 📝

All training runs automatically save logs:

```bash
# List all logs
uv run view_logs.py --list

# View latest log
uv run view_logs.py --latest

# View specific log
uv run view_logs.py --view 1

# Search in logs
uv run view_logs.py --search "F1 Score"

# View last 100 lines
uv run view_logs.py --latest --lines 100 --tail
```

**Log files:** `logs/training_YYYYMMDD_HHMMSS.log`

Xem chi tiết: [LOGGING_GUIDE.md](LOGGING_GUIDE.md)

## Using the Trained Model

### Single Model
```python
from adwc_dfs import ADWCDFS

# Load trained model
model = ADWCDFS.load('results/adwc_dfs_model.pkl')

# Predict on new data
y_pred_proba = model.predict_proba(X_new)
y_pred = model.predict(X_new, threshold=0.5)

# Get feature importance
importance = model.get_feature_importance()
print(importance.head(10))

# Get predictions from individual cascade models
cascade_preds = model.get_cascade_predictions(X_new)
print(f"Easy model: {cascade_preds['easy'][:5]}")
print(f"Medium model: {cascade_preds['medium'][:5]}")
print(f"Hard model: {cascade_preds['hard'][:5]}")
```

### Ensemble Model ⭐
```python
from ensemble_voting import VotingEnsemble

# Load trained ensemble
ensemble = VotingEnsemble.load('results/ensemble_model.pkl')

# Predict with soft voting (balanced)
y_pred = ensemble.predict(X_new, threshold=0.13)

# Predict with aggressive voting (maximize recall)
y_pred = ensemble.predict_aggressive(X_new, min_votes=2)

# Two-stage detection
y_pred = ensemble.predict_two_stage(
    X_new, 
    stage1_threshold=0.13, 
    stage2_threshold=0.05
)

# Get probability scores
y_pred_proba = ensemble.predict_proba(X_new)
```

## Customizing Configuration

```python
from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig

# Create custom config
config = ADWCDFSConfig()

# Adjust hyperparameters
config.K_NEIGHBORS = 50  # More neighbors (slower but more accurate)
config.ALPHA = 0.5  # Increase weight for CCDR
config.SCALE_POS_WEIGHT_HARD = 20.0  # More focus on hard frauds

# Train with custom config
model = ADWCDFS(config=config, verbose=1)
model.fit(X_train, y_train)
```

## Understanding the Output

### Difficulty Score Distribution
The model automatically identifies easy, medium, and hard samples:
- **Easy samples** (DS < 33rd percentile): Clear-cut cases
- **Medium samples** (33rd ≤ DS < 67th percentile): Typical patterns
- **Hard samples** (DS ≥ 67th percentile): Edge cases at decision boundaries

### Feature Importance
Top meta-features typically include:
1. **confidence_trajectory**: How predictions change from easy to hard model
2. **mean_prediction**: Average prediction across cascade
3. **entropy_pred**: Disagreement between models
4. **DS**: Overall difficulty score
5. **LID**: Local intrinsic dimensionality

### Performance Metrics
- **Precision**: Of predicted frauds, how many are actually frauds?
- **Recall**: Of actual frauds, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall
- **PR AUC**: Area under precision-recall curve (best for imbalanced data)
- **Recall@Precision=0.9**: Maximum recall achievable at 90% precision

## Troubleshooting

### Memory Issues
If you run out of memory:
```bash
# Use smaller sample
uv run train.py --sample_frac 0.05

# Reduce k_neighbors
uv run train.py --k_neighbors 20
```

### Slow Training
If training is too slow:
```bash
# Reduce cascade model complexity
# Edit config.py:
CASCADE_PARAMS = {
    'n_estimators': 50,  # Reduce from 100
    'max_depth': 5,      # Reduce from 7
    ...
}
```

### Poor Performance
If performance is not satisfactory:
```bash
# Increase scale_pos_weight for more focus on fraud
# Increase k_neighbors for better density estimation
uv run train.py --k_neighbors 50

# Or adjust threshold
y_pred = model.predict(X_test, threshold=0.3)  # Lower threshold = higher recall
```

## Next Steps

1. **Experiment with hyperparameters**: Try different values for `K_NEIGHBORS`, `ALPHA`, `BETA`, `GAMMA`
2. **Feature engineering**: Add domain-specific features to the input data
3. **Threshold tuning**: Find optimal threshold for your use case
4. **Production deployment**: Use `model.save()` and `ADWCDFS.load()` for serving

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the code in `adwc_dfs/` directory
- Look at example outputs in `results/` directory after training

## Performance Tips

For best results:
- Use at least 20,000 samples for training
- Ensure features are standardized (handled automatically)
- Balance between precision and recall based on business needs
- Monitor feature importance to understand model behavior
- Retrain periodically if data distribution changes (concept drift)

---

**Happy fraud detection! 🎯**
