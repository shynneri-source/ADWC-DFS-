# ADWC-DFS Algorithm Documentation

## Implementation

**Base Algorithm:** LightGBM Gradient Boosting  
**Language:** Python 3.12+  
**Key Dependencies:** scikit-learn, LightGBM, NumPy  
**Current Version:** Optimized for 86.71% recall on fraud detection  

This document describes the mathematical foundation and implementation details of the ADWC-DFS algorithm. All formulas and parameters reflect the current production configuration.

## Mathematical Foundation

### Stage 1: Local Density Profiling

#### Local Intrinsic Dimensionality (LID)
```
LID_i = -1/k × Σ(log(d_j / d_k))
```
where:
- `d_j` = distance to j-th nearest neighbor
- `d_k` = distance to k-th nearest neighbor
- Higher LID indicates more complex local geometry

#### Class-Conditional Density Ratio (CCDR)
```
CCDR_i = log(ρ_fraud(x_i) + ε) - log(ρ_legit(x_i) + ε)
```
where:
- `ρ_fraud(x_i)` = proportion of fraud in k-neighborhood
- `ρ_legit(x_i)` = proportion of legit in k-neighborhood
- `ε = 1e-6` to avoid log(0)
- CCDR ≈ 0 indicates class overlap region

#### Difficulty Score (DS)
```
DS_i = α·|CCDR_i| + β·LID_i + γ·(1 - max_similarity_i)
```
where:
- `α = 0.4` (default): weight for class overlap
- `β = 0.3` (default): weight for local complexity
- `γ = 0.3` (default): weight for dissimilarity
- `max_similarity_i` = cosine similarity to nearest neighbor

### Stage 2: Cascade Training

#### Stratification
```
Easy:   DS_i < percentile(DS, 33)
Medium: percentile(DS, 33) ≤ DS_i < percentile(DS, 67)
Hard:   DS_i ≥ percentile(DS, 67)
```

#### Training Data for Each Specialist

**Easy Specialist:**
```
Data: Easy ∪ Medium
Weights: w_easy = 1.0, w_medium = 0.4
Scale_pos_weight: 40.0
```

**Medium Specialist:**
```
Data: Easy ∪ Medium ∪ Hard (all data)
Weights: w_easy = 0.2, w_medium = 1.0, w_hard = 0.6
Scale_pos_weight: 60.0
```

**Hard Specialist:**
```
Data: Hard ∪ Medium
Weights: w_hard = 1.0, w_medium = 0.5
Scale_pos_weight: 80.0
```

### Stage 3: Dynamic Feature Synthesis

#### Disagreement Features
```
disagreement_score = |P_easy - P_hard|

entropy_pred = -Σ P_m log(P_m)  for m ∈ {easy, medium, hard}

disagreement_easy_medium = |P_easy - P_medium|
disagreement_medium_hard = |P_medium - P_hard|
```

#### Confidence Geometry Features
```
confidence_variance = Var([P_easy, P_medium, P_hard])

confidence_std = Std([P_easy, P_medium, P_hard])

confidence_trajectory = (P_hard - P_easy) / (DS + ε)

prediction_range = max([P_easy, P_medium, P_hard]) - min([P_easy, P_medium, P_hard])

mean_prediction = (P_easy + P_medium + P_hard) / 3
```

#### Local Consensus Features
```
P_avg = (P_easy + P_medium + P_hard) / 3

neighbor_avg_pred = mean(P_avg[neighbors])

consensus_strength = 1 - |neighbor_avg_pred - P_avg|

neighbor_pred_variance = Var(P_avg[neighbors])

neighbor_majority_vote = mode(round(P_avg[neighbors]))
```

### Stage 4: Adaptive Meta-Classifier

#### Adaptive Sample Weighting
```
DS_normalized = DS / max(DS)

α_i = α_base × (1 + DS_normalized)

uncertainty_i = entropy_pred × confidence_variance

weight_i = α_i × (1 + uncertainty_i)
```
where:
- `α_base = 10.0` (default)
- Higher weight for difficult and uncertain samples

#### Meta-Features
```
X_meta = [
    P_easy, P_medium, P_hard,           # Cascade predictions (3)
    DS, LID, CCDR,                      # Density features (3)
    disagreement_score,                  # Disagreement (4)
    entropy_pred,
    disagreement_easy_medium,
    disagreement_medium_hard,
    confidence_variance,                 # Confidence geometry (5)
    confidence_std,
    confidence_trajectory,
    prediction_range,
    mean_prediction,
    neighbor_avg_pred,                   # Local consensus (4)
    consensus_strength,
    neighbor_pred_variance,
    neighbor_majority_vote
]
```
Total: 19 meta-features

## Complexity Analysis

### Training
```
Stage 1 (k-NN + LID + CCDR + DS):  O(n·k·d + n·k)
Stage 2 (3 specialists):            O(3·n·log(n)·d·T)
Stage 3 (feature synthesis):        O(n·(k + f))
Stage 4 (meta-classifier):          O(n·log(n)·f'·T')

Total: O(n·k·d + n·log(n)·d·T)
```
where:
- `n` = number of samples
- `k` = number of neighbors (30)
- `d` = number of features
- `f` = number of meta-features (19)
- `T` = number of trees per cascade model (200)
- `T'` = number of trees in meta-model (120)

### Inference (per sample)
```
Stage 1: O(k·d)          # k-NN search
Stage 2: O(3·d·log(T))   # 3 cascade predictions
Stage 3: O(k + f)        # Feature synthesis
Stage 4: O(f'·log(T'))   # Meta-classifier

Total: O(k·d + d·log(T) + f·log(T'))
```

Typical: ~10-20ms per sample

### Memory
```
k-NN model:        O(n·d)
3 cascade models:  O(3·T·d)
Meta-model:        O(T'·f)
Training data:     O(n·d)

Total: O(n·d + T·d)
```

Note: Using LightGBM models with T=200 trees per cascade model and T'=120 trees for meta-model

## Why ADWC-DFS Works

### 0. Choice of LightGBM

**Why LightGBM over XGBoost:**
- **Faster Training**: Histogram-based algorithm with gradient-based one-side sampling
- **Lower Memory**: Leaf-wise growth vs level-wise
- **Better Performance**: Native categorical feature support
- **Production Ready**: Optimized for deployment scenarios
- **Scale Handling**: Built-in handling of class imbalance with `scale_pos_weight`

**LightGBM Configuration:**
```python
CASCADE_PARAMS = {
    'n_estimators': 200,      # More trees for better accuracy
    'max_depth': 6,           # Moderate depth
    'learning_rate': 0.02,    # Low for stability
    'num_leaves': 31,         # Default leaf-wise growth
    'min_child_samples': 15,  # Prevent overfitting
    'subsample': 0.7,         # Row sampling
    'colsample_bytree': 0.7,  # Feature sampling
    'reg_alpha': 0.3,         # L1 regularization
    'reg_lambda': 1.0,        # L2 regularization
}
```

### 1. Multi-Level Imbalance Handling

**Local Level (Stage 1):**
- LID detects complex regions where minority class is buried
- CCDR identifies class overlap zones
- No assumption of global class distribution

**Model Level (Stage 2):**
- Three specialists with different training strategies
- Each model focuses on different aspect of data
- No single model forced to learn everything

**Loss Level (Stage 4):**
- Adaptive weighting increases focus on hard samples
- Uncertainty modulation for dynamic focusing
- Automatic adjustment without manual tuning

### 2. Meta-Learning from Disagreement

**Insight:** When models disagree, it signals uncertainty

**Exploitation:**
- High disagreement → uncertain region → higher DS
- Low consensus → outlier → potential fraud
- High entropy → need more careful examination

**Result:** Meta-classifier learns "where models struggle" = "where frauds hide"

### 3. No Synthetic Data

**Problem with SMOTE:**
- Creates synthetic samples by interpolation
- May generate unrealistic fraud patterns
- Overfitting risk on duplicated samples

**ADWC-DFS approach:**
- Uses sample weighting instead of resampling
- Preserves original data distribution
- Multiple views of same data (specialists)

### 4. Production-Ready Design

**Fast:** O(n·k·d) vs O(n²·d²) for graph methods

**Interpretable:** 
- Feature importance from LightGBM models
- Disagreement patterns are explainable
- Difficulty score is intuitive
- SHAP values available for predictions

**Robust:**
- No graph construction (stable)
- Local estimation (handles drift)
- Cascade design (fail-safe)
- LightGBM's built-in regularization

**Performance:**
- Single model: ~87% recall, ~42% precision
- Ensemble voting: ~90%+ recall achievable
- Inference: ~10-20ms per sample

## Comparison with Alternatives

### vs Standard Ensemble (Bagging/Boosting)
| Aspect | Standard Ensemble | ADWC-DFS |
|--------|------------------|----------|
| Data | Same data, different samples | Stratified by difficulty |
| Models | Independent | Specialized LightGBM |
| Combination | Voting/averaging | Meta-learning |
| Imbalance | Class weights only | Multi-level handling |
| Weights | Fixed per class | Adaptive per sample |

### vs Cost-Sensitive Learning
| Aspect | Cost-Sensitive | ADWC-DFS |
|--------|----------------|----------|
| Weighting | Fixed costs | Adaptive weights |
| Granularity | Class-level | Sample-level |
| Context | Global | Local topology |
| Uncertainty | Not considered | Explicitly modeled |
| Algorithm | Single model | Cascade + Meta |

### vs Active Learning
| Aspect | Active Learning | ADWC-DFS |
|--------|----------------|----------|
| Strategy | Query labels | Use existing labels |
| Iteration | Sequential | One-shot training |
| Uncertainty | For labeling | For weighting |
| Production | Requires human | Fully automated |
| Model | Any | LightGBM optimized |

## Hyperparameter Guide

### Critical Parameters

**K_NEIGHBORS (20-50):**
- Lower (20-30): Faster, more local
- Higher (40-50): Slower, more global
- Impact: Affects all density features
- Default: 30

**ALPHA, BETA, GAMMA (sum ≈ 1.0):**
- ALPHA (0.3-0.5): CCDR weight, class overlap
- BETA (0.2-0.4): LID weight, local complexity
- GAMMA (0.2-0.4): Similarity weight, outliers
- Adjust based on data characteristics
- Defaults: α=0.4, β=0.3, γ=0.3

**SCALE_POS_WEIGHT_*** (1-100):**
- Easy (20-50): Moderate focus on minority
- Medium (40-80): Strong focus
- Hard (60-100): Very strong focus on minority
- Increase if recall is too low
- Current best: Easy=40, Medium=60, Hard=80

### Secondary Parameters

**CASCADE_PARAMS (LightGBM):**
- **n_estimators** (50-300): Number of boosting trees
  - Default: 200 (best performance)
  - Trade-off: Accuracy vs speed
- **max_depth** (5-10): Tree depth
  - Default: 6
  - Lower: Faster, less overfitting
  - Higher: More expressive
- **learning_rate** (0.01-0.1): Step size
  - Default: 0.02 (for stability)
- **num_leaves** (20-50): Max leaves per tree
  - Default: 31
- **min_child_samples** (10-30): Min samples per leaf
  - Default: 15 (prevents overfitting)
- **subsample** (0.6-0.9): Row sampling
  - Default: 0.7
- **colsample_bytree** (0.6-0.9): Feature sampling
  - Default: 0.7

**META_PARAMS (LightGBM):**
- **n_estimators**: 120 (lighter than cascade)
- **max_depth**: 5 (keep it shallow to avoid overfitting)
- **learning_rate**: 0.02
- **num_leaves**: 20
- Keep it lightweight to avoid overfitting on meta-features

**ALPHA_BASE (5-15):**
- Controls adaptive sample weighting strength
- Default: 10.0 (aggressive weighting for hard samples)
- Lower: More uniform weighting
- Higher: Stronger focus on difficult samples

## Tuning Strategy

### Step 1: Start with Defaults
Run with default config to get baseline performance. Current defaults are optimized for high recall (86.71% on test data).

### Step 2: Adjust K_NEIGHBORS
- If training too slow: reduce to 20
- If performance plateaus: increase to 50
- Default of 30 is usually optimal

### Step 3: Tune SCALE_POS_WEIGHT
- **Low recall (<80%)**: Increase all three weights (e.g., 50, 70, 90)
- **Low precision (<40%)**: Decrease HARD weight
- **Balanced needed**: Use more conservative values (30, 50, 70)
- Current best for recall: Easy=40, Medium=60, Hard=80

### Step 4: Adjust Difficulty Weights (α, β, γ)
- Check difficulty distribution plots
- If fraud concentrated in one stratum: adjust weights
- α (CCDR): Higher for class overlap problems
- β (LID): Higher for complex geometry
- γ (Similarity): Higher for outlier detection

### Step 5: Fine-tune Model Complexity
- **Overfitting signs**: High train accuracy, low test
  - Reduce n_estimators (150), max_depth (5)
  - Increase regularization (reg_alpha, reg_lambda)
- **Underfitting signs**: Low train and test accuracy
  - Increase n_estimators (250), max_depth (7-8)
  - Reduce regularization

### Step 6: Adjust ALPHA_BASE
- Default 10.0 provides strong adaptive weighting
- Lower (5-8): More balanced weighting
- Higher (12-15): Very aggressive on hard samples
- Monitor if meta-classifier overfits on difficult samples

## Known Limitations

1. **Memory:** Stores k-NN index (O(n·d))
   - Solution: Use approximate NN (FAISS, Annoy)
   - Current: Feasible up to ~1M samples

2. **Cold Start:** New users have no neighbors in training set
   - Solution: Fallback to medium specialist prediction
   - Alternative: Retrain periodically with new data

3. **Concept Drift:** Needs periodic retraining
   - Solution: Monitor drift score, retrain meta-model
   - Recommendation: Monthly retraining for production

4. **High Dimensions:** k-NN performance degrades with d > 100
   - Solution: Dimensionality reduction (PCA, feature selection)
   - LightGBM handles high-dim better than k-NN

5. **Training Time:** 3 cascade models + meta = 4× base training time
   - Mitigation: LightGBM is fast, parallel training
   - Typical: 2-5 minutes for 200K samples

6. **Hyperparameter Sensitivity:** Performance depends on proper tuning
   - Provided: Best configuration achieving 86.71% recall
   - Tools: Grid search utilities included

## Future Enhancements

1. **Online Learning:** Incremental updates to meta-model without full retraining
2. **Approximate NN:** Use FAISS for scaling to millions of samples
3. **Auto-tuning:** Bayesian optimization for hyperparameter search
4. **Drift Detection:** Automatic retraining triggers based on performance monitoring
5. **Interpretability:** Enhanced SHAP values and feature importance visualization
6. **GPU Acceleration:** LightGBM GPU support for faster training
7. **Streaming Support:** Real-time predictions with model caching
8. **Multi-class Extension:** Extend beyond binary classification

## Current Features (Implemented)

✅ **Ensemble Voting:** Multiple models with voting for 90%+ recall  
✅ **Model Persistence:** Save/load trained models  
✅ **Batch Processing:** Efficient inference on large datasets  
✅ **Visualization:** Comprehensive plotting utilities  
✅ **Logging:** Structured logging with configurable levels  
✅ **Best Configuration:** Pre-tuned for optimal recall (86.71%)  
✅ **Production Ready:** Complete API with example usage

---

**Implementation:** All code available in `adwc_dfs/` directory using LightGBM  
**Documentation:** See README.md, QUICKSTART.md, and inline code comments  
**Performance:** Validated on real-world fraud detection dataset
