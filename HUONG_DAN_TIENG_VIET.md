# Hướng Dẫn Sử Dụng ADWC-DFS (Tiếng Việt)

## 🎯 Giới Thiệu

**ADWC-DFS** (Adaptive Density-Weighted Cascade with Dynamic Feature Synthesis) là một thuật toán machine learning tiên tiến được thiết kế đặc biệt cho bài toán **phát hiện gian lận** với dữ liệu mất cân bằng.

### Đặc điểm nổi bật:
- ✅ Không cần resampling (SMOTE, oversampling)
- ✅ Nhanh và hiệu quả (O(n·k·d))
- ✅ Dễ hiểu và giải thích được
- ✅ Sẵn sàng triển khai production
- ✅ Ít hyperparameter cần tune

## 🏗️ Kiến Trúc

Thuật toán gồm 4 giai đoạn:

### Giai đoạn 1: Phân Tích Mật Độ Cục Bộ
Tính toán độ khó cho mỗi mẫu dựa trên:
- **LID**: Độ phức tạp không gian cục bộ
- **CCDR**: Tỷ lệ mật độ giữa các class
- **DS**: Điểm số độ khó tổng hợp

### Giai đoạn 2: Huấn Luyện Cascade
Ba mô hình chuyên biệt:
- **Easy Model**: Học các mẫu dễ
- **Medium Model**: Học tất cả mẫu
- **Hard Model**: Tập trung vào mẫu khó

### Giai đoạn 3: Tổng Hợp Đặc Trưng Động
Tạo meta-features từ:
- Sự bất đồng giữa các models
- Độ tin cậy của predictions
- Đồng thuận từ neighbors

### Giai đoạn 4: Meta-Classifier
Mô hình cuối cùng với:
- Adaptive weighting
- Uncertainty-based learning

## 📦 Cài Đặt

```bash
# Di chuyển vào thư mục project
cd /home/shynn/source/ADWC-DFS

# Cài đặt dependencies với UV (khuyến nghị)
uv sync

# Hoặc dùng pip
pip install -r requirements.txt
```

## 🚀 Sử Dụng Nhanh

### 1. Chạy Demo (Khuyến nghị bắt đầu)

```bash
uv run demo.py
```

Demo sẽ:
- Load 10% dữ liệu training
- Huấn luyện model ADWC-DFS
- Đánh giá performance
- Hiển thị feature importance
- Show các predictions mẫu

**Thời gian chạy:** ~2-3 phút

### 2. Training Đầy Đủ

```bash
# Train với toàn bộ dữ liệu
uv run train.py --train_path data/train.csv --test_path data/test.csv
# 📝 Log tự động: logs/training_YYYYMMDD_HHMMSS.log

# Test nhanh với 10% dữ liệu
uv run train.py --sample_frac 0.1

# Tùy chỉnh cấu hình
uv run train.py --k_neighbors 50 --output_dir experiments/run1

# Training không lưu log (nếu muốn nhanh hơn)
uv run train.py --no_log
```

**Kết quả được lưu vào:**
- `results/adwc_dfs_model.pkl` - Model đã train
- `results/metrics.csv` - Metrics đánh giá
- `results/feature_importance.csv` - Độ quan trọng features
- `results/plots/` - Các biểu đồ
- `logs/training_*.log` - **Log đầy đủ quá trình training** 📝

### 2.5. Training Ensemble (Nâng cao Recall) ⭐

```bash
# Test nhanh ensemble với 10% data (~5-10 phút)
python test_ensemble.py

# Train ensemble đầy đủ với 5 models (~30-60 phút)
python ensemble_voting.py --n_models 5

# Test với 10% data
python ensemble_voting.py --n_models 5 --sample_frac 0.1

# Aggressive với 7 models
python ensemble_voting.py --n_models 7
```

**Kết quả Ensemble:**
- `results/ensemble_model.pkl` - Ensemble đã train
- `results/ensemble_results.csv` - Performance metrics
- Recall: **90%+** (cải thiện từ 87%)

**📚 Xem chi tiết:** [ENSEMBLE_USAGE_GUIDE.md](ENSEMBLE_USAGE_GUIDE.md)

### 3. So Sánh Với Baselines

```bash
uv run evaluate.py --sample_frac 0.1 --output_csv results/comparison.csv
# 📝 Log: logs/evaluation_YYYYMMDD_HHMMSS.log
```

So sánh với:
- XGBoost với class weights
- SMOTE + XGBoost  
- Random Forest
- ADWC-DFS

## 📝 Xem Training Logs

**Tất cả training tự động lưu log!**

```bash
# Liệt kê tất cả logs
uv run view_logs.py --list

# Xem log mới nhất
uv run view_logs.py --latest

# Xem log cụ thể (số thứ tự từ list)
uv run view_logs.py --view 1

# Tìm kiếm trong logs
uv run view_logs.py --search "F1 Score"

# Xem 100 dòng cuối
uv run view_logs.py --latest --lines 100 --tail

# Chỉ xem training logs
uv run view_logs.py --list --prefix training
```

**Log files:** `logs/training_YYYYMMDD_HHMMSS.log`

**Xem chi tiết:** [LOGGING_GUIDE.md](LOGGING_GUIDE.md) - Hướng dẫn đầy đủ về logging

## 💻 Sử Dụng Trong Code

### Cách 1: Cơ Bản (Single Model)

```python
from adwc_dfs import ADWCDFS

# Khởi tạo model với config mặc định
model = ADWCDFS(verbose=1)

# Huấn luyện
model.fit(X_train, y_train)

# Dự đoán
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=0.5)

# Feature importance
importance = model.get_feature_importance()
print(importance.head(10))
```

### Cách 1.5: Ensemble (Recall cao hơn) ⭐

```python
from ensemble_voting import VotingEnsemble

# Tạo ensemble với 5 models
ensemble = VotingEnsemble(n_models=5, voting='soft', verbose=1)

# Huấn luyện
ensemble.fit(X_train, y_train)

# Dự đoán với soft voting (cân bằng)
y_pred = ensemble.predict(X_test, threshold=0.13)

# Hoặc aggressive voting (maximize recall)
y_pred = ensemble.predict_aggressive(X_test, min_votes=2)

# Save ensemble
ensemble.save('results/ensemble_model.pkl')

# Load và sử dụng
loaded = VotingEnsemble.load('results/ensemble_model.pkl')
```

### Cách 2: Tùy Chỉnh Config

```python
from adwc_dfs import ADWCDFS
from adwc_dfs.config import ADWCDFSConfig

# Tạo config tùy chỉnh
config = ADWCDFSConfig()
config.K_NEIGHBORS = 50           # Tăng số neighbors
config.ALPHA = 0.5                # Tăng trọng số CCDR
config.SCALE_POS_WEIGHT_HARD = 20 # Focus nhiều hơn vào fraud

# Huấn luyện với config tùy chỉnh
model = ADWCDFS(config=config, verbose=1)
model.fit(X_train, y_train)
```

### Cách 3: Save và Load Model

```python
from adwc_dfs import ADWCDFS

# Train và save
model = ADWCDFS()
model.fit(X_train, y_train)
model.save('my_model.pkl')

# Load và sử dụng
loaded_model = ADWCDFS.load('my_model.pkl')
predictions = loaded_model.predict_proba(X_new)
```

## 🔧 Tùy Chỉnh Hyperparameters

### Parameters Quan Trọng

**1. K_NEIGHBORS** (20-50)
- Mặc định: 30
- Nhỏ hơn (20): Nhanh hơn, local hơn
- Lớn hơn (50): Chậm hơn, global hơn

```python
config.K_NEIGHBORS = 40
```

**2. ALPHA, BETA, GAMMA** (tổng ≈ 1.0)
- ALPHA: Trọng số cho CCDR (class overlap)
- BETA: Trọng số cho LID (complexity)
- GAMMA: Trọng số cho similarity

```python
config.ALPHA = 0.4
config.BETA = 0.3
config.GAMMA = 0.3
```

**3. SCALE_POS_WEIGHT** (1-20)
- Easy: 5 (mặc định)
- Medium: 10 (mặc định)
- Hard: 15 (mặc định)
- Tăng nếu recall thấp

```python
config.SCALE_POS_WEIGHT_EASY = 7
config.SCALE_POS_WEIGHT_MEDIUM = 12
config.SCALE_POS_WEIGHT_HARD = 18
```

## 📊 Hiểu Kết Quả

### Metrics Quan Trọng

- **Precision**: Trong số dự đoán là fraud, bao nhiêu % thật sự là fraud
- **Recall**: Trong số fraud thật, bắt được bao nhiêu %
- **F1 Score**: Trung bình điều hòa của Precision và Recall
- **PR AUC**: Diện tích dưới đường cong PR (tốt cho imbalanced data)
- **ROC AUC**: Diện tích dưới đường cong ROC

### Performance Mong Đợi (10% data)

**Single Model:**
| Metric | Giá Trị |
|--------|---------|
| Precision | ~0.69 |
| Recall | ~0.44 |
| F1 Score | ~0.54 |
| PR AUC | ~0.63 |
| ROC AUC | ~0.95 |

**Ensemble (5 models):** ⭐
| Strategy | Recall | Precision |
|----------|--------|-----------|
| Soft Voting | 88-90% | 16-18% |
| Aggressive (2/5) | 90-92% | 14-16% |
| Aggressive (1/5) | 92-95% | 12-14% |

**Lưu ý:** Kết quả tốt hơn với nhiều dữ liệu hơn!

### Top Features Thường Thấy

1. **confidence_trajectory**: Thay đổi prediction từ easy đến hard
2. **mean_prediction**: Prediction trung bình
3. **entropy_pred**: Độ bất đồng giữa models
4. **DS**: Difficulty score
5. **LID**: Local dimensionality

## 🎛️ Tuning Threshold

```python
# Thử các threshold khác nhau
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = model.predict(X_test, threshold=threshold)
    # Đánh giá...

# Threshold thấp (0.3-0.4): Recall cao, nhiều false positive
# Threshold cao (0.6-0.7): Precision cao, nhiều false negative
# Mặc định (0.5): Cân bằng
```

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi: Out of Memory

**Nguyên nhân:** k-NN tốn nhiều memory

**Giải pháp:**
```bash
# Giảm K_NEIGHBORS
uv run train.py --k_neighbors 20

# Hoặc dùng ít data hơn
uv run train.py --sample_frac 0.05
```

### Lỗi: Training Quá Chậm

**Giải pháp:**
```python
# Giảm độ phức tạp cascade models
config.CASCADE_PARAMS = {
    'n_estimators': 50,    # Giảm từ 100
    'max_depth': 5,        # Giảm từ 7
    ...
}
```

### Lỗi: Performance Kém

**Giải pháp:**
```python
# Tăng focus vào fraud
config.SCALE_POS_WEIGHT_HARD = 25

# Tăng số neighbors
config.K_NEIGHBORS = 50

# Giảm threshold
y_pred = model.predict(X_test, threshold=0.3)
```

## 📚 Tài Liệu Tham Khảo

### Cho Người Mới Bắt Đầu:
1. Đọc `QUICKSTART.md`
2. Chạy `demo.py`
3. Thử `train.py --sample_frac 0.1`
4. ⭐ Test ensemble: `python test_ensemble.py`

### Cho Người Có Kinh Nghiệm:
1. Đọc `README.md`
2. Chạy full training
3. ⭐ Train ensemble: `python ensemble_voting.py --n_models 5`
4. Đọc `ENSEMBLE_USAGE_GUIDE.md` ⭐
5. So sánh với baselines
6. Tune hyperparameters

### Cho Chuyên Gia:
1. Đọc `ALGORITHM.md`
2. Xem `example_usage.py`
3. ⭐ Tùy chỉnh ensemble strategies
4. Tùy chỉnh các stages
5. Mở rộng với domain features

## 💡 Tips và Tricks

### 1. Data Preparation
```python
# Luôn standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 2. Monitoring Training
```python
# Bật verbose để theo dõi
model = ADWCDFS(verbose=1)
```

### 3. Feature Engineering
```python
# Thêm domain-specific features trước khi train
# VD: transaction hour, day of week, amount bins, etc.
```

### 4. Threshold Selection
```python
# Chọn threshold dựa trên business requirement
# Cost của false positive vs false negative
if cost_FN > 10 * cost_FP:
    threshold = 0.3  # Ưu tiên recall
else:
    threshold = 0.7  # Ưu tiên precision
```

## 🎯 Use Cases

### 1. Fraud Detection (Phát hiện gian lận)
```python
# Credit card fraud, insurance fraud, etc.
model = ADWCDFS()
model.fit(X_transactions, y_is_fraud)
```

### 2. Anomaly Detection (Phát hiện bất thường)
```python
# Network intrusion, system failures, etc.
config.K_NEIGHBORS = 50  # Cần neighbors nhiều hơn
model = ADWCDFS(config=config)
```

### 3. Medical Diagnosis (Chẩn đoán y tế)
```python
# Rare disease detection
config.SCALE_POS_WEIGHT_HARD = 30  # Focus cao vào positive
model = ADWCDFS(config=config)
```

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Check các file documentation
2. Xem example scripts
3. Đọc code comments
4. Test với sample nhỏ trước

## 🎉 Kết Luận

ADWC-DFS là thuật toán:
- ✅ **Hoàn chỉnh** - Đầy đủ tính năng
- ✅ **Dễ sử dụng** - API đơn giản
- ✅ **Hiệu quả** - Performance tốt
- ✅ **Production-ready** - Sẵn sàng triển khai

**Bắt đầu ngay:**
```bash
cd /home/shynn/source/ADWC-DFS
uv run demo.py
```

---

**Chúc bạn phát hiện fraud thành công! 🎯🚀**
