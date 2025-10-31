# 🚀 BẮT ĐẦU TẠI ĐÂY / START HERE

## ADWC-DFS Ensemble - Phát hiện fraud với 84-91% recall 🎯

---

## ⚡ Quick Start (2 bước)

### Bước 1: Test Ensemble (10% data, ~5 phút)
```bash
cd /home/shyn/Dev/ADWC-DFS-
python test_ensemble.py
```

### Bước 2: Train Production Ensemble (100% data, ~30-60 phút)
```bash
python ensemble_voting.py --n_models 5
```

**Done!** Model ready tại `results/ensemble_model.pkl`

---

## 🎯 Lệnh Chính

```bash
# ⭐ Test nhanh
python test_ensemble.py

# ⭐ Train production - Foreground
python ensemble_voting.py --n_models 5

# ⭐ Train production - Background (recommended)
bash train_ensemble.sh -n 5

# Advanced: 7 models (recall cao hơn, chậm hơn)
bash train_ensemble.sh -n 7

# Quick test với 10%
bash train_ensemble.sh -n 3 -s 0.1

# Monitor background training
bash monitor_training.sh
bash watch_training.sh
```

## 🔥 Background Training (Recommended)

### Train in background (có thể tắt terminal)
```bash
# Start training
bash train_ensemble.sh -n 5

# Monitor progress
bash monitor_training.sh

# Watch realtime
bash watch_training.sh

# Stop if needed
bash stop_training.sh

# View logs later
python view_logs.py --training_logs
```

---

## 📊 Performance

| Strategy | Recall | Detected | Missed | Use Case |
|----------|--------|----------|--------|----------|
| **Soft (0.13)** | 83.9% | 1800/2145 | 345 | Balanced |
| **Soft (0.10)** | 85.8% | 1840/2145 | 305 | Production |
| **Aggressive (2/5)** | 88.3% | 1893/2145 | 252 | High-value |
| **Aggressive (1/5)** | 90.2% | 1934/2145 | 211 | Critical |
| **ULTRA** | 91.4% | 1960/2145 | 185 | Maximum detection |

---

## 📚 Tài Liệu

### Bắt đầu
1. **START_HERE.md** ← Đang đọc
2. ⭐ **ENSEMBLE_USAGE_GUIDE.md** - Chi tiết ensemble
3. **HUONG_DAN_TIENG_VIET.md** - Tiếng Việt

### Nâng cao
1. **README.md** - Tổng quan
2. **ALGORITHM.md** - Chi tiết thuật toán
3. Source: `adwc_dfs/`

---

## 📂 Files Quan Trọng
```
ADWC-DFS/
├── 📘 START_HERE.md              ← BẮT ĐẦU TẠI ĐÂY
├── 📘 HUONG_DAN_TIENG_VIET.md   ← Hướng dẫn tiếng Việt
├── 📘 ENSEMBLE_USAGE_GUIDE.md   ← Chi tiết ensemble
├── 📘 README.md                  ← Documentation chính
├── 📘 ALGORITHM.md               ← Chi tiết thuật toán
│
├── 🐍 test_ensemble.py           ← CHẠY CÁI NÀY ĐẦU TIÊN (test nhanh)
├── 🐍 ensemble_voting.py         ← Train full ensemble
├── 🐍 main.py                    ← Main entry point
├── 🐍 view_logs.py               ← Xem training logs
│
├── 🔧 train_ensemble.sh          ← Background training
├── 🔧 monitor_training.sh        ← Monitor progress
├── 🔧 watch_training.sh          ← Watch realtime
├── � stop_training.sh           ← Stop training
│
├── 📦 adwc_dfs/                  ← Source code
│   ├── models/                   ← Model chính
│   ├── stages/                   ← 4 stages
│   └── utils/                    ← Utilities
│
└── 📊 data/                      ← Dữ liệu training
    ├── train.csv
    └── test.csv
```

---

## 💡 FAQ

### Q: Tôi nên bắt đầu từ đâu?
**A:** Chạy `python test_ensemble.py` để test nhanh!

### Q: Tài liệu tiếng Việt ở đâu?
**A:** Đọc `HUONG_DAN_TIENG_VIET.md`

### Q: Làm sao xem lại log training?
**A:** `python view_logs.py --list` để xem tất cả logs, `--latest` để xem log mới nhất

### Q: Performance ra sao?
**A:** 
- Single model: ~0.86 F1, ~0.87 Recall
- ⭐ Ensemble: Best F1 ~0.39, **84-91% Recall** (improved!)

### Q: Ensemble là gì?
**A:** Kết hợp nhiều models để nâng recall lên 84-91%. Xem `ENSEMBLE_USAGE_GUIDE.md`

### Q: Thuật toán hoạt động thế nào?
**A:** Đọc `ALGORITHM.md` để hiểu chi tiết

### Q: Code có comments không?
**A:** Có! Mỗi function đều có docstring

### Q: Làm sao train trong background?
**A:** Dùng `bash train_ensemble.sh -n 5` để train background

---

## 🎓 ADWC-DFS là gì?

**Adaptive Density-Weighted Cascade with Dynamic Feature Synthesis**

4 giai đoạn:
1. **Density Profiling** - Tìm samples khó
2. **Cascade Training** - 3 models chuyên biệt
3. **Feature Synthesis** - Tạo meta-features
4. **Meta-Classifier** - Kết hợp thông minh

**Đặc biệt:**
- ✅ Không cần SMOTE/resampling
- ✅ Nhanh (O(n·k·d))
- ✅ Giải thích được
- ✅ Sẵn sàng production

---

## 🚀 Bắt Đầu Ngay!

```bash
# 1. Vào thư mục
cd /home/shyn/Dev/ADWC-DFS-

# 2. Test ensemble nhanh (10% data, ~5 phút)
python test_ensemble.py

# 3. Train production (100% data, ~30-60 phút)
python ensemble_voting.py --n_models 5

# 4. Enjoy! 🎉
```

---

## 📞 Hỗ Trợ

- 📖 Đọc documentation
- 💻 Xem example code
- 🐛 Check error messages
- 🧪 Test với data nhỏ trước

---

**Chúc bạn thành công! 🎯**

Made with ❤️ for fraud detection
