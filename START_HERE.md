# 🚀 BẮT ĐẦU TẠI ĐÂY / START HERE

## Xin chào! Đây là ADWC-DFS - Thuật toán phát hiện fraud tiên tiến 🎯

---

## ⚡ Quick Start (3 bước)

### Bước 1: Cài đặt
```bash
cd /home/shynn/source/ADWC-DFS
uv sync
```

### Bước 2: Chạy Demo
```bash
uv run demo.py
```

### Bước 3: Xem kết quả
- Performance metrics
- Feature importance
- Sample predictions

**Thời gian:** ~2-3 phút

---

## 📚 Tài Liệu Theo Cấp Độ

### 🟢 Người Mới Bắt Đầu
1. **START_HERE.md** ← Bạn đang ở đây!
2. **HUONG_DAN_TIENG_VIET.md** - Hướng dẫn tiếng Việt đầy đủ
3. **QUICKSTART.md** - Hướng dẫn nhanh bằng tiếng Anh
4. **LOGGING_GUIDE.md** - 📝 Cách xem và quản lý logs
5. ⭐ **ENSEMBLE_USAGE_GUIDE.md** - Hướng dẫn ensemble (nâng recall)
6. Chạy `demo.py` để xem nó hoạt động
7. Chạy `test_ensemble.py` để test ensemble

### 🟡 Người Có Kinh Nghiệm
1. **README.md** - Tổng quan và sử dụng
2. **train.py** - Script training đầy đủ
3. ⭐ **ensemble_voting.py** - Ensemble implementation
4. **evaluate.py** - So sánh với baselines
5. **example_usage.py** - Các ví dụ nâng cao

### 🔴 Chuyên Gia / Researcher
1. **ALGORITHM.md** - Chi tiết toán học
2. **PROJECT_SUMMARY.md** - Tổng quan project
4. Source code trong `adwc_dfs/`

---

## 🎯 Các Lệnh Quan Trọng

```bash
# Demo nhanh (10% data) - Tự động lưu log!
uv run demo.py
# 📝 Log: logs/demo_YYYYMMDD_HHMMSS.log

# Training đầy đủ (single model)
uv run train.py --train_path data/train.csv --test_path data/test.csv
# 📝 Log: logs/training_YYYYMMDD_HHMMSS.log

# Training nhanh (10% data)
uv run train.py --sample_frac 0.1

# ⭐ NEW: Test Ensemble (nâng recall lên 90%+)
python test_ensemble.py

# ⭐ NEW: Train Full Ensemble
python ensemble_voting.py --n_models 5

# So sánh với baselines
uv run evaluate.py --sample_frac 0.1
# 📝 Log: logs/evaluation_YYYYMMDD_HHMMSS.log

# Xem logs
uv run view_logs.py --list          # Liệt kê logs
uv run view_logs.py --latest        # Xem log mới nhất
uv run view_logs.py --search "F1"   # Tìm trong logs
```

---

## 📂 Cấu Trúc Quan Trọng

```
ADWC-DFS/
├── 📘 START_HERE.md              ← BẮT ĐẦU TẠI ĐÂY
├── 📘 HUONG_DAN_TIENG_VIET.md   ← Hướng dẫn tiếng Việt
├── 📘 QUICKSTART.md              ← Quick start guide
├── 📘 README.md                  ← Documentation chính
├── 📘 ALGORITHM.md               ← Chi tiết thuật toán
│
├── 🐍 demo.py                    ← CHẠY CÁI NÀY ĐẦU TIÊN
├── 🐍 train.py                   ← Training script
├── 🐍 evaluate.py                ← Comparison script
├── 🐍 example_usage.py           ← Examples
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
**A:** Chạy `uv run demo.py` ngay!

### Q: Tài liệu tiếng Việt ở đâu?
**A:** Đọc `HUONG_DAN_TIENG_VIET.md`

### Q: Làm sao xem lại log training?
**A:** `uv run view_logs.py --list` để xem tất cả logs, `--latest` để xem log mới nhất

### Q: Performance ra sao?
**A:** 
- Single model: ~0.86 F1, ~0.87 Recall
- ⭐ Ensemble: ~0.86 F1, **~0.90+ Recall** (improved!)

### Q: Ensemble là gì?
**A:** Kết hợp nhiều models để nâng recall lên 90%+. Xem `ENSEMBLE_USAGE_GUIDE.md`

### Q: Thuật toán hoạt động thế nào?
**A:** Đọc `ALGORITHM.md` để hiểu chi tiết

### Q: Code có comments không?
**A:** Có! Mỗi function đều có docstring

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
cd /home/shynn/source/ADWC-DFS

# 2. Chạy demo
uv run demo.py

# 3. Enjoy! 🎉
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
