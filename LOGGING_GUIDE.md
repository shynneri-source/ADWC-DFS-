# 📝 ADWC-DFS Logging Guide

## Overview

Tất cả các script training (`train.py`, `evaluate.py`, `demo.py`) đều tự động lưu log vào thư mục `logs/` với tên file có timestamp để dễ quản lý.

## Tính Năng Logging

### 1. Tự Động Lưu Log

Mỗi lần chạy training/evaluation, toàn bộ output từ terminal sẽ được lưu vào file log:

```bash
# Chạy training - log tự động được tạo
uv run train.py --sample_frac 0.1

# Output:
# 📝 Log file: logs/training_20241005_143022.log
```

### 2. Format Tên File Log

Tên file log có format: `{prefix}_{timestamp}.log`

- **prefix:** 
  - `training` - Training script
  - `evaluation` - Evaluation script  
  - `demo` - Demo script
  
- **timestamp:** `YYYYMMDD_HHMMSS`

**Ví dụ:**
- `training_20241005_143022.log` - Training lúc 14:30:22 ngày 05/10/2024
- `evaluation_20241005_150130.log` - Evaluation lúc 15:01:30
- `demo_20241005_152045.log` - Demo lúc 15:20:45

### 3. Nội Dung Log

Log file chứa:
- Timestamp bắt đầu training
- Configuration parameters
- Progress của từng stage
- Metrics và results
- Error messages (nếu có)
- Thời gian hoàn thành

## Cách Sử Dụng

### Chạy Với Logging (Mặc Định)

```bash
# Training với log
uv run train.py --train_path data/train.csv --test_path data/test.csv

# Evaluation với log
uv run evaluate.py --sample_frac 0.1

# Demo với log
uv run demo.py
```

### Tắt Logging

Nếu không muốn lưu log (chạy nhanh hơn một chút):

```bash
# Training không lưu log
uv run train.py --no_log

# Evaluation không lưu log
uv run evaluate.py --no_log
```

### Chỉ Định Thư Mục Log

```bash
# Lưu log vào thư mục khác
uv run train.py --log_dir my_logs
```

## Xem Log Files

### 1. Sử Dụng Script view_logs.py

#### Liệt Kê Tất Cả Log Files

```bash
uv run view_logs.py --list

# Output:
################################################################################
                                  Log Files
################################################################################

#    Filename                                 Size         Modified
--------------------------------------------------------------------------------
1    training_20241005_143022.log            15.2 KB      2024-10-05 14:35:12
2    evaluation_20241005_150130.log          8.5 KB       2024-10-05 15:03:45
3    demo_20241005_152045.log                3.2 KB       2024-10-05 15:21:30
```

#### Xem Log Mới Nhất

```bash
uv run view_logs.py --latest
```

#### Xem Log Cụ Thể

```bash
# Xem theo số thứ tự
uv run view_logs.py --view 1

# Xem theo tên file
uv run view_logs.py --file logs/training_20241005_143022.log
```

#### Xem N Dòng Đầu/Cuối

```bash
# Xem 50 dòng đầu
uv run view_logs.py --latest --lines 50

# Xem 100 dòng cuối
uv run view_logs.py --latest --lines 100 --tail
```

#### Filter Theo Prefix

```bash
# Chỉ xem training logs
uv run view_logs.py --list --prefix training

# Xem evaluation log mới nhất
uv run view_logs.py --latest --prefix evaluation
```

#### Tìm Kiếm Trong Logs

```bash
# Tìm keyword "error"
uv run view_logs.py --search error

# Tìm trong training logs only
uv run view_logs.py --search "F1 Score" --prefix training

# Tìm metrics
uv run view_logs.py --search "precision"
```

### 2. Sử Dụng Lệnh Linux

```bash
# Liệt kê logs
ls -lht logs/

# Xem log mới nhất
tail -f logs/training_*.log | tail -1

# Xem 100 dòng cuối
tail -100 logs/training_20241005_143022.log

# Tìm kiếm
grep "F1 Score" logs/training_*.log

# Xem real-time (trong khi training)
tail -f logs/training_*.log
```

## Ví Dụ Thực Tế

### Ví Dụ 1: Training và Xem Log

```bash
# Bước 1: Chạy training
uv run train.py --sample_frac 0.1

# Output sẽ hiển thị:
# 📝 Log file: logs/training_20241005_143022.log

# Bước 2: Sau khi training xong, xem log
uv run view_logs.py --latest
```

### Ví Dụ 2: So Sánh Nhiều Lần Training

```bash
# Training lần 1
uv run train.py --sample_frac 0.1 --k_neighbors 20

# Training lần 2
uv run train.py --sample_frac 0.1 --k_neighbors 30

# Training lần 3
uv run train.py --sample_frac 0.1 --k_neighbors 50

# Xem tất cả training logs
uv run view_logs.py --list --prefix training

# So sánh results
uv run view_logs.py --search "F1 Score" --prefix training
```

### Ví Dụ 3: Debug Lỗi

```bash
# Nếu training bị lỗi
uv run train.py --sample_frac 0.5

# Xem log để tìm lỗi
uv run view_logs.py --latest --search error

# Hoặc xem toàn bộ log
uv run view_logs.py --latest
```

### Ví Dụ 4: Monitor Training Real-time

```bash
# Terminal 1: Chạy training
uv run train.py --train_path data/train.csv

# Terminal 2: Theo dõi log real-time
tail -f logs/training_*.log
```

## Quản Lý Log Files

### Dọn Dẹp Logs Cũ

```bash
# Xóa logs cũ hơn 7 ngày
find logs/ -name "*.log" -mtime +7 -delete

# Xóa logs cũ hơn 30 ngày
find logs/ -name "*.log" -mtime +30 -delete

# Giữ lại 10 logs mới nhất, xóa phần còn lại
ls -t logs/*.log | tail -n +11 | xargs rm -f
```

### Backup Logs

```bash
# Backup toàn bộ logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/

# Backup chỉ training logs
tar -czf training_logs_$(date +%Y%m%d).tar.gz logs/training_*.log
```

### Tổ Chức Logs

```bash
# Tạo thư mục theo tháng
mkdir -p logs/archive/2024_10
mv logs/training_202410*.log logs/archive/2024_10/

# Tạo thư mục theo experiment
mkdir -p logs/experiment_1
mv logs/training_20241005_14*.log logs/experiment_1/
```

## Log Content Structure

Một log file điển hình có cấu trúc:

```
================================================================================
ADWC-DFS Training Log
Started: 2024-10-05 14:30:22
================================================================================

############################################################
                 ADWC-DFS Training Pipeline                 
############################################################

Timestamp: 2024-10-05 14:30:22
Configuration:
  Train path: data/train.csv
  Test path: data/test.csv
  ...

Loading data...
[Training progress...]

============================================================
              Stage 1: Local Density Profiling              
============================================================
[Stage 1 details...]

============================================================
                 Stage 2: Cascade Training                  
============================================================
[Stage 2 details...]

============================================================
             Stage 3: Dynamic Feature Synthesis             
============================================================
[Stage 3 details...]

============================================================
             Stage 4: Meta-Classifier Training              
============================================================
[Stage 4 details...]

[Results and metrics...]

############################################################
Training completed in X.XX seconds
############################################################
```

## Tips & Best Practices

### 1. Đặt Tên Experiment

Thêm prefix mô tả vào tên log:

```bash
# Không tốt
uv run train.py --sample_frac 0.1

# Tốt hơn - rename log sau
mv logs/training_20241005_143022.log logs/exp1_k20_training_20241005_143022.log
```

### 2. Ghi Chú Trong Log

Thêm comments vào đầu training:

```bash
# Tạo note file
echo "Experiment: Testing k_neighbors=50 with full data" > logs/exp_notes.txt
uv run train.py --k_neighbors 50
```

### 3. Compare Multiple Runs

```bash
# Extract metrics từ logs
grep "F1 Score:" logs/training_*.log > results_comparison.txt

# Hoặc dùng search
uv run view_logs.py --search "Key Test Metrics" --prefix training
```

### 4. Archive Important Runs

```bash
# Lưu lại runs quan trọng
mkdir -p logs/important_runs
cp logs/training_20241005_143022.log logs/important_runs/best_model_v1.log
```

## Troubleshooting

### Log File Không Được Tạo

```bash
# Check quyền write
ls -ld logs/

# Tạo thư mục nếu chưa có
mkdir -p logs
```

### Log File Quá Lớn

```bash
# Check size
du -h logs/*.log

# Compress logs cũ
gzip logs/training_202409*.log
```

### Không Thấy Output Trong Terminal

Nếu dùng `--no_log`, output chỉ ra terminal.
Nếu không dùng `--no_log`, output vào cả terminal và file.

## Advanced Usage

### Custom Log Directory Per Experiment

```bash
# Experiment 1
uv run train.py --log_dir logs/exp1 --k_neighbors 20

# Experiment 2
uv run train.py --log_dir logs/exp2 --k_neighbors 30

# Compare
uv run view_logs.py --log_dir logs/exp1 --latest
uv run view_logs.py --log_dir logs/exp2 --latest
```

### Parse Logs Programmatically

```python
import re

def extract_metrics(log_file):
    metrics = {}
    with open(log_file) as f:
        for line in f:
            if 'F1 Score:' in line:
                metrics['f1'] = float(re.search(r'(\d+\.\d+)', line).group(1))
            if 'PR AUC:' in line:
                metrics['pr_auc'] = float(re.search(r'(\d+\.\d+)', line).group(1))
    return metrics

# Extract từ tất cả logs
import glob
for log_file in glob.glob('logs/training_*.log'):
    metrics = extract_metrics(log_file)
    print(f"{log_file}: {metrics}")
```

## Summary

**Logging tự động:**
- ✅ Mọi output được lưu vào file
- ✅ Timestamp trong tên file
- ✅ Dễ dàng xem lại
- ✅ So sánh multiple runs
- ✅ Debug errors

**Commands:**
```bash
# Training với log (default)
uv run train.py

# Xem logs
uv run view_logs.py --list
uv run view_logs.py --latest
uv run view_logs.py --search "keyword"

# Manage logs
ls -lht logs/
tail -f logs/training_*.log
```

**Best practice:**
1. Luôn để logging bật (default)
2. Xem log sau mỗi run để verify
3. Backup logs quan trọng
4. Dọn dẹp logs cũ định kỳ

---

📝 **Mọi training run đều được log tự động - không lo mất thông tin!**
