# 📊 DATA ANALYSIS REPORT - ASPECT-BASED SENTIMENT TRIPLET EXTRACTION

## 1. TÓNG QUAN DỮ LIỆU (Overview)

### Dataset Statistics
```
Total sentences:  5,989
Total entities:  20,504 (avg 3.42 per sentence)
Total triplets:  10,252 (avg 1.71 per sentence)
Entity-to-Triplet ratio: 2:1 (mỗi triplet có 2 thực thể: aspect + opinion)
```

### Dataset Breakdown
| Dataset | Train | Dev | Test | Total |
|---------|-------|-----|------|-------|
| **14lap** (Laptops) | 906 | 219 | 328 | 1,453 |
| **14res** (Restaurants) | 1,266 | 310 | 492 | 2,068 |
| **15res** (Restaurants) | 605 | 148 | 322 | 1,075 |
| **16res** (Restaurants) | 857 | 210 | 326 | 1,393 |

---

## 2. PHÂN TÍCH CHI TIẾT (Detailed Analysis)

### A. Sentiment Distribution (Phân bố cảm xúc)

**Problem:** Mất cân bằng dữ liệu LỚNNN! ⚠️
```
Positive (POSITIVE): 57.5% - 74.5%  (chiếm ưu thế)
Negative (NEGATIVE): 19.3% - 33.0%  (thiểu số)
Neutral  (NEUTRAL):  3.5% - 9.6%    (rất hiếm)
```

**Nguyên nhân:** Dữ liệu review tự nhiên → người dùng thường review khi rất hài lòng hoặc rất không hài lòng, ít neutral.

**Ảnh hưởng:**
- Model dễ bị overfitting vào POSITIVE
- Hiệu suất trên NEGATIVE & NEUTRAL kém
- F1-score không phản ánh đúng chất lượng

### B. Entity Characteristics (Đặc điểm thực thể)

**Entity Span Length Distribution:**
```
1 token:  74.5% - 82.4% (hầu hết là single-word)
2 tokens: 12-19%        (short phrases)
3+ tokens: < 5%         (multi-word entities hiếm)
```

**Ý nghĩa:**
- Thực thể chủ yếu là từ đơn (adjectives, nouns)
- VD: "good", "battery", "fast", "screen"
- Ít multi-word expressions: "battery life", "customer service"

### C. Triplet Patterns (Mẫu triplet)

**Pattern Structure:**
```
Triplet = (aspect/target, opinion, sentiment_type)
```

**Mẫu chính (99%+):**
- target → opinion (aspect points to opinion word)
- Luôn là cặp aspect-opinion, KHÔNG khi nào là opinion-target

**Multiplicity (Số triplet mỗi câu):**
- Trung bình: 1.6-2.0 triplet/câu
- Một số câu có 5-6 triplet (multi-aspect multi-opinion)
- Điều này làm tăng độ khó: phải trích xuất chính xác toàn bộ mối quan hệ

---

## 3. LÀM THẾ NÀO CÓ THỂ CẢI THIỆN MÔ HÌNH (Improvement Ideas)

### 🎯 Chiến lược 1: Xử lý Mất Cân Bằng Dữ Liệu

#### A. Weighted Loss (Trọng số mất mát)
```python
# Tính class weights dựa trên phân bố dữ liệu
class_weights = {
    'POSITIVE': 1.0,      # Phổ biến
    'NEGATIVE': 2.5-3.0,  # Hiếm hơn → trọng số cao
    'NEUTRAL': 5.0-8.0    # Rất hiếm → trọng số rất cao
}

# Sử dụng trong loss function
loss = weighted_cross_entropy(pred, target, weights=class_weights)
```

#### B. Focal Loss (Cho hard examples)
```python
# Tập trung vào những mẫu khó phân loại
focal_loss = -α * (1 - p_t)^γ * log(p_t)
# γ=2 thường tốt, tập trung vào hard negatives
```

#### C. Oversampling Minority Classes
```python
# Nhân đôi/nhân ba mẫu NEGATIVE và NEUTRAL trong training
# Hoặc dùng: SMOTE, mixup, cutmix
```

### 🎯 Chiến lược 2: Cải Thiện Trích Xuất Thực Thể

#### A. Character-level Encoding
```python
# Hiện tại: token-level
# Cải thiện: character-level CNN

input: "battery life"  →  char-level features
[b][a][t][t][e][r][y][ ][l][i][f][e]
        ↓
     CNN filters
        ↓
   better boundary detection
```

**Lợi ích:** Tự động học multi-word entities, không phụ thuộc tokenization

#### B. Soft Attention over Entity Spans
```python
# Thay vì hard selection, dùng attention weights
# Cho phép model học linh hoạt hơn
attention_weights = softmax(score)  # 0.0 - 1.0
entity_repr = sum(attention_weights * token_reps)
```

### 🎯 Chiến lược 3: Tận Dụng Thông Tin Cú Pháp

Hiện tại code đã dùng dependency parsing (TỐT!), nhưng có thể mở rộng:

```python
# Lấy syntactic path giữa aspect và opinion
# VD: "The battery is very good"
#
#     battery ---nsubj---> is ---xcomp---> good
#
# Path: [battery] ---(nsubj)---> [is] ---(xcomp)---> [good]
# Dùng path này để guide model

syntax_path_features = extract_dependency_path(aspect, opinion, tree)
```

### 🎯 Chiến lược 4: Data Augmentation

#### A. Aspect/Opinion Paraphrasing
```
Original:  "The battery is good"  → (battery, good, POSITIVE)
Paraphrase: "The battery is great" → (battery, great, POSITIVE)
Paraphrase: "The power cell is nice" → (power cell, nice, POSITIVE)
```

#### B. Swap Polarity (Cẩn thận!)
```
Original:  "The battery is good" → (battery, good, POSITIVE)
Negation:  "The battery is not bad" → (battery, bad, NEGATIVE) ❌
           "The battery is terrible" → (battery, terrible, NEGATIVE) ✓
```

#### C. Back-Translation
```
English:  "The screen is beautiful"
French:   "L'écran est magnifique"
English:  "The display is magnificent"
```

### 🎯 Chiến lược 5: Joint Learning (Học đồng thời)

```python
# Thay vì riêng lẻ:
# Entity extraction → Sentiment classification

# Dùng Multi-task Learning:
#                    ┌─ Entity Decoder ─┐
# Shared Encoder ─→ ├ Relation Decoder ├→ Final Triplet
#                    └─Sentiment Decoder┘

# Lợi ích:
# 1. Encoder học shared representations
# 2. Các task hỗ trợ lẫn nhau
# 3. Giảm overfitting
```

### 🎯 Chiến lược 6: Curriculum Learning

```
Epoch 1-10:   Học trên sentences với 1-2 triplet
Epoch 11-20:  Học trên sentences với 2-4 triplet
Epoch 21+:    Học trên toàn bộ data (5+ triplet)

Lợi ích: Model học gradually, không overwhelmed
```

---

## 4. ĐIỂM CỤ THỂ THEO TỪng DATASET

### 14lap (Laptop Reviews)
- **Đặc điểm:** Cân bằng nhất (POSITIVE: 57.5%)
- **Điểm yếu:** NEUTRAL hiếm (9.6%)
- **Gợi ý:** Tập trung cải thiện NEGATIVE (33%)

### 14res/15res/16res (Restaurant Reviews)
- **Đặc điểm:** Cực kỳ lệch về POSITIVE (73-75%)
- **Điểm yếu:** NEUTRAL rất hiếm (3.5-7.3%)
- **Gợi ý:** Urgent need for oversampling NEGATIVE/NEUTRAL

---

## 5. CROSS-DOMAIN EXPERIMENT (Thử nghiệm liên miền)

### Ý tưởng
```
Train on: 14res + 15res + 16res (nhà hàng)
Test on:  14lap (laptop)

Hoặc reverse:
Train on: 14lap (laptop)
Test on:  14res (nhà hàng)

Kiểm tra: Model có generalize sang domain mới không?
```

**Kết quả dự đoán:**
- Nếu tốt → Model học được syntactic/semantic patterns chung
- Nếu kém → Cần domain adaptation techniques

---

## 6. ERROR ANALYSIS FRAMEWORK

```python
def analyze_errors():
    errors = {
        'false_positive': [],    # Dự đoán nhưng không có
        'false_negative': [],    # Bỏ lỡ
        'sentiment_wrong': [],   # Đúng triplet nhưng sentiment sai
        'span_wrong': [],        # Sentiment đúng nhưng span sai
    }
    
    for pred, gold in predictions:
        if pred != gold:
            # Phân loại lỗi
            categorize_error(pred, gold, errors)
    
    # Phân tích theo:
    # 1. Sentiment type (POSITIVE vs NEGATIVE vs NEUTRAL)
    # 2. Entity length (1-token vs multi-token)
    # 3. Distance (close vs far aspect-opinion)
    # 4. Sentence length
    
    return errors
```

---

## 7. TRIỂN KHAI GỢI Ý

### Ưu tiên cao (High Priority)
1. ✅ **Implement weighted loss** → +2-3% F1
2. ✅ **Fix class imbalance** → +3-5% F1
3. ✅ **Error analysis** → Xác định bottleneck thực tế

### Ưu tiên trung (Medium Priority)
4. **Focal loss** → +1-2% F1
5. **Character-level encoding** → +1-2% F1
6. **Data augmentation** → +2-4% F1

### Ưu tiên thấp (Low Priority)
7. **Curriculum learning** → +0.5-1% F1
8. **Joint learning** → Phức tạp, gain không rõ

---

## 8. METRICS CẦN THEO DÕI

```python
# Không chỉ micro F1, mà cả:
metrics = {
    'overall_f1': calculate_f1(all_preds, all_golds),
    
    'f1_positive': calculate_f1(positive_preds, positive_golds),
    'f1_negative': calculate_f1(negative_preds, negative_golds),
    'f1_neutral': calculate_f1(neutral_preds, neutral_golds),
    
    'f1_short_entities': calculate_f1_for_length_range(1, 2),
    'f1_long_entities': calculate_f1_for_length_range(3, 100),
    
    'f1_close_pairs': calculate_f1_for_distance_range(1, 5),
    'f1_far_pairs': calculate_f1_for_distance_range(6, 100),
}
```

---

## 📝 TÓM TẮT HÀNH ĐỘNG

| # | Action | Effort | Expected Gain | Timeline |
|---|--------|--------|---------------|----------|
| 1 | Weighted Loss | ⭐ | ⭐⭐⭐ | 1-2h |
| 2 | Oversampling | ⭐ | ⭐⭐ | 1h |
| 3 | Error Analysis | ⭐⭐ | ⭐⭐⭐ | 2-3h |
| 4 | Focal Loss | ⭐ | ⭐⭐ | 1h |
| 5 | Char-level CNN | ⭐⭐⭐ | ⭐⭐ | 4-6h |
| 6 | Data Augmentation | ⭐⭐ | ⭐⭐⭐ | 3-4h |
| 7 | Curriculum Learning | ⭐⭐⭐ | ⭐ | 3-4h |
| 8 | Joint Learning | ⭐⭐⭐⭐ | ⭐⭐ | 6-8h |

---

**Created:** 2026-01-29
**Analysis scripts:** `data_analysis.py`, `advanced_analysis.py`
