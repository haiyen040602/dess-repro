# CameraCoQE Dataset Analysis Report

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 1,665 |
| **Total Unique Sentences** | 1,665 |
| **Total Triplets** | 2,382 |
| **Total Entities** | 4,171 |
| **Total Sentiments** | 2,382 |
| **Avg Entities/Sample** | 2.51 |
| **Avg Triplets/Sample** | 1.43 |
| **Avg Sentiments/Sample** | 1.43 |

## 📈 Data Split Statistics

| Split | Samples | Unique Sentences | Triplets | Entities | Avg Tokens |
|-------|---------|------------------|----------|----------|-----------|
| **TRAIN** | 1,066 | 1,066 | 1,526 | 2,669 | 24.7 |
| **DEV** | 266 | 266 | 376 | 652 | 23.5 |
| **TEST** | 333 | 333 | 480 | 850 | 23.6 |

## 🏷️ Entity Types

| Type | Count | Percentage |
|------|-------|-----------|
| **target** | 2,112 | 50.6% |
| **opinion** | 2,059 | 49.4% |

## 📏 Entity Length Distribution (tokens)

### Overall Distribution
| Length | 1 | 2 | 3-5 | 6-10 | 11+ | Avg |
|--------|---|---|-----|------|-----|-----|
| **Count** | 2,636 | 914 | 511 | 88 | 22 | 1.72 |
| **%** | 63.2% | 21.9% | 12.3% | 2.1% | 0.5% | - |

### By Split
| Split | 1 | 2 | 3-5 | 6-10 | 11+ | Avg Length |
|-------|---|---|-----|------|-----|-----------|
| TRAIN | 1,707 | 562 | 327 | 56 | 17 | 1.74 |
| DEV | 397 | 167 | 73 | 12 | 3 | 1.71 |
| TEST | 532 | 185 | 111 | 20 | 2 | 1.72 |

**Key Insight:** 63.2% of entities are single-token, 21.9% are 2-token. Most entities are short.

## 🎯 Triplet Distribution per Sample

### TRAIN (1,066 samples)
| Triplets/Sample | Count | Percentage |
|-----------------|-------|-----------|
| 1 | 759 | 71.2% |
| 2 | 212 | 19.9% |
| 3 | 63 | 5.9% |
| 4 | 21 | 2.0% |
| 5 | 7 | 0.7% |
| 6+ | 4 | 0.4% |
| **Max** | **17** | - |
| **Avg** | **1.43** | - |

### DEV (266 samples)
| Triplets/Sample | Count | Percentage |
|-----------------|-------|-----------|
| 1 | 188 | 70.7% |
| 2 | 56 | 21.1% |
| 3 | 13 | 4.9% |
| 4 | 8 | 3.0% |
| 5 | 1 | 0.4% |
| **Max** | **5** | - |
| **Avg** | **1.41** | - |

### TEST (333 samples)
| Triplets/Sample | Count | Percentage |
|-----------------|-------|-----------|
| 1 | 232 | 69.7% |
| 2 | 73 | 21.9% |
| 3 | 17 | 5.1% |
| 4 | 8 | 2.4% |
| 5 | 2 | 0.6% |
| 9 | 1 | 0.3% |
| **Max** | **9** | - |
| **Avg** | **1.44** | - |

**Key Insight:** ~71% of samples have exactly 1 triplet. Most sentences are simple comparative opinions.

## 💭 Sentiment Labels Distribution (4-Label Scheme)

### Overall Distribution
| Label | Count | Percentage |
|-------|-------|-----------|
| **BETTER** | 1,405 | 59.0% |
| **EQUAL** | 436 | 18.3% |
| **WORSE** | 396 | 16.6% |
| **DIFFERENT** | 145 | 6.1% |
| **Total** | **2,382** | **100%** |

### By Split

#### TRAIN (1,526 sentiments)
| Label | Count | Percentage |
|-------|-------|-----------|
| BETTER | 901 | 59.0% |
| EQUAL | 279 | 18.3% |
| WORSE | 255 | 16.7% |
| DIFFERENT | 91 | 6.0% |

#### DEV (376 sentiments)
| Label | Count | Percentage |
|-------|-------|-----------|
| BETTER | 226 | 60.1% |
| EQUAL | 67 | 17.8% |
| WORSE | 65 | 17.3% |
| DIFFERENT | 18 | 4.8% |

#### TEST (480 sentiments)
| Label | Count | Percentage |
|-------|-------|-----------|
| BETTER | 278 | 57.9% |
| EQUAL | 90 | 18.8% |
| WORSE | 76 | 15.8% |
| DIFFERENT | 36 | 7.5% |

**Key Insight:** BETTER class is dominant (59%), while DIFFERENT is minority (6.1%). Class imbalance exists but not severe.

## 📝 Token Statistics

### Overall
| Metric | Train | Dev | Test |
|--------|-------|-----|------|
| **Total Tokens** | 26,278 | 6,261 | 7,869 |
| **Unique Tokens** | 3,391 | 1,461 | 1,682 |
| **Avg Tokens/Sample** | 24.7 | 23.5 | 23.6 |
| **Max Tokens/Sample** | 148 | 95 | 73 |
| **Vocabulary Coverage** | - | - | - |

## 📊 Visualizations Generated

Two comprehensive visualization files have been created:

1. **cameracoque_analysis.png** - 6-panel overview including:
   - Overall sentiment distribution (pie chart)
   - Sentiment distribution by split (bar chart)
   - Sample and entity counts
   - Entity length distribution
   - Average entities/sentiments per sample
   - Tokens per sample

2. **cameracoque_sentiment_by_split.png** - Detailed sentiment distribution for each split

## 🔍 Key Findings

### Data Characteristics
1. **Balanced split:** 64% train, 16% dev, 20% test
2. **Simple comparative sentences:** 71% have exactly 1 triplet
3. **Short entities:** 85% of entities are ≤ 2 tokens
4. **Moderate sentence length:** ~24 tokens average
5. **Comparative nature:** Dataset specifically designed for comparative opinion extraction

### Label Distribution
1. **Class imbalance:** BETTER (59%) >> EQUAL (18.3%) >> WORSE (16.6%) > DIFFERENT (6.1%)
2. **Consistent across splits:** Label distribution is stable across train/dev/test
3. **4-label scheme:** Properly extracted from cameraCOQE comparator field
   - 1 → BETTER
   - 0 → EQUAL
   - -1 → WORSE
   - 2 → DIFFERENT

### Entity Analysis
1. **Balanced entity types:** target (50.6%) vs opinion (49.4%)
2. **Mostly single/dual-token:** 85% of entities are 1-2 tokens
3. **Average 2.51 entities per sample**

### Potential Challenges
1. **Class imbalance:** May need weighted loss functions for DIFFERENT class
2. **Sparse representation:** Some classes (DIFFERENT) have limited examples
3. **Simple patterns:** High single-triplet prevalence may limit model learning

## 💾 Dataset Files

### Format: D2E2S (training format)
- `data/cameraCOQE/train_dep_triple_polarity_result.json` (1,066 samples)
- `data/cameraCOQE/dev_dep_triple_polarity_result.json` (266 samples)
- `data/cameraCOQE/test_dep_triple_polarity_result.json` (333 samples)

### Format: ASTE (intermediate format)
- `data/cameraCOQE_ASTE/train_aste.json` (1,066 samples)
- `data/cameraCOQE_ASTE/dev_aste.json` (266 samples)
- `data/cameraCOQE_ASTE/test_aste.json` (333 samples)

## 🚀 Training Recommendations

1. **Use weighted loss:** Higher weight for DIFFERENT and WORSE classes
2. **Monitor per-class metrics:** F1-score per sentiment label
3. **Batch size:** 2-4 (for 16GB RAM, use gradient accumulation)
4. **Learning rate:** 5e-5 (DeBERTa fine-tuning)
5. **Epochs:** 5-10 (depends on validation performance)
6. **Early stopping:** Monitor macro-F1 on dev set

## 📌 Configuration

Update `Parameter.py` to train on cameraCOQE:
```python
parser.add_argument(
    "--dataset", default="cameraCOQE", type=str
)
```

Update `data/types.json` with 4-label sentiment scheme:
```json
{
  "sentiment": {
    "BETTER": {"short": "BET", "verbose": "BETTER", "symmetric": false},
    "EQUAL": {"short": "EQU", "verbose": "EQUAL", "symmetric": false},
    "WORSE": {"short": "WOR", "verbose": "WORSE", "symmetric": false},
    "DIFFERENT": {"short": "DIF", "verbose": "DIFFERENT", "symmetric": false}
  }
}
```

---

**Report Generated:** February 1, 2026  
**Dataset:** CameraCoQE (Comparative Opinion Extraction)  
**Total Coverage:** 1,665 samples with 2,382 aspect-sentiment triplets
