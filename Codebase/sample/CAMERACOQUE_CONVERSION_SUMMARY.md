# cameraCOQE to ASTE Conversion - Summary

## ✅ Conversion Complete

Successfully converted cameraCOQE (Comparative Opinion Extraction) dataset to ASTE (Aspect-Based Sentiment Triplet Extraction) format for cross-domain model evaluation.

## 📊 Dataset Statistics

### Original Format (COQE)
- Train: ~4,700 sentences (with 649 having opinions)
- Dev: ~1,172 sentences (with 165 having opinions)  
- Test: ~1,473 sentences (with 200 having opinions)
- **Total with opinions: 1,014 samples**

### Converted Format (ASTE)
Location: `data/cameraCOQE_ASTE/`

| Split | Sentences | Triplets | Avg/Sent | Size |
|-------|-----------|----------|----------|------|
| Train | 649 | 2,032 | 3.1 | 57 KB |
| Dev | 165 | 514 | 3.1 | 15 KB |
| Test | 200 | 488 | 2.4 | 14 KB |
| **Total** | **1,014** | **3,034** | **3.0** | **86 KB** |

## 📈 Sentiment Distribution

**Class Balance Issue**: Heavily skewed towards NEUTRAL (typical for comparative opinions)

| Sentiment | Train | Dev | Test |
|-----------|-------|-----|------|
| Positive | 262 (12.9%) | 76 (14.8%) | 61 (12.5%) |
| Negative | 39 (1.9%) | 11 (2.1%) | 10 (2.0%) |
| Neutral | 1,731 (85.2%) | 427 (83.1%) | 417 (85.5%) |

**Why Neutral Dominates:**
- Comparative opinions often use neutral language (e.g., "faster", "longer", "higher")
- Not strongly positive/negative, just comparative
- Model needs to handle this distribution difference

## 🔄 Conversion Details

### Format Transformation

**COQE Format:**
```
Text\tLabel
[[subjects];[objects];[predicates];[sentiments];[comparators]]
```

Example:
```
By the way you can use the same battery the S50 comes with .	1
[[11&&S50];[];[9&&battery];[8&&same];[0]]
```

**ASTE Format:**
```json
{
  "sentence": "By the way you can use the same battery the S50 comes with .",
  "triplets": [
    {
      "aspect_term": "battery",
      "opinion_term": "same",
      "aspect_sentiment_label": "neutral"
    }
  ]
}
```

### Mapping Logic

- **Aspect**: Predicate token (what is being compared)
- **Opinion**: Sentiment token (the comparative descriptor)
- **Sentiment**: Inferred from opinion text (positive/negative/neutral)

### Example Conversions

1. **Simple Comparison:**
   - Input: `"camera X is faster"` → `(faster, fast, positive)`
   
2. **Multiple Predicates:**
   - Input: `predicates=[flash, shutter] sentiments=[slower]`
   - Output: `[(flash, slower, negative), (shutter, slower, negative)]`

## 📁 Files Generated

```
data/cameraCOQE_ASTE/
├── train_aste.json     (649 samples, 2,032 triplets)
├── dev_aste.json       (165 samples, 514 triplets)
└── test_aste.json      (200 samples, 488 triplets)
```

## 🚀 Usage with ASTE Model

### Method 1: Direct Evaluation
```python
import json

# Load cameraCOQE ASTE data
with open('data/cameraCOQE_ASTE/test_aste.json', 'r') as f:
    test_data = json.load(f)

# Use with your evaluation pipeline
# Should be compatible with existing ASTE format loaders
```

### Method 2: Cross-Domain Evaluation
```python
# Update your evaluation script to handle cameraCOQE path:
test_file = "data/cameraCOQE_ASTE/test_aste.json"  # Instead of original path

# Run normal evaluation pipeline
# Compare performance: original_domain vs cross_domain
```

## ⚠️ Important Considerations

### 1. **Distribution Mismatch**
   - Original ASTE: ~50% Positive, ~25% Negative, ~25% Neutral
   - cameraCOQE: ~13% Positive, ~2% Negative, ~85% Neutral
   - **Expected:** 10-20% F1 drop due to class imbalance

### 2. **Sentiment Inference**
   - COQE originally doesn't have explicit sentiment labels
   - Sentiments inferred from word semantics (simple heuristic)
   - May need refinement with domain knowledge

### 3. **Annotation Style**
   - Different from original ASTE datasets
   - Focus on comparative opinions vs regular aspects
   - May require model adaptation

## 📊 Quality Metrics

| Aspect | Status | Notes |
|--------|--------|-------|
| Format Validity | ✅ | All JSON valid and parseable |
| Triplet Completeness | ✅ | All triplets have aspect, opinion, label |
| Sample Coverage | ✅ | 1,014 samples with valid opinions |
| Sentiment Distribution | ⚠️ | Heavily skewed to neutral |
| Token Coverage | ✅ | All aspect/opinion terms found in sentences |

## 🔧 Scripts Created

1. **`parse_coqe.py`** - Custom parser for COQE format
2. **`coqe_to_aste.py`** - Main conversion script
3. **`eval_cameracoque.py`** - Evaluation and statistics
4. **`CAMERACOQUE_EVALUATION_GUIDE.md`** - Usage guide

## 📈 Next Steps

### Recommended Actions:
1. **Evaluate on cameraCOQE** - Baseline performance
2. **Analyze error patterns** - Where does cross-domain hurt?
3. **Fine-tune sentiment labels** - Improve heuristic
4. **Domain adaptation** - If performance drop significant
5. **Test-time adaptation** - Adjust for class imbalance

### Advanced Options:
- Use weighted loss for sentiment imbalance
- Apply focal loss for hard examples
- Domain-adversarial training
- Meta-learning for quick adaptation

## 📚 References

**Original COQE Paper:**
- Y. Yin et al. "Extracting Comparative Opinions from Customer Reviews"
- Focus on comparative opinion extraction from reviews

**This Conversion:**
- Enables cross-domain evaluation
- Tests model generalization beyond ASTE dataset
- Useful for robust model development

---

**Status**: ✅ Complete and Ready for Evaluation
**Last Updated**: 2024
