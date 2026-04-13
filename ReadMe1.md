# Model Configuration Update Instructions

## Key Model Parameters

[dev] sentences=1732 comparative=349 non_comparative=1383
[dev] mapped_label_counts={'DIF': 33, 'COM+': 203, 'COM-': 51, 'EQL': 133, 'SUP+': 28, 'COM': 18, 'SUP-': 1, 'SUP': 4}
[dev] underthesea_used=1605 fallback_linear=127
[test] sentences=3269 comparative=660 non_comparative=2609
[test] mapped_label_counts={'EQL': 244, 'COM+': 392, 'DIF': 53, 'SUP+': 71, 'COM': 19, 'COM-': 122, 'SUP': 2, 'SUP-': 5}

### 1. Pretrained Model

```python
parser.add_argument(
    "--pretrained_deberta_name", default="microsoft/deberta-v2-xxlarge", type=str
)
```

- **Description:** Specifies the pretrained DeBERTa model to use.
- **Default Value:** `microsoft/deberta-v2-xxlarge`
- **Modification:** Update this to use a different version of DeBERTa if necessary.

### 2. Feature Dimension

```python
parser.add_argument(
    "--deberta_feature_dim",
    type=int,
    default=1536,
    help="dimension of pretrained deberta feature",
)
```

- **Description:** Dimension of features extracted from the pretrained DeBERTa model.
- **Default Value:** `1536`
- **Modification:** Update this based on the feature size of the pretrained model.

### 3. Hidden Layer Dimension

```python
parser.add_argument(
    "--hidden_dim", type=int, default=768, help="hidden layer dimension."
)
```

- **Description:** Size of the hidden layer used in the model.
- **Default Value:** `768`
- **Modification:** Set this to half of `emb_dim` if a bidirectional LSTM is used.

### 4. Embedding Dimension

```python
parser.add_argument(
    "--emb_dim", type=int, default=1536, help="Word embedding dimension."
)
```

- **Description:** Dimension of the word embeddings.
- **Default Value:** `1536`
- **Modification:** Adjust this value as needed based on model requirements.

## Important Notes

- **Consistency Rule:** When using a bidirectional LSTM, ensure that `hidden_dim` is set to half of `emb_dim`.
- **Model Compatibility:** If switching to a different pretrained DeBERTa model, verify that the `deberta_feature_dim` matches the feature dimension of the new model.

## Update Instructions

1. Locate the parameter block in your code (usually in the argument parser).
2. Modify the specific values as necessary.
3. Ensure that `hidden_dim` is updated to `emb_dim / 2` if bidirectional LSTM is used.
4. Test the model to ensure compatibility with updated parameters.

---

### Example

```python
# Updating to a smaller DeBERTa model
parser.add_argument(
    "--pretrained_deberta_name", default="microsoft/deberta-v2-base", type=str
)
parser.add_argument(
    "--deberta_feature_dim",
    type=int,
    default=768,
    help="dimension of pretrained deberta feature",
)
parser.add_argument(
    "--hidden_dim", type=int, default=384, help="hidden layer dimension."
)
parser.add_argument(
    "--emb_dim", type=int, default=768, help="Word embedding dimension."
)
```
