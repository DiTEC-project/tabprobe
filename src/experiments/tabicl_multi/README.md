# TabICL Multi-Output Modification

Modified fork of TabICL for multi-output prediction supporting association rule mining.

## Key Changes from Original TabICL

### 1. `model/learning.py` (ICLearning)

**Added Parameters:**
- `classes_per_feature: list[int]` - Number of classes for each feature (e.g., [6,3,11,7,3,3,2,6,2,2])

**Modified `__init__`:**
- Line 67: Store `classes_per_feature` for per-feature softmax
- Lines 70-75: Calculate `total_output_dim = sum(classes_per_feature)` and `multi_output` flag
- Line 90: Decoder outputs `total_output_dim` instead of `max_classes`
- Line 93: InferenceManager configured for `total_output_dim`

**Added Method:**
- `_predict_multi_output(R_context, R_query, ...)` (lines 240-292)
  - New prediction paradigm: (X_context, X_query) instead of (X, y_train)
  - Skips y_encoder (no single-feature conditioning needed)
  - Applies per-feature softmax: each feature's probabilities sum to 1 independently (Aerial-style)

**Modified `_predict_standard`:**
- Line 332-336: Conditional slicing - keep all dimensions for multi-output, slice to `num_classes` for single-output

### 2. `model/tabicl.py` (TabICL)

**Added Parameters:**
- `classes_per_feature: list[int]` - Passed through to ICLearning

**Modified `__init__`:**
- Line 92: Accept `classes_per_feature` parameter
- Line 110: Store for later use
- Line 147: Pass to ICLearning to enable multi-output

**Added Method:**
- `predict_multi_output(X_context, X_query, ...)` (lines 274-327)
  - Embeds context and query separately (not concatenated)
  - Calls `ICLearning._predict_multi_output` instead of `forward`
  - Returns predictions for all features simultaneously

## Architecture Comparison

| Aspect | Original TabICL | Modified (Multi-Output) |
|--------|----------------|------------------------|
| Input paradigm | X=[train, test], y_train | X_context, X_query |
| Output dimension | max_classes (10) | sum(classes_per_feature) (e.g., 45) |
| Softmax | Global across all classes | Per-feature (each sums to 1) |
| Prediction method | `forward()` with y_train | `predict_multi_output()` with X_context/X_query |
| Use case | Single-feature classification | Multi-feature reconstruction (rule mining) |

## Usage

```python
from src.experiments.tabicl_multi.model.tabicl import TabICL

# Initialize with feature class counts
model = TabICL(classes_per_feature=[6, 3, 11, 7, 3, 3, 2, 6, 2, 2])

# Add noise to context (denoising approach)
noisy_context = (context_table + np.random.randn(*context_table.shape) * 0.5).clip(0, 1)

# Convert to tensors
X_context = torch.FloatTensor(noisy_context).unsqueeze(0)
X_query = torch.FloatTensor(query_matrix).unsqueeze(0)

# Multi-output prediction
predictions = model.predict_multi_output(X_context, X_query)
# Shape: (1, n_queries, 45) with per-feature softmax
```

## Notes

- All changes are marked with inline comments: ADDED, CHANGED, MODIFIED, or NEW
- Original TabICL functionality preserved (single-output mode still works)
- No retraining performed - uses random initialization for new decoder weights