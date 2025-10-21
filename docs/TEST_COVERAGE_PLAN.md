# Test Coverage Improvement Plan

Based on current coverage analysis (39% overall):

---

## Current Coverage Status

```
Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
ml_event_tagger/__init__.py         2      0   100%
ml_event_tagger/config.py          12      0   100%
ml_event_tagger/model.py           38     38     0%   10-134
ml_event_tagger/preprocess.py     104     49    53%   83-85, 194-213, 217-266
ml_event_tagger/serve.py          119     15    87%   114, 133, 142, 151-155, 236, 319-320, 331-336
ml_event_tagger/train.py          167    167     0%   11-329
-------------------------------------------------------------
TOTAL                             442    269    39%
```

---

## Priority 1: Critical Gaps (High Value, Low Cost)

### 1.1 Model Creation Tests (`model.py` - 0% coverage)

**Impact:** High - Core ML functionality  
**Effort:** Low - Simple unit tests  
**Target:** 80%+ coverage

**Recommended Tests:**

```python
# tests/test_model.py

def test_create_model_returns_keras_model():
    """Model creation returns a Keras Sequential model."""
    
def test_create_model_correct_architecture():
    """Model has expected layers (Embedding, Pooling, Dense, etc.)."""
    
def test_create_model_correct_output_shape():
    """Model output shape matches number of tags."""
    
def test_create_model_is_compiled():
    """Model is compiled with optimizer, loss, and metrics."""
    
def test_create_model_inference():
    """Model can perform inference on dummy data."""
```

**Lines to cover:** 10-134 (entire file)  
**Estimated time:** 30 minutes  
**Value:** Validates core ML architecture

---

### 1.2 Preprocessing Edge Cases (`preprocess.py` - 53% coverage)

**Impact:** Medium - Data quality affects model performance  
**Effort:** Low - Extend existing tests  
**Target:** 75%+ coverage

**Missing Coverage:**

- Lines 83-85: Error handling in `prepare_dataset()`
- Lines 194-213: `save_preprocessed_data()` function
- Lines 217-266: `if __name__ == "__main__"` script (low priority)

**Recommended Tests:**

```python
# tests/test_preprocess.py (add to existing file)

def test_prepare_dataset_with_empty_events():
    """Handle empty event list gracefully."""
    
def test_prepare_dataset_with_malformed_tags():
    """Handle events with invalid tag structure."""
    
def test_save_preprocessed_data():
    """Save function writes correct files."""
    
def test_save_preprocessed_data_creates_directory():
    """Save function creates output directory if needed."""
```

**Lines to cover:** 83-85, 194-213  
**Estimated time:** 20 minutes  
**Value:** Ensures robustness for edge cases

---

### 1.3 Serve Error Handling (`serve.py` - 87% coverage)

**Impact:** Medium - Production reliability  
**Effort:** Low - Edge case tests  
**Target:** 95%+ coverage

**Missing Coverage:**

- Line 114: Model loading error (already covered by startup)
- Line 133: Tokenizer loading error
- Line 142: Model config loading error
- Lines 151-155: Shutdown handler
- Line 236: Model prediction exception
- Lines 319-320: Root endpoint error cases
- Lines 331-336: Predict endpoint exception handling

**Recommended Tests:**

```python
# tests/test_serve.py (add to existing file)

def test_predict_handles_preprocessing_error():
    """Predict endpoint handles text preprocessing errors."""
    
def test_predict_handles_tokenization_error():
    """Predict endpoint handles tokenization failures."""
    
def test_predict_with_extremely_long_text():
    """Predict endpoint handles very long input text."""
    
def test_lifespan_shutdown():
    """Lifespan shutdown completes cleanly."""
```

**Lines to cover:** 151-155, 236, 331-336  
**Estimated time:** 30 minutes  
**Value:** Production error resilience

---

## Priority 2: Training Pipeline (Low Priority for Demo)

### 2.1 Training Code (`train.py` - 0% coverage)

**Impact:** Low - Training is one-time, expensive to test  
**Effort:** High - Requires fixtures, mocks, time  
**Target:** Skip for MVP demo (acceptable for portfolio)

**Rationale:**
- Training code is run manually, not in production
- Expensive to test (requires model training)
- Would need fixtures/mocks for TensorFlow
- Better validated through:
  - Manual training runs
  - Jupyter notebook validation
  - Model performance metrics

**Alternative Validation:**
- ✅ Already validated via notebook
- ✅ Model artifacts exist (h5, json)
- ✅ Training history plots verified

**Recommendation:** **SKIP** - Not worth effort for demo project

---

## Priority 3: Script Entry Points (Low Value)

### 3.1 Main Script Blocks

**Missing:**
- Lines 217-266 in `preprocess.py` (`if __name__ == "__main__"`)
- Lines 11-329 in `train.py` (`if __name__ == "__main__"`)

**Recommendation:** **SKIP** - Low value for coverage metrics

**Rationale:**
- Script entry points are for manual execution
- Testing them provides little value
- Better validated through integration tests
- Increases test complexity

---

## Implementation Plan

### Phase 1: Model Tests (30 min)
1. Create `tests/test_model.py`
2. Add 5 model creation tests
3. Target: 80%+ coverage of `model.py`

### Phase 2: Preprocessing Edge Cases (20 min)
1. Extend `tests/test_preprocess.py`
2. Add 4 edge case tests
3. Target: 75%+ coverage of `preprocess.py`

### Phase 3: Serve Error Handling (30 min)
1. Extend `tests/test_serve.py`
2. Add 4 error handling tests
3. Target: 95%+ coverage of `serve.py`

**Total Time:** ~80 minutes  
**Expected Coverage:** 39% → 65-70% overall

---

## Target Coverage Goals

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `__init__.py` | 100% | 100% | ✅ Done |
| `config.py` | 100% | 100% | ✅ Done |
| `model.py` | 0% | 80% | 🔴 High |
| `preprocess.py` | 53% | 75% | 🟡 Medium |
| `serve.py` | 87% | 95% | 🟡 Medium |
| `train.py` | 0% | 0% | ⚪ Skip |
| **TOTAL** | **39%** | **65-70%** | **Target** |

---

## Recommendation Summary

**Do:**
1. ✅ Add model creation tests (`test_model.py`)
2. ✅ Add preprocessing edge cases
3. ✅ Add serve error handling tests

**Don't:**
1. ❌ Test training pipeline (expensive, low value)
2. ❌ Test `__main__` blocks (low value)
3. ❌ Chase 100% coverage (diminishing returns)

**Rationale for 65-70% Target:**
- Covers all critical production code paths
- Reasonable for a demo/portfolio project
- Focuses on code that runs in production API
- Skips expensive-to-test training code
- Industry standard for microservices is 60-80%

---

## Next Steps

1. Review this plan
2. Decide if 65-70% coverage is acceptable for MVP
3. If yes, implement Phase 1 (model tests) first
4. Run coverage again to verify improvements
5. Update documentation with final coverage metrics

**Estimated effort:** ~80 minutes total  
**Expected outcome:** Professional test coverage for production code
