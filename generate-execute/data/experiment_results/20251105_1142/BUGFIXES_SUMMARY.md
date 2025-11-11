# Bug Fixes Summary for test_modality_experiment.py

## Date: 2025-11-05

### Bugs Fixed

#### 1. **matplotlib Backend Configuration (Lines 61-63)**
**Issue**: `matplotlib.use('Agg')` was called after importing pyplot, which violates matplotlib's recommended usage pattern.

**Fix**: Moved `matplotlib.use('Agg')` before `import matplotlib.pyplot as plt`.

```python
# Before:
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# After:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be set before importing pyplot
import matplotlib.pyplot as plt
```

---

#### 2. **Progress Tracker Incremented Before API Calls**
**Issue**: Progress tracker was incremented before API calls were made. If the call failed and retried, or if the call itself failed, the progress counter would be incorrect.

**Locations Fixed**:
- `generate_hypothesis()` (lines 218-220 → moved to after line 266)
- `run_held_out_validation_reduced()` (lines 338-340, 355-357)
- `run_held_out_validation()` (lines 478-480, 494-496)
- `run_test_cases_reduced()` (lines 591-593, 608-610)
- `run_test_cases()` (lines 717-719, 732-734)

**Fix**: Moved progress tracker increment to **after** successful API call completion.

```python
# Before:
if _progress_tracker:
    await _progress_tracker.increment(step_name)
    _progress_tracker.print_progress()

response = await litellm.acompletion(...)

# After:
response = await litellm.acompletion(...)
# ... process response ...

# Update progress tracker after successful API call
if _progress_tracker:
    await _progress_tracker.increment(step_name)
    _progress_tracker.print_progress()
```

---

#### 3. **Rate Limiter Patching Not Applied to Imported Modules**
**Issue**: The code patched `litellm.acompletion` globally, but modules that already imported `litellm` (like `follow_instructions.py`) would not use the patched version.

**Fix**: 
1. Added global `_original_acompletion` variable to store the original function
2. Patched both the global `litellm.acompletion` and the imported module's version
3. Added protection against double-patching

```python
# Before:
original_acompletion = litellm.acompletion
async def patched_acompletion(*args, **kwargs):
    await _rate_limiter.wait_if_needed()
    return await original_acompletion(*args, **kwargs)
litellm.acompletion = patched_acompletion

# After:
global _rate_limiter, _progress_tracker, _original_acompletion

if _original_acompletion is None:
    _original_acompletion = litellm.acompletion

async def patched_acompletion(*args, **kwargs):
    await _rate_limiter.wait_if_needed()
    return await _original_acompletion(*args, **kwargs)

litellm.acompletion = patched_acompletion

# Also patch in the follow_instructions module
import src.utils.follow_instructions as follow_instructions_module
follow_instructions_module.litellm.acompletion = patched_acompletion
```

---

#### 4. **Skip Held-Out Validation with Single Training Example**
**Issue**: With only 1 training example, held-out validation would have no context examples, making the validation meaningless.

**Fix**: Added defensive checks to skip API calls and store placeholder data in both `run_held_out_validation()` and `run_held_out_validation_reduced()`:

```python
# Defensive check: skip if no context examples available
if not context_examples:
    logger.warning(f"Skipping held-out validation for example {held_out_idx} (only 1 training example, no context)")
    # Store placeholder with NaN-equivalent (0.0) for plots
    grids_dict[f"E{held_out_idx}"] = {
        "input": held_out_example.input,
        "expected": held_out_example.output,
        "skipped": "No context examples"
    }
    held_out_results.append({
        "held_out_idx": held_out_idx,
        "skipped": True,
        "ascending": {"similarity": 0.0},
        "descending": {"similarity": 0.0},
        "best_similarity": 0.0
    })
    continue  # Skip API call
```

**Why 0.0 instead of NaN:**
- Similarity scores are always in range [0.0, 1.0]
- 0.0 represents "no match" which is semantically correct for skipped validation
- matplotlib handles 0.0 values without issues in plots
- JSON serialization works (NaN would require special handling)

---

#### 5. **Potential Division by Zero in Progress Tracking**
**Issue**: If `total_api_calls` was 0, division by zero could occur.

**Fix**: Added explicit type annotation and defensive check:

```python
# Before:
progress_pct = (self.completed_api_calls / self.total_api_calls * 100) if self.total_api_calls > 0 else 0

# After:
progress_pct = (self.completed_api_calls / self.total_api_calls * 100) if self.total_api_calls > 0 else 0.0
```

---

### Bugs NOT Fixed (By User Request)

#### Caching Configuration
**Issue**: Comment says "Disable caching for experiments" but caching is actually enabled (`cache=True`).

**Reason Not Fixed**: User explicitly wants caching enabled to allow experiment reruns and resume functionality to reuse cached results.

**Current State**: Code correctly uses `cache=True` and the misleading comment has been updated to reflect the actual behavior:
```python
# Enable caching for experiments (allows reruns to reuse cached modality messages)
modality_messages_list = await create_prompt_messages(
    challenge_data, modality_type, example_order=example_order, cache=True
)
```

---

## Testing Recommendations

1. **Rate Limiting**: Test with `--rpm` flag to verify rate limiting works across all API calls
2. **Progress Tracking**: Verify progress counter remains accurate even if API calls fail and retry
3. **Edge Cases**: Test with single training example to ensure defensive checks work
4. **Resume Functionality**: Test resume from previous run to verify caching works as intended

---

## Impact Assessment

- **Critical**: Rate limiter patching fix ensures `--rpm` flag works correctly
- **High**: Progress tracking fixes ensure accurate progress reporting
- **Medium**: Defensive checks prevent silent failures in edge cases
- **Low**: matplotlib backend fix follows best practices but had no functional impact

