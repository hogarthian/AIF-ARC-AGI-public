# Pre-Fix State Documentation

**Bug Fix Applied:** 2025-11-05  
**Bug Description:** Test case instructions (T0, T1) were mismatched with input grids due to incorrect labeling in `follow_instructions.py`  
**Status:** Fixed in code, but pre-fix results contain invalid test case scores

**Additional Bug Discovered:** The data loader (`load_challenges_from_arc_prize_json`) was not loading ground truth outputs from `arc-agi_evaluation_solutions.json`. Test cases showed `has_ground_truth: false` even though ground truth exists. Fixed by adding `solutions_path` parameter to load solutions from separate file.

## Experiment Status at Time of Fix

### ✅ Completed Before Fix (10/14)
These experiments have both `results.json` and `results_reduced.json`, **BUT their test case results are invalid**:

1. `col_only_ascending` - ✅ Complete, ❌ Test cases invalid
2. `col_only_descending` - ✅ Complete, ❌ Test cases invalid
3. `image_only_ascending` - ✅ Complete, ❌ Test cases invalid
4. `image_only_descending` - ✅ Complete, ❌ Test cases invalid
5. `row_col_ascending` - ✅ Complete, ❌ Test cases invalid
6. `row_col_descending` - ✅ Complete, ❌ Test cases invalid
7. `row_image_ascending` - ✅ Complete, ❌ Test cases invalid
8. `row_image_descending` - ✅ Complete, ❌ Test cases invalid
9. `row_only_ascending` - ✅ Complete, ❌ Test cases invalid
10. `row_only_descending` - ✅ Complete, ❌ Test cases invalid

**Note:** All test case results in these experiments scored 0.0 due to the bug. Held-out validation results are valid (scores 0.3-0.9).

### ⚠️ Partial Before Fix (1/14)
These have hypothesis but incomplete results (will be overwritten when resumed):

1. `col_image_ascending` - Has hypothesis.json, incomplete results

### ⏳ Not Started (3/14)
Will be run fresh with the bug fix applied:

1. `col_image_descending`
2. `row_col_image_ascending`
3. `row_col_image_descending`

## What Will Be Overwritten When Resuming

When you resume the experiment, the following will be **re-run and overwritten**:

1. **`col_image_ascending`** - Will complete normally (was interrupted during held-out validation)
2. **`col_image_descending`** - Will run fresh
3. **`row_col_image_ascending`** - Will run fresh
4. **`row_col_image_descending`** - Will run fresh

**IMPORTANT:** The 10 completed experiments will **NOT** be automatically re-run. They will be skipped because `detect_completed_experiments()` checks for the existence of both `results.json` and `results_reduced.json`.

## Invalid Test Case Results (To Be Fixed Later)

All 10 completed experiments have test case results that need to be regenerated:

- **Held-out validation:** ✅ Valid (scores are correct)
- **Test cases (normal):** ❌ Invalid (all scored 0.0 due to bug)
- **Test cases (reduced):** ❌ Invalid (all scored 0.0 due to bug)

### Expected Pattern of Invalid Results

Looking at the invalid test case results, you'll see:
- All `similarity: 0.0` for test cases
- Uncertainty messages mentioning mismatches like:
  - "The instructions referred to the test case as 'T1', but the input data was labeled 'Test Input 0'"
  - Instructions describing features that don't match the provided input grid

## Script to Fix Invalid Test Cases

After the full experiment completes, create a script (`fix_pre_fix_test_cases.py`) that:

1. **Identifies pre-fix experiments:** Load experiments from `PRE_FIX_STATE.md` (the 10 listed above)
2. **For each experiment:**
   - Load `hypothesis.json` (contains valid instructions for T0, T1)
   - Load challenge data
   - Re-run only the test cases (normal + reduced) using `run_test_cases()` and `run_test_cases_reduced()`
   - Preserve held-out validation results (they're valid)
   - Update `results.json` and `results_reduced.json` with corrected test case scores
   - Update main `results.json` with corrected test case results

### Key Code Pattern

```python
# For each pre-fix experiment:
modality_type, order_name = ...

# Load existing results
results_dir = output_dir / f"{modality_type}_{order_name}"
with open(results_dir / "hypothesis.json") as f:
    hypothesis_data = json.load(f)
    belief = ... # Reconstruct from hypothesis_data

# Preserve held-out results (they're valid)
with open(results_dir / "results.json") as f:
    existing_results = json.load(f)
held_out_results = {k: v for k, v in existing_results.items() if k.startswith("E")}

# Re-run test cases only
test_results_normal, test_grids_normal = await run_test_cases(...)
test_results_reduced, test_grids_reduced = await run_test_cases_reduced(...)

# Combine: preserved held-out + new test cases
corrected_results = {**held_out_results}
# Add corrected test case results...
```

### Important Notes

- **Preserve held-out validation results:** They were generated correctly before the bug
- **Only regenerate test cases:** T0 and T1 results need to be fixed
- **Preserve all other metadata:** grids.json, hypothesis.json, reasoning_content, etc.
- **Update summary:** Update main `results.json` with corrected test case scores for plotting

## Verification After Fix

After regenerating test cases, verify:
1. Test case similarities are no longer all 0.0
2. Uncertainty messages no longer mention input labeling mismatches
3. Test case instructions match the input grids they're applied to
4. Plots generated from corrected results show meaningful test case scores

