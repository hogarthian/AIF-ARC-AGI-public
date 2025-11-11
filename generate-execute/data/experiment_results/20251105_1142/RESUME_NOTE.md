# Resume Information for Experiment 20251105_1142

**Challenge ID:** 13e47133  
**Session:** 13e47133-modality_experiment-20251105_1142  
**Interrupted at:** ~87/168 API calls (51.8%) - interrupted during `col_image_ascending` held-out validation  
**Bug Fix Applied:** After interruption, fixed test case input labeling bug (see BUG_ANALYSIS.md)

## ⚠️ IMPORTANT: Pre-Fix vs Post-Fix Results

**See `PRE_FIX_STATE.md` for detailed tracking of what was completed before the bug fix.**

### Critical Note
All experiments completed **before the bug fix** have:
- ✅ **Valid held-out validation results** (scores 0.3-0.9)
- ❌ **Invalid test case results** (all scored 0.0 due to bug)

The bug caused test case instructions (T0, T1) to be mismatched with their input grids. This has been fixed.

## Experiment Status

### Completed Before Fix (10/14) - Test Cases Invalid
These experiments have both `results.json` and `results_reduced.json`, **BUT their test case results need to be regenerated**:

1. ✅ `row_only_ascending` - Complete, ❌ Test cases invalid
2. ✅ `row_only_descending` - Complete, ❌ Test cases invalid
3. ✅ `col_only_ascending` - Complete, ❌ Test cases invalid
4. ✅ `col_only_descending` - Complete, ❌ Test cases invalid
5. ✅ `image_only_ascending` - Complete, ❌ Test cases invalid
6. ✅ `image_only_descending` - Complete, ❌ Test cases invalid
7. ✅ `row_col_ascending` - Complete, ❌ Test cases invalid
8. ✅ `row_col_descending` - Complete, ❌ Test cases invalid
9. ✅ `row_image_ascending` - Complete, ❌ Test cases invalid
10. ✅ `row_image_descending` - Complete, ❌ Test cases invalid

**Action Required:** After full experiment completes, run `fix_pre_fix_test_cases.py` to regenerate test case results (see PRE_FIX_STATE.md)

### Partial Before Fix (1/14)
These have hypothesis but incomplete results (will be overwritten when resumed):

1. ⚠️ `col_image_ascending` - Has hypothesis, interrupted during held-out validation

### Not Started (3/14)
Will be run fresh with the bug fix applied:

1. ⏳ `col_image_descending` - Will run with fix
2. ⏳ `row_col_image_ascending` - Will run with fix
3. ⏳ `row_col_image_descending` - Will run with fix

## Resume Instructions

To resume this experiment, run:

```bash
uv run python test_modality_experiment.py \
  --challenge-id 13e47133 \
  --resume-from-dir modality_experiment_results/13e47133/20251105_1142 \
  --rpm 60
```

The script will:
- Skip the 10 completed experiments (will NOT overwrite them automatically)
- Resume `col_image_ascending` from held-out validation (hypothesis already exists)
- Start fresh for the remaining 3 experiments (`col_image_descending`, `row_col_image_ascending`, `row_col_image_descending`)

**Note:** To regenerate test cases for the 10 pre-fix experiments, you'll need to manually run a fix script (see PRE_FIX_STATE.md for details).

## Important Notes About Data Format

### Old Data Format (Before Fix on 2025-11-05)
The completed experiments (`row_only`, `col_only`, `image_only`) were run with a bug where:
- Hypothesis was generated with a specific order (ascending or descending)
- BUT `follow_instructions_twice` was called, which always ran **both** orders
- This resulted in **2x API calls** per test case
- The `ascending` and `descending` scores in results may differ (they are real different runs)

### New Data Format (After Fix)
Starting from `row_col_ascending` resume and all future experiments:
- Hypothesis is generated with a specific order
- Only the matching order is run (using `follow_instructions_to_generate_grid` directly)
- Results are duplicated for backward compatibility (asc/desc fields are identical)
- This results in **1x API call** per test case (50% reduction)

### Implications for Analysis
- **Old data** (completed experiments): Asc/desc scores may differ - this is real data showing how order affects results
- **New data** (resumed and future): Asc/desc scores are identical - this is expected since only one order is run
- **Both formats are valid** for comparison, but be aware that:
  - Old runs show actual differences between order choices
  - New runs only test the hypothesis generation order (not the execution order)

### When Plotting
The plotting code detects mixed old/new format data and logs a note. When analyzing:
- Old format data (different asc/desc scores) represents real order effects
- New format data (identical asc/desc scores) only tests hypothesis generation order
- Compare old vs new data carefully, understanding they measure different things

## API Call Count

- **Total estimate:** 168 calls (14 experiments × 12 calls each: 1 hypothesis + 2×3 held-out + 2×2 test + 2×3 held-out-reduced + 2×2 test-reduced)
- **Completed before fix:** ~120 calls (10 experiments completed)
- **Remaining:** ~48 calls (1 partial + 3 new experiments)
  - `col_image_ascending`: ~36 calls remaining (was interrupted during held-out)
  - `col_image_descending`: ~12 calls
  - `row_col_image_ascending`: ~12 calls
  - `row_col_image_descending`: ~12 calls

## Files Structure

Each completed experiment directory contains:
- `hypothesis.json` - LLM response with reasoning_content
- `results.json` - Scores for normal context (with training examples)
- `results_reduced.json` - Scores for reduced context (no training examples)
- `grids.json` - All generated grids (input, expected, ascending, descending)
- `grids_reduced.json` - Reduced context grids

Partial experiments only have `hypothesis.json`.

