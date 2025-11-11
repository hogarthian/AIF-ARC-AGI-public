# Bug Analysis: Test Case Instruction/Input Mismatch

## Summary

**CRITICAL BUG FOUND**: The test case instructions (T0, T1) are being mismatched with the actual test inputs during execution, causing all test cases to score 0.0.

## Root Cause

### The Bug Location
File: `src/utils/follow_instructions.py`, line 181

```python
modality_messages_list = await create_prompt_messages(
    challenge_data=temp_challenge,
    modality_type=modality_type,
    example_order=example_order if include_training_examples else None,
    test_idx=None,  # ❌ BUG: Always None, ignoring the test_idx parameter
    cache=use_cache
)
```

### What's Happening

1. **During Hypothesis Generation**:
   - All test cases are shown to the model
   - Model generates instructions `T0` and `T1` based on the order in `challenge_data.test`
   - `T0` refers to `challenge_data.test[0]`
   - `T1` refers to `challenge_data.test[1]`

2. **During Test Case Execution**:
   - For `test_idx=0`: Uses instruction `T0` + `challenge_data.test[0].input` ✅ Correct (by coincidence)
   - For `test_idx=1`: Uses instruction `T1` BUT the input is labeled as "Test Input 0" ❌ WRONG

3. **The Problem**:
   - `follow_instructions_to_generate_grid` creates a `TempChallenge` with only ONE test:
     ```python
     self.test = [type('TestInput', (), {'input': test_input})()]
     ```
   - Then calls `create_prompt_messages` with `test_idx=None`
   - Since `temp_challenge.test` has length 1, `create_prompt_messages` uses `test_indices = [0]`
   - So the test is **always labeled as "Test Input 0"** regardless of which actual test case we're processing
   - When processing `test_idx=1`, the instruction says "T1" but the modality shows "Test Input 0"
   - This confuses the LLM, causing it to see a mismatch between instruction and input

## Evidence from Results

### From `row_col_ascending/results.json`:

**T0 (test_idx=0)**:
- Uncertainty: "The instruction for test case T0 states 'The bottom hole contains seed T10 (blue, 1)'. However, based on the input grid coordinates, pixel T10 (row 10, column T) is located within the top hole of the 'B' shape, not the bottom hole."
- This suggests T0 instruction might actually be for T1's input, or there's confusion about which grid is being processed.

**T1 (test_idx=1)**:
- Uncertainty: "The instructions referred to the test case as 'T1', but the input data was labeled 'Test Input 0'."
- **This directly confirms the bug**: Instruction says T1, but input is labeled as Test Input 0.

### All Test Cases Score 0.0

- Both test cases scored 0.0 similarity
- Held-out validation works fine (scores 0.3-0.9)
- This confirms the bug only affects test case execution

## The Fix

**FIXED** ✅

The fix involves two changes:

1. **Modify `TempChallenge` in `src/utils/follow_instructions.py`** to store the original test index:
   - Added `original_test_idx` parameter to `TempChallenge.__init__`
   - Store `test_idx` parameter when creating `TempChallenge`

2. **Modify `_create_test_inputs_message` in `src/utils/modality_encoder.py`** to use the original test index:
   - Check if `challenge_data` has `original_test_idx` attribute
   - If present, override `test_indices` to use it for labeling

This ensures that when processing `test_idx=1`, the test input is labeled as "Test Input 1" (matching the T1 instruction) instead of "Test Input 0".

## Impact

- **Severity**: CRITICAL - All test case results are invalid (0.0 scores)
- **Scope**: Affects all modality experiments
- **Held-out validation**: Not affected (works correctly)
- **Fix Complexity**: Medium - Need to modify test labeling logic

## Next Steps

1. Fix the bug in `follow_instructions_to_generate_grid`
2. Rerun the experiment to verify test cases now score correctly
3. Consider if we need to regenerate existing results or just fix going forward

