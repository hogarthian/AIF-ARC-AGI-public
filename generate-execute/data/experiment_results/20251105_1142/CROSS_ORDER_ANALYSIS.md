# Cross-Order Effect Analysis

# Human conclusions based on jupyternotebook plots:

1. All the dots are allone the 45 degree line (asc score == desc score), indicates that switching example orders between hypothesis gen and follow_instruction stages, doesn't make any impact on the scores (asc hyp + asc test == asc hyp + desc test)
2. Example orders in hyp stage change the instructions, but not impacting the scores consistently:
   2a. In some cases, asc hyp works better, other times desc hyp works better.
   2b. Image_only works worst, but in other times, adding image to text modality always improve
   2c. Orders doesn't change the instruction quality! Our original theory is that score will be higher for the examples provided earlier (score 0>1>2 in asc hyp, 2>1>0 in desc hyp), but in the end, it looks like the exp complexity determines the final score. (Ex2 is the easiest, always highest score, 1 is the hardest, always lowest score.)
   2d. row_only is a little bit better than col_only, but row_col seems to get worse. adding image removed the difference.
3. Reduced run (exclude training examples from modality and filter out example instructions) provided almost the same score, just a little bit lower. This is important. It means that the instruction quality actually dominated the final score. Which make sense, at the generation stage, the examples provide little added value, because we included all the hidden clue from training in the working hypothesis and general field already. So improving the initial hypothesis quality is the key.
4. Test seems always have a lower score.
5. Source of error seems to come from missing certain details, so iteration would be the key. I think we done our job in creating a good perception.

## Overview

This analysis examines whether instructions created with one example order (ascending/descending) perform differently when tested with the same order vs. the opposite order.

**Key Question**: Do instructions created with ascending order perform better when tested with ascending order, or are they order-agnostic?

## Key Findings

### 1. **Row-Only Modality** - Minimal Cross-Order Effect
- **Ascending hypothesis (held-out)**: Mean difference = +0.015 (asc test slightly better)
- **Descending hypothesis (held-out)**: Mean difference = -0.011 (desc test slightly better)
- **Conclusion**: Very small differences (~1-2%), suggesting instructions are largely order-agnostic for row-only modality. Test cases show mixed results with larger variability.

### 2. **Col-Only Modality** - Moderate Cross-Order Effect
- **Ascending hypothesis (held-out)**: Mean difference = +0.022 (asc test better)
- **Descending hypothesis (held-out)**: Mean difference = +0.062 (asc test better, but more variable)
- **Conclusion**: Moderate differences (2-6%), with some variability. Instructions created with descending order show larger differences when tested with different orders, especially in reduced context.

### 3. **Image-Only Modality** - Moderate Cross-Order Effect
- **Ascending hypothesis (held-out normal)**: Mean difference = -0.033 (desc test better)
- **Ascending hypothesis (held-out reduced)**: Mean difference = +0.049 (asc test better)
- **Descending hypothesis**: Mean difference = -0.019 to +0.037 (mixed)
- **Test cases**: Both orders show asc test better (+0.054 and +0.051)
- **Conclusion**: Moderate differences, with context-dependent effects. Reduced context shows stronger order sensitivity.

### 4. **Row-Col Modality** - Moderate Cross-Order Effect
- **Ascending hypothesis (held-out normal)**: Mean difference = +0.031 (asc test better)
- **Ascending hypothesis (held-out reduced)**: Mean difference = +0.108 (asc test MUCH better)
- **Descending hypothesis**: Mean difference = -0.010 to -0.025 (desc test better)
- **Conclusion**: Strong order sensitivity in reduced context for ascending hypothesis (+10.8%). Descending hypothesis shows opposite pattern.

### 5. **Row-Image Modality** - Moderate Cross-Order Effect
- **Ascending hypothesis (held-out)**: Mean difference = +0.009 (minimal)
- **Ascending hypothesis (held-out reduced)**: Mean difference = -0.066 (desc test better)
- **Test cases**: Desc test better for ascending hypothesis (-0.066 to -0.097)
- **Conclusion**: Context-dependent effects. Reduced context shows stronger order sensitivity, with descending test performing better.

### 6. **Col-Image Modality** - Minimal to Moderate Cross-Order Effect
- **Ascending hypothesis (held-out)**: Mean difference = +0.014 to +0.023 (asc test slightly better)
- **Descending hypothesis (held-out reduced)**: Mean difference = -0.036 (desc test better)
- **Conclusion**: Small to moderate differences depending on context. Generally minimal effects in normal context.

### 7. **Row-Col-Image Modality** - Minimal Cross-Order Effect
- **Ascending hypothesis (held-out)**: Mean difference = -0.002 to +0.013 (minimal)
- **Descending hypothesis**: Mean difference = +0.011 to -0.019 (minimal)
- **Conclusion**: Very small differences (~1-2%), suggesting full modality combination is largely order-agnostic.

## Detailed Statistics

### Held-Out Validation (Normal Context)

| Modality | Hypothesis Order | n | Mean Asc Test | Mean Desc Test | Mean Diff | Std Diff | Max Diff |
|----------|-----------------|---|---------------|----------------|-----------|----------|----------|
| row_only | ascending | 3 | 0.879 | 0.863 | +0.015 | 0.013 | +0.031 |
| row_only | descending | 3 | 0.606 | 0.616 | -0.011 | 0.008 | 0.000 |
| col_only | ascending | 3 | 0.570 | 0.547 | +0.022 | 0.021 | +0.050 |
| col_only | descending | 3 | 0.617 | 0.555 | +0.062 | 0.147 | +0.265 |
| image_only | ascending | 3 | 0.353 | 0.385 | -0.033 | 0.048 | +0.002 |
| image_only | descending | 3 | 0.402 | 0.421 | -0.019 | 0.052 | +0.042 |
| row_col | ascending | 3 | 0.386 | 0.355 | +0.031 | 0.034 | +0.078 |
| row_col | descending | 3 | 0.619 | 0.629 | -0.010 | 0.075 | +0.050 |
| row_image | ascending | 3 | 0.785 | 0.776 | +0.009 | 0.010 | +0.022 |
| row_image | descending | 3 | 0.571 | 0.578 | -0.007 | 0.025 | +0.020 |
| col_image | ascending | 3 | 0.753 | 0.739 | +0.014 | 0.020 | +0.043 |
| col_image | descending | 3 | 0.605 | 0.589 | +0.016 | 0.024 | +0.050 |
| row_col_image | ascending | 3 | 0.814 | 0.816 | -0.002 | 0.002 | 0.000 |
| row_col_image | descending | 3 | 0.628 | 0.617 | +0.011 | 0.029 | +0.050 |

### Held-Out Validation (Reduced Context)

| Modality | Hypothesis Order | n | Mean Asc Test | Mean Desc Test | Mean Diff | Std Diff | Max Diff |
|----------|-----------------|---|---------------|----------------|-----------|----------|----------|
| row_only | ascending | 3 | 0.854 | 0.853 | +0.001 | 0.013 | +0.017 |
| row_only | descending | 3 | 0.612 | 0.629 | -0.017 | 0.022 | +0.008 |
| col_only | ascending | 3 | 0.566 | 0.559 | +0.007 | 0.009 | +0.020 |
| col_only | descending | 3 | 0.500 | 0.511 | -0.010 | 0.252 | +0.295 |
| image_only | ascending | 3 | 0.492 | 0.443 | +0.049 | 0.124 | +0.220 |
| image_only | descending | 3 | 0.446 | 0.410 | +0.037 | 0.056 | +0.115 |
| row_col | ascending | 3 | 0.360 | 0.252 | +0.108 | 0.112 | +0.228 |
| row_col | descending | 3 | 0.567 | 0.592 | -0.025 | 0.032 | +0.003 |
| row_image | ascending | 3 | 0.739 | 0.805 | -0.066 | 0.046 | -0.030 |
| row_image | descending | 3 | 0.574 | 0.579 | -0.005 | 0.004 | 0.000 |
| col_image | ascending | 3 | 0.773 | 0.750 | +0.023 | 0.028 | +0.063 |
| col_image | descending | 3 | 0.523 | 0.559 | -0.036 | 0.026 | 0.000 |
| row_col_image | ascending | 3 | 0.816 | 0.803 | +0.013 | 0.012 | +0.030 |
| row_col_image | descending | 3 | 0.614 | 0.633 | -0.019 | 0.020 | 0.000 |

### Test Cases (Normal Context)

| Modality | Hypothesis Order | n | Mean Asc Test | Mean Desc Test | Mean Diff | Std Diff | Max Diff |
|----------|-----------------|---|---------------|----------------|-----------|----------|----------|
| row_only | ascending | 2 | 0.225 | 0.225 | 0.000 | 0.000 | 0.000 |
| row_only | descending | 2 | 0.201 | 0.365 | -0.164 | 0.163 | -0.001 |
| col_only | ascending | 2 | 0.148 | 0.154 | -0.006 | 0.016 | +0.010 |
| col_only | descending | 2 | 0.131 | 0.123 | +0.008 | 0.004 | +0.012 |
| image_only | ascending | 2 | 0.412 | 0.358 | +0.054 | 0.005 | +0.059 |
| image_only | descending | 2 | 0.436 | 0.386 | +0.051 | 0.007 | +0.058 |
| row_col | ascending | 2 | 0.227 | 0.277 | -0.051 | 0.053 | +0.002 |
| row_col | descending | 2 | 0.226 | 0.229 | -0.003 | 0.007 | +0.003 |
| row_image | ascending | 2 | 0.432 | 0.498 | -0.066 | 0.005 | -0.061 |
| row_image | descending | 2 | 0.456 | 0.483 | -0.027 | 0.049 | +0.022 |
| col_image | ascending | 2 | 0.523 | 0.543 | -0.020 | 0.020 | 0.000 |
| col_image | descending | 2 | 0.272 | 0.264 | +0.008 | 0.021 | +0.029 |
| row_col_image | ascending | 2 | 0.359 | 0.339 | +0.021 | 0.023 | +0.043 |
| row_col_image | descending | 2 | 0.534 | 0.527 | +0.007 | 0.002 | +0.009 |

### Test Cases (Reduced Context)

| Modality | Hypothesis Order | n | Mean Asc Test | Mean Desc Test | Mean Diff | Std Diff | Max Diff |
|----------|-----------------|---|---------------|----------------|-----------|----------|----------|
| row_only | ascending | 2 | 0.217 | 0.225 | -0.008 | 0.008 | 0.000 |
| row_only | descending | 2 | 0.393 | 0.382 | +0.011 | 0.033 | +0.044 |
| col_only | ascending | 2 | 0.141 | 0.230 | -0.089 | 0.058 | -0.031 |
| col_only | descending | 2 | 0.143 | 0.137 | +0.006 | 0.007 | +0.013 |
| image_only | ascending | 2 | 0.411 | 0.371 | +0.041 | 0.001 | +0.041 |
| image_only | descending | 2 | 0.415 | 0.384 | +0.031 | 0.014 | +0.044 |
| row_col | ascending | 2 | 0.244 | 0.226 | +0.018 | 0.018 | +0.036 |
| row_col | descending | 2 | 0.224 | 0.229 | -0.005 | 0.003 | -0.002 |
| row_image | ascending | 2 | 0.360 | 0.457 | -0.097 | 0.032 | -0.066 |
| row_image | descending | 2 | 0.511 | 0.434 | +0.076 | 0.042 | +0.118 |
| col_image | ascending | 2 | 0.495 | 0.555 | -0.060 | 0.041 | -0.019 |
| col_image | descending | 2 | 0.232 | 0.238 | -0.006 | 0.017 | +0.011 |
| row_col_image | ascending | 2 | 0.352 | 0.353 | -0.001 | 0.012 | +0.011 |
| row_col_image | descending | 2 | 0.521 | 0.521 | -0.001 | 0.007 | +0.007 |

## Interpretation

### What This Means

1. **Row-Only**: Instructions are largely order-agnostic. The small differences suggest that row-based instructions don't depend heavily on example order. Test cases show some variability but overall minimal effect.

2. **Col-Only**: Moderate sensitivity to order. Instructions may benefit from using the same order for generation and testing, but the effect is not consistent. Reduced context shows larger variability.

3. **Image-Only**: Moderate order sensitivity with context-dependent effects. Test cases consistently show ascending test performing better (+3-5%), while held-out validation shows mixed results depending on context.

4. **Row-Col**: Strong order sensitivity in reduced context for ascending hypothesis (+10.8%). Descending hypothesis shows opposite pattern, suggesting order effects are hypothesis-dependent.

5. **Row-Image**: Context-dependent effects. Reduced context shows stronger order sensitivity, with descending test performing better for ascending hypothesis. Test cases also favor descending test.

6. **Col-Image**: Small to moderate differences depending on context. Generally minimal effects in normal context, but reduced context shows some order sensitivity.

7. **Row-Col-Image**: Very small differences (~1-2%), suggesting full modality combination is largely order-agnostic. This may indicate that combining all modalities provides robustness against order effects.

### Recommendations

**For Your Experiment Design:**

1. **Keep the current approach** (testing both orders) if you want to:
   - Understand order sensitivity across modalities
   - Find the best order for each modality
   - Measure robustness of instructions

2. **Consider matching orders** (generate and test with same order) if you want to:
   - Optimize performance for each modality
   - Reduce variability in results
   - Test the hypothesis that instructions work best with the order they were created with

### Current Data Status

- ✅ **Completed**: 14/14 experiments (all 7 modalities × 2 orders)
- ✅ **Data Fixed**: All invalid grids and missing test results have been corrected
- ✅ **Analysis Updated**: Complete cross-order analysis with all modalities

### Key Insights

1. **Order effects are modality-dependent**: Some modalities (row_only, row_col_image) show minimal order effects, while others (row_col, row_image) show stronger effects, especially in reduced context.

2. **Context matters**: Reduced context generally amplifies order effects, suggesting that training examples provide some robustness against order sensitivity.

3. **Hypothesis order matters**: Instructions created with ascending order may perform differently than those created with descending order, even when tested with the same order.

4. **Full modality combination is robust**: Row-col-image modality shows minimal order effects, suggesting that combining all modalities provides the most robust instructions.

## Files Generated

- `plots/cross_order_effects_normal.png` - Cross-order line plots (normal context)
- `plots/cross_order_effects_reduced.png` - Cross-order line plots (reduced context)
- `plots/cross_order_scatter_normal.png` - Scatter plots (normal context)
- `plots/cross_order_scatter_reduced.png` - Scatter plots (reduced context)
- `plots/cross_order_statistics.json` - Detailed statistics

## Notes

- **Test cases**: Now include ground truth scores (previously missing/invalid). Test case analysis shows order effects similar to held-out validation, with some modalities showing stronger effects in test cases (e.g., row_image, col_only reduced).

- **Reduced context**: Shows more pronounced order effects across most modalities, especially for row_col (+10.8%) and row_image (-6.6% to -9.7%). This suggests training examples provide robustness against order sensitivity.

- **Sample sizes**: Small (n=1-3 per condition), so these findings should be interpreted with caution. However, consistent patterns across modalities suggest real effects.

- **Data quality**: All invalid grids and missing test results have been fixed. Analysis reflects complete and corrected dataset.

