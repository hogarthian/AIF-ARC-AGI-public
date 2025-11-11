# Hypothesis Quality Analysis Report

Author: composer 1

## Executive Summary

Analysis of experiment results reveals multiple systematic issues in hypothesis generation that lead to ambiguity during instruction following. The primary problems are:

1. **Format Similarity Over Adaptation**: Instructions follow rigid step patterns rather than adapting to example-specific nuances
2. **Missing Implementation Details**: High-level descriptions lack specific algorithmic details needed for deterministic execution
3. **Boundary Case Ambiguity**: Unclear handling of edge cases (divider cells, region boundaries, inheritance)
4. **Coordinate/Region Specification Issues**: Vague region definitions and missing coordinate-level details
5. **Pattern Generation Ambiguity**: Unclear spiral/frame generation algorithms and color ordering rules

## Key Findings from Results Analysis

### Performance Patterns
- **E2 consistently highest scores**: Simplest example, instructions work best
- **E0 middle scores**: Moderate complexity, some ambiguity
- **E1 worst scores**: Most complex, instructions fail to capture nuances
- **Tests worse than held-out**: Instructions generalize poorly to unseen cases

- Reduced context amplifies order effects, suggesting instructions lack robustness
- Full modality (row_col_image) shows minimal order effects, suggesting redundancy helps

## Detailed Problem Analysis

### 1. Format Similarity Over Adaptation

**Problem**: Instructions follow similar step patterns across examples, forcing examples into a template rather than adapting to their specific characteristics.

**Evidence from Hypotheses**:

#### Example: `row_only_ascending/hypothesis.json`

All examples follow identical structure:
- E0: "1. The divider color is red (2). It forms a vertical line at column I and horizontal lines at rows 10 and 16. 2. This creates two vertical columns..."
- E1: "1. The divider color is red (2). It forms a vertical line at column K and a horizontal line at row 13. 2. This creates two vertical columns..."
- E2: "1. The divider color is magenta (6). It forms a vertical line at column G and a horizontal line at row 7. 2. This creates two vertical columns..."

**Issue**: Each example has unique characteristics that aren't emphasized:
- E0 has inheritance patterns that differ from E1
- E1 has yellow background special case
- E2 has asymmetric layout with no inheritance

**Impact**: The instruction-following agent tries to apply the same template to all cases, missing example-specific nuances.

**Recommendation**: 
- Prompt should emphasize: "Each example instruction should adapt to that example's unique characteristics. Don't force examples into identical step patterns."
- Add explicit guidance: "If Example E1 has a special case (like yellow background), make that the PRIMARY focus of E1's instructions, not buried in step 3."

### 2. Missing Implementation Details

**Problem**: Instructions describe WHAT to do but not HOW to do it algorithmically.

**Evidence from Uncertainty Reports**:

From `results.json`, common uncertainties include:
- "The instructions did not specify how to transform non-divider cells that lie on a divider row"
- "The exact pixel-by-pixel stencil for drawing the letters was not provided"
- "The method for creating the 'relative pattern' for tiling was assumed"
- "The spiral termination condition was not explicitly defined"

**Specific Examples**:

#### Spiral Generation Ambiguity
**From `row_image_ascending/hypothesis.json`**:
```
"5. For cases that require a pattern (1.b.i and 1.c), fill the region with concentric rectangular layers. Start with the outermost layer and the first color in the pattern list. For each subsequent inner layer, use the next color, cycling through the list until the region is full."
```

**Missing Details**:
- How to identify "outermost layer" algorithmically
- What happens when region is non-rectangular?
- How to handle odd-shaped regions?
- Termination condition (when to stop)?
- How to handle cells that are equidistant from multiple boundaries?

**From `row_col_image_ascending/hypothesis.json`**:
```
"6. When generating a spiral, determine the color order by the distance of the original seed cells from the partition's primary corner (closest to a grid corner). The outermost layer uses the color of the closest seed."
```

**Missing Details**:
- How to calculate "distance" (Manhattan? Euclidean? Chebyshev?)
- What is "primary corner" exactly? (top-left? closest to A1?)
- How to break ties?
- What if multiple seeds are equidistant?

**Recommendation**:
- Add explicit requirement: "For any algorithmic step, provide pseudocode-level detail: how to iterate, how to calculate, how to break ties, termination conditions."
- Example: Instead of "fill with concentric layers", specify: "1) Initialize set of unfilled cells = all background cells in region. 2) While unfilled cells exist: a) Find all cells on the boundary of unfilled set (cells with at least one neighbor outside unfilled set or outside region). b) Fill these boundary cells with current color. c) Remove filled cells from unfilled set. d) Advance to next color in pattern list (wrap around)."

### 3. Boundary Case Ambiguity

**Problem**: Instructions don't clearly specify how to handle edge cases at region boundaries, divider cells, and inheritance.

**Evidence from Uncertainty Reports**:

#### Divider Cell Handling
Multiple reports mention:
- "The instructions did not specify how to transform non-divider cells that lie on a divider row"
- "The cells in row 13 that were not part of the red (2) divider line (i.e., cells A13:J13) ambiguous"
- "The instruction 'a horizontal line at row 7' was interpreted to mean the entire row acts as a divider"

**From `row_only_ascending/hypothesis.json`**:
```
"1. Identify the single divider color by finding which color forms solid, axis-aligned lines. These divider cells remain unchanged in the output."
```

**Missing Details**:
- What if a divider line is incomplete (only partial row/column)?
- What about cells that are NOT divider color but lie on divider rows/columns?
- How to handle divider intersections?

#### Inheritance Ambiguity
**From `row_only_ascending/hypothesis.json`**:
```
"6. If S is empty: retrieve the last transformation T from the stack. If T was a solid fill, fill R with C_bg. If T was a pattern, apply that same pattern and palette to R."
```

**Uncertainty Reports Show**:
- "The general instructions for inheritance in seedless regions (Step 6) appear to conflict with the specific instructions for the test case"
- "I assumed the rule might be to inherit the most recent multi-color pattern from the column, rather than the most recent transformation of any type"

**Missing Details**:
- What if stack is empty (top region)?
- What if multiple patterns exist above?
- How to "apply same pattern" when region sizes differ?

**Recommendation**:
- Add explicit boundary case handling: "For every rule, specify: what happens at boundaries? what happens with empty sets? what happens with ties?"
- Example: "Divider cells: All cells with color X that form continuous lines spanning full width/height. These cells remain unchanged. Non-divider cells on divider rows: [specific rule]. Empty stack: [specific rule]."

### 4. Coordinate/Region Specification Issues

**Problem**: Region definitions are vague or incomplete, leading to interpretation errors.

**Evidence**:

#### Vague Region Definitions
**From `row_only_ascending/hypothesis.json`**:
```
"VC1 (A1:H9, A11:H15, A17:H20): The top region (A1:H9) has background blue(1) and seeds S={orange(7), teal(8)}."
```

**Issues**:
- Doesn't specify exact boundaries (inclusive/exclusive?)
- Doesn't mention what happens to cells on boundaries
- Doesn't specify how to handle partial regions

**From `row_col_image_ascending/hypothesis.json`**:
```
"Partition P1 (A1:H9) has seeds {orange(7), teal(8)}."
```

**Uncertainty Reports Show**:
- "The instructions specify regions like 'A14:J20', which leaves the cells in row 13, columns A-J (A13:J13) undefined"
- "The instruction for region A10:G15 listed more frame colors than could fit in the region's dimensions"

**Recommendation**:
- Require explicit coordinate specifications: "For each region, specify: exact inclusive boundaries (e.g., 'rows 1-9 inclusive, columns A-H inclusive'), what cells are included/excluded, how boundary cells are handled."
- Add validation: "Verify that all cells in the grid are accounted for in exactly one region (or explicitly marked as dividers/unchanged)."

### 5. Pattern Generation Ambiguity

**Problem**: Spiral/frame generation algorithms are underspecified.

**Evidence**:

#### Color Ordering Ambiguity
**From `row_image_ascending/hypothesis.json`**:
```
"Order the seeds by their position in the grid (top-to-bottom, left-to-right)."
```

**From `row_col_image_ascending/hypothesis.json`**:
```
"determine the color order by the distance of the original seed cells from the partition's primary corner"
```

**Uncertainty Reports Show**:
- "The rule for ordering seed colors when their Manhattan distances are equal is not specified"
- "A consistent tie-breaking rule could not be determined from the examples"

**Missing Details**:
- Exact tie-breaking rules
- Distance calculation method
- Primary corner definition

#### Frame/Spiral Generation
**From `row_only_ascending/hypothesis.json`**:
```
"9. If |P|=2, P={c1, c2}, the transformation is alternating nested frames. If 7 is in P, the outermost frame is color 7. Otherwise, the outermost frame is max(c1, c2). The next frame is the other color, and so on."
```

**Uncertainty Reports Show**:
- "The instruction for region A10:G15 listed more frame colors than could fit in the region's dimensions"
- "I assumed this meant to use as many frames as possible from the list, in the given order"

**Missing Details**:
- How thick are frames? (1 pixel? variable?)
- How to handle regions too small for multiple frames?
- Termination condition for frame generation?

**Recommendation**:
- Require algorithmic specification: "For pattern generation, provide: frame thickness, iteration algorithm, termination condition, tie-breaking rules, edge case handling."
- Example: "Frames are 1-pixel thick. Generate frames by: 1) Start with outer boundary. 2) Fill with color 1. 3) Shrink boundary by 1 pixel on all sides. 4) Fill with color 2. 5) Repeat until remaining area is 2x2 or smaller, then fill with final color."

### 6. Example-Specific Nuances Not Emphasized

**Problem**: Critical example-specific rules are buried in generic instructions rather than highlighted.

**Evidence**:

#### Yellow Background Special Case
**From `row_image_ascending/hypothesis.json`**:
```
"b. If there is 1 seed:
   i. If the 'fillable' color is yellow (4), set the pattern colors to be the seed color and yellow (4).
   ii. Otherwise, flood-fill the entire region with the seed color."
```

**Issue**: This critical special case is buried in a sub-bullet. E1's instructions should START with this special case, not hide it.

**From `row_only_ascending/hypothesis.json`**:
```
"3. VC1 (A1:J12, A14:J20): The top region (A1:J12) has background yellow(4) and one seed S={green(3)}. Because the background is yellow(4), the palette becomes P={3,4}."
```

**Better Approach**: E1 instructions should start with: "**CRITICAL**: This example demonstrates the yellow background special case. When background is yellow(4) and there is 1 seed, use pattern instead of flood-fill..."

**Recommendation**:
- Prompt should require: "For each example, identify the PRIMARY unique characteristic that distinguishes it from others. Make that the focus of that example's instructions."
- Add: "If an example demonstrates a special case or exception, that exception should be the FIRST thing mentioned in that example's instructions."

### 7. Test Case Instructions Too Generic

**Problem**: Test case instructions follow the same template as examples, missing test-specific details.

**Evidence**:

**From `row_only_ascending/hypothesis.json`**:
```
"T0": "1. The divider color is teal (8). It forms vertical lines at columns L and U, and horizontal lines at rows 8, 11, 15, 26. 2. This creates three vertical columns of regions. 3. VC1 (A-K): The top region has seeds {1,2,6}, generating nested frames..."
```

**Issues**:
- Doesn't specify exact region boundaries
- Doesn't mention test-specific complications
- Follows same template as examples

**Uncertainty Reports Show**:
- "The instructions for the test case contained several points of ambiguity and contradiction with the visual input grid"
- "The test case instructions specified seed colors for three regions that were not present in the input grid for those regions"

**Recommendation**:
- Require test-specific detail: "For test cases, provide: exact coordinates for all regions, explicit seed locations, any test-specific complications or edge cases, verification steps."
- Add: "Test case instructions should be MORE detailed than example instructions, not less, because there are no visual examples to reference."

## Prompt Engineering Recommendations

### Current Prompt Issues

**From `hypothesis_fast_nodes.py`**:
- Line 257: "**5-year-old level of detail**: You should document the puzzle in a way that a 5-year-old can understand and solve the puzzle without trying to discover any thing by themselves."

**Problem**: This guidance is too vague. "5-year-old level" doesn't translate to "algorithmic detail."

**From `models.py`**:
- Line 159: "HOW to execute the general algorithm for training example {i} (E{i}), listing concrete steps with actual counts/coordinates/colors derived from that example."

**Problem**: "Concrete steps" doesn't guarantee algorithmic detail. Steps can be concrete but still ambiguous.

### Recommended Prompt Additions

1. **Algorithmic Completeness Requirement**:
```
For every transformation step, provide algorithmic detail sufficient for deterministic implementation:
- Iteration methods (how to iterate through cells/regions)
- Calculation formulas (how to compute distances, areas, etc.)
- Termination conditions (when to stop loops/recursion)
- Tie-breaking rules (how to handle equal values/distances)
- Edge case handling (what happens at boundaries, with empty sets, etc.)
```

2. **Example Adaptation Requirement**:
```
Each example instruction should adapt to that example's unique characteristics:
- Identify the PRIMARY distinguishing feature of each example
- Make that feature the focus of that example's instructions
- Don't force all examples into identical step patterns
- If an example demonstrates a special case, that case should be emphasized, not buried
```

3. **Boundary Case Specification**:
```
For every rule involving regions, boundaries, or partitions:
- Specify exact inclusive/exclusive boundaries
- Define how boundary cells are handled
- Specify what happens to cells not explicitly mentioned
- Verify all cells are accounted for
```

4. **Pattern Generation Detail**:
```
For spiral/frame/pattern generation:
- Specify frame thickness (pixels)
- Provide step-by-step iteration algorithm
- Define termination condition
- Specify how to handle non-rectangular regions
- Provide tie-breaking rules for color ordering
```

5. **Test Case Detail Requirement**:
```
Test case instructions must be MORE detailed than example instructions:
- Provide exact coordinates for all regions
- Specify exact seed locations (not just colors)
- Mention any test-specific complications
- Include verification steps
- Don't assume visual examples are available
```

## Impact on Follow Instructions

**From `follow_instructions.py`**:
- The agent receives instructions and tries to follow them deterministically
- When instructions are ambiguous, the agent must make assumptions
- These assumptions are documented in "uncertainty" field
- High uncertainty correlates with lower scores

**Key Issues**:
1. **Ambiguity leads to interpretation errors**: Different agents interpret ambiguous instructions differently
2. **Missing details force assumptions**: Agents must guess implementation details
3. **Format similarity causes template matching**: Agents try to match patterns rather than understand logic
4. **Boundary cases cause failures**: Edge cases not specified lead to errors

## Recommendations Summary

### Immediate Actions

1. **Update Prompt in `hypothesis_fast_nodes.py`**:
   - Add explicit algorithmic detail requirements
   - Emphasize example-specific adaptation
   - Require boundary case specifications

2. **Update Model Descriptions in `models.py`**:
   - Change "concrete steps" to "algorithmic pseudocode-level steps"
   - Add requirement for tie-breaking rules
   - Require explicit coordinate specifications

3. **Add Validation Checks**:
   - Verify all cells are accounted for
   - Check for ambiguous region definitions
   - Validate algorithmic completeness

### Long-term Improvements

1. **Iterative Refinement**: Use uncertainty reports to refine instructions
2. **Example-Specific Templates**: Different instruction templates for different example types
3. **Algorithmic Verification**: Test instructions can be executed deterministically
4. **Boundary Case Testing**: Explicitly test edge cases during instruction generation

## Conclusion

The primary issue is that instructions prioritize format consistency over algorithmic completeness and example-specific adaptation. This leads to ambiguity during instruction following, causing lower scores especially on complex examples and test cases.

The solution requires:
1. **More specific prompt guidance** emphasizing algorithmic detail
2. **Example-adaptive instructions** that highlight unique characteristics
3. **Explicit boundary case handling** for all rules
4. **Test-specific detail** beyond example templates

By addressing these issues, we can improve instruction quality and reduce ambiguity during the follow-instructions phase.

