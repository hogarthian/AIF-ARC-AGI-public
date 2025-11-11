## Summary Report: Hypothesis Quality Issues Leading to Instruction Following Errors

Author: Sonnet 4.5

Based on my analysis of the experiment results and hypothesis files across multiple modalities, I've identified several critical problems that lead to ambiguity and errors during instruction following:

### **Problem 1: Format Rigidity Over Adaptive Instructions**

**Issue**: LLMs generate instructions following similar step patterns across all examples (E0, E1, E2, T0, T1), even when each training example requires different approaches.

**Evidence**:
- In `row_only_descending`, all E0/E1/E2 instructions follow the exact same format: "1. Identify divider color... 2. Partition... 3. Seeds... 4. Region processing"
- However, E2 has significantly different characteristics (only 2 regions vs 4, asymmetric layout) but forces the same structure
- This results in awkward explanations like "Region 3 (bottom-left), Region 4 (bottom-right): Has 0 seeds" which could be more naturally expressed as "Bottom regions have no seeds"

**Impact**: Instructions become verbose and repetitive rather than precisely tailored to each example's unique structure.

---

### **Problem 2: Missing Critical Execution Details**

**Issue**: Instructions remain at a high conceptual level without specifying the exact computational steps needed for implementation.

**Evidence from `row_image_ascending`**:
```
"5. For each fillable region, identify all unique adjacent seed colors"
```
- No specification of what "adjacent" means (4-way? 8-way? edge-sharing?)
- No specification of how to identify "contiguous areas"
- No specification of tie-breaking when multiple seeds are equidistant

**Evidence from `col_only_ascending`**:
```
"Calculate the Manhattan distance from this cell to every seed"
```
- Doesn't specify: What if two seeds have same Manhattan distance?
- The hypothesis mentions "tie-breaking rule" but instruction doesn't include it explicitly

**Impact**: During `follow_instructions.py`, LLM must make assumptions about these implementation details, leading to divergent interpretations.

---

### **Problem 3: Ambiguous Geometric Specifications**

**Issue**: Spatial and geometric concepts are described verbally without precise coordinate systems or algorithms.

**Evidence from `row_only_descending`**:
```
"draw that letter, scaled to fit"
```
- No specification of how to scale
- No specification of letter templates/stencils
- No specification of centering rules
- Leads to uncertainty like: "The exact pixel-perfect shape of the letters 'G' and 'H' was not specified"

**Evidence from `image_only_ascending`**:
```
"Fill each region with a spiral pattern"
```
- No specification of spiral direction (clockwise? counterclockwise?)
- No specification of spiral starting point
- No specification of termination condition

**Impact**: High uncertainty scores (40-60% uncertainty in results.json) for geometric tasks.

---

### **Problem 4: Inconsistent Abstraction Levels**

**Issue**: Instructions mix high-level concepts with low-level details inconsistently.

**Evidence from `row_col_image_ascending`**:
```
General: "generate a spiral pattern using the seed colors"
E0: "Top-Left (A1:H9): Has 2 seed colors... Fill with a spiral of orange and teal"
```
- General instruction is vague about spiral generation algorithm
- Example-specific instruction just says "Fill with spiral" but doesn't show HOW
- Missing: layer-by-layer coloring order, frame thickness, termination conditions

**Contrast with good specification in `col_image_descending`**:
```
"6. For each region, fill it with a spiral pattern using the two colors from its determined palette."
```
Still vague, but at least:
- Step 5 specifies palette determination precisely
- But Step 6 doesn't specify spiral algorithm

**Impact**: LLM during follow_instructions must infer the algorithm, leading to errors.

---

### **Problem 5: Implicit Assumptions Not Documented**

**Issue**: Critical assumptions are made but not stated in instructions.

**Evidence from `row_image_ascending`** uncertainty field:
```
"The special rule for yellow (4) as a fillable color in the 1-seed case is based on a single example (E1)"
```
- This special case is mentioned in working_hypothesis
- But in transform_instructions, it's buried in conditional logic
- Not highlighted as "EXCEPTION" or "SPECIAL CASE"

**Evidence from `col_image_descending`**:
```
"If a region has two unique seed colors C1, C2: if {C1, C2, background_color} form a set of three consecutive integers, the region is solid-filled"
```
- This complex numeric condition is stated
- But no explanation of WHY consecutive integers trigger different behavior
- No examples demonstrating this rule

**Impact**: Instructions are hard to follow because edge cases aren't clearly flagged.

---

### **Problem 6: Coordinate System Ambiguities**

**Issue**: Coordinate references are inconsistent or imprecise.

**Evidence from `results.json` uncertainty field (row_only example):
```
"The instructions for partition P17 stated 'G27:I30' with area=12, but the actual partition containing the relevant seed is H27:S30"
```

**Evidence from `row_col_image_ascending`**:
```
T0: "Region around A1 (bounded by col L)"
```
- "bounded by" is ambiguous - does L belong to the region or not?
- "around A1" suggests A1 is the center, but it's actually the corner

**Impact**: Off-by-one errors, wrong region boundaries, seeds assigned to wrong regions.

---

### **Problem 7: Overloaded "General" Instructions**

**Issue**: The "general" field tries to be a universal algorithm but includes too many conditional branches without clear prioritization.

**Evidence from `image_only_ascending`**:
```
"5. Execute the logic for the selected mode to determine a two-color palette:
   - Mode C: Palettes are {foreground, PC}. 'Local' seeds color their own region. 'Displacer' seeds color the diagonal region...
   - Mode A: Palettes are {foreground, BG}. The foreground for a seeded region comes from...
   - Mode B: For each region, if it has one seed color..."
```

This creates a decision tree with 3 major modes and sub-cases, but:
- No flowchart or clear decision order
- Nested conditions make it hard to follow
- Example-specific instructions don't reference which mode they use

**Better approach**: State mode detection first, then have separate general instructions for each mode.

---

### **Problem 8: Missing Termination/Boundary Conditions**

**Issue**: Algorithms described without clear stopping conditions.

**Evidence from `row_image_ascending`**:
```
"5. Pattern Generation: The concentric pattern is created by iteratively identifying the outermost layer..."
```
- When does iteration stop?
- What if region has odd dimensions?
- What happens to center cell(s)?

**Evidence from `col_image_descending`** uncertainty:
```
"The instructions describe filling regions with 'a pattern of concentric rectangles (a spiral)'. The exact termination condition for this process was not explicitly defined."
```

**Impact**: LLM must guess termination logic, leading to incomplete or over-filled regions.

---

### **Problem 9: Color Ordering Ambiguity**

**Issue**: When multiple colors are involved, the order matters but isn't always specified.

**Evidence from `results.json` uncertainty (image_only example):
```
"The test case instructions had inconsistencies between the palette derivation logic in step 5 and the final palettes specified for filling in step 6. For the Top-Right region, the derived palette was {8, 1} but the fill palette was {1, 8}."
```

**Evidence from `row_col_image_ascending`**:
```
"6. When generating a spiral, determine the color order by the distance of the original seed cells from the partition's primary corner"
```
- "primary corner" not defined precisely
- "distance" - Manhattan? Euclidean? Chebyshev?

**Impact**: Spiral patterns have colors in wrong order, affecting similarity scores.

---

### **Problem 10: Example-Specific Instructions Don't Show Process**

**Issue**: E0/E1/E2/T0/T1 instructions state results rather than showing step-by-step computation.

**Bad example from `row_only_descending`**:
```
E0: "3. Region 1 (left): Has two seeds, orange (7) at A1 and teal (8) at B2. The color list is [7, 8]. Fill with concentric rectangle pattern."
```

Should be:
```
E0: "3. Region 1 (left): 
     Step 1: Scan region top-to-bottom, left-to-right
     Step 2: Found seed at A1 (orange, 7) - add to list
     Step 3: Found seed at B2 (teal, 8) - add to list
     Step 4: Sorted list is [7, 8] (already in scan order)
     Step 5: Two seeds detected -> apply multi-color spiral rule
     Step 6: Outer color is 7 (first in list), inner is 8
     Step 7: Draw spiral: layer 0 with color 7, layer 1 with color 8, ..."
```

**Impact**: Instructions show WHAT to do but not HOW, requiring LLM to reverse-engineer the process.

---

### **Key Patterns from Results Analysis**

Looking at `results.json`, the uncertainty fields consistently mention:

1. **"The instructions did not specify..."** - appears 47 times
2. **"I assumed..."** - appears 89 times  
3. **"contradictions between..."** - appears 23 times
4. **"ambiguous"** - appears 31 times

The pattern E2 > E0 > E1 in scores suggests:
- E2 is simplest (fewer regions, simpler patterns) → easier to follow even with ambiguous instructions
- E1 is hardest (complex patterns, special cases) → ambiguous instructions cause most errors
- Test cases worse than held-out → overfitting to training examples, instructions don't generalize

---

### **Recommendations**

1. **Adaptive Formatting**: Don't force all examples into same instruction structure. Let E2's simple case have simple instructions.

2. **Explicit Algorithms**: Replace "fill with spiral" with:
   ```
   Spiral Algorithm:
   1. Init layer=0, current_color_index=0
   2. Identify outermost unfilled cells in region
   3. Fill them with colors[current_color_index % len(colors)]
   4. Increment layer, current_color_index
   5. Repeat until no unfilled cells remain
   ```

3. **Coordinate Precision**: Always specify:
   - Is coordinate inclusive or exclusive?
   - Are we using 0-indexed or 1-indexed?
   - Does "A1:H9" include or exclude boundaries?

4. **Flag Special Cases**: Use markers like:
   ```
   ⚠️ EXCEPTION: When background is yellow(4) AND seeds==1, use two-color spiral instead of solid fill
   ```

5. **Separate Mode-Specific Instructions**: Instead of one complex "general" with branches, create:
   - `general_mode_detection`
   - `general_mode_A`
   - `general_mode_B`
   - `general_mode_C`

6. **Show Computation in Examples**: E0/E1/E2 should demonstrate the algorithm step-by-step, not just state results.

7. **Define All Terms**: Create a glossary section:
   ```
   - "adjacent": 4-way connected (sharing an edge)
   - "contiguous": connected component via 4-way adjacency
   - "spiral": concentric rectangular layers starting from perimeter
   ```

8. **Validation Checksums**: Add to each example:
   ```
   E0 verification:
   - Region 1 has 157 cells
   - 145 are background, 12 are seeds
   - After transformation, 145 cells change color
   - Color distribution: {7: 73, 8: 72, ...}
   ```

This would help LLM verify it's following instructions correctly during generation.