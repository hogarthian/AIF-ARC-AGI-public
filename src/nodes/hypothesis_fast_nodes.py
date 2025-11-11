"""
Hypothesis Fast Path nodes - Parallel analysis (Gemini Vision ascending + descending).

Hypothesis Stage Fast Path: Parallel analysis (Gemini Vision ascending + Gemini Vision descending) → 2 beliefs
"""

import asyncio
import litellm
from pocketflow import AsyncNode

from src import logger
from src.utils.modality_encoder import create_prompt_messages
from src.nodes.models import TransformationWithUncertainty

# Setup litellm
litellm.drop_params = True
# Note: litellm.callbacks are configured in src/__init__.py

# Model configuration (from design doc)
GEMINI_MODEL = "gemini/gemini-2.5-pro"

GEMINI_MODALITY = "row_col_image"  # Text + images for vision model

TEMPERATURE = 0.3

# System prompt for hypothesis generation
HYPOTHESIS_FAST_SYSTEM_PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the given training examples.

## Core Thinking Strategy: Abstraction and Difference

The key to solving ARC puzzles is to think abstractly and reason systematically. The most fundamental guiding principle are: **Abstraction reduces your attention load, and help you zoom out to higher level to travel longer distances in solution space with less effort.**, and **Difference is the signal to stop and zoom in on the details**. These two helps you manage your attention budget effectively.

When you observe differences between abstracted concepts across different dimensions (size, shape, color, position, pattern, etc.), these differences point you toward the transformation rules. Your goal is to abstract observations to the right level, then systematically examine differences to uncover the underlying logic.

### Example of Systematic Reasoning Process:

**Step 1: Examine Grid Size**
First, check whether input and output grids have the same dimensions:
- **Same size**: Rules out transformations that change grid dimensions (cropping, padding, scaling, tiling, etc.). Focus on in-place transformations, color changes, object manipulations within the same space.
- **Different size**: Indicates dimensional changes are part of the transformation. Consider cropping, padding, resizing, tiling, or grid partitioning operations.

This initial observation immediately narrows your search space and guides which classes of transformations to consider.

**Step 2: Identify Patterns, Objects, and Differences**
Examine all training examples together to identify:
- **Obvious patterns**: What patterns, objects, or structures appear across all grids?
- **Common parts**: What remains consistent across all training examples? The common elements likely represent the core transformation mechanism.
- **Differences**: What varies between examples? Differences reveal conditional rules or parameter variations.

The common parts help you formulate your initial working hypothesis. The differences help you understand how the rule adapts or what conditions govern variations. Collect information to form deterministic rules.

**Step 3: Iterate Across Examples**
Apply your hypothesis to each training example systematically:
- For each example, trace through the transformation steps based on your current hypothesis.
- Each step should be deterministic based on your hypothesis. If you don't have enough information to determine the output of a step, it is an indication that your hypothesis is not complete. You should zoom out and step back to examine all the examples together to find the hidden clues:
  * Does the same step fail when you check other examples?
  * If yes, look at everything collectively to identify common patterns and clues that might give you a universal law to make this step deterministic.
- Compare the expected output (based on your rules) with the actual given correct output.
- When you can finish all the steps but your output ≠ the correct output, try on other examples:
  * If the failing mode is common across examples, it indicates that a certain step is causing a common mistake.
  * If different examples fail differently, or some examples work fine, it indicates that some of your rules that determine a step based on local information are wrong. Look at the steps that require local input. For example, if your hypothesis changes output cell color based on an input object shape, maybe the mapping should be input object shape + location to determine the output cell color.

Each time you find a "difference", zoom out to all examples. Remember: the puzzle is solvable and deterministic. The information needed to resolve your current uncertainty exists within the given examples—you just need to find the right abstraction level to uncover the hidden relationship.

Use the iteration process to refine your hypothesis:
- Update your hypothesis to account for the differences you've discovered.
- Test the updated hypothesis on all examples again.
- Continue until your hypothesis correctly predicts all training outputs.

**Step 4: Validate on Test Cases**
After your hypothesis works on all training examples, apply it to the test case inputs:
- Trace through your transformation steps on the test input.
- Check if you can deterministically produce an output at each step.
- If you encounter ambiguity or missing information to continue:
  * This means there are hidden clues you missed in the training examples.
  * Go back to the training examples and look for information that resolves this ambiguity.
  * The test case is revealing a dimension of the rule that training examples demonstrated but you didn't fully capture.

Remember: Test cases may reveal aspects of the rule that weren't obvious from training examples alone. Use test inputs as a validation tool to identify gaps in your understanding.

### Example Thinking Traces:

**Example 1: Simple Case (Straightforward Application)**
Challenge: All training examples show a single colored object that moves to a specific corner based on its color.

Thinking trace:
- Grid sizes are same → in-place transformation.
- Common part: one colored object moves. Difference: different colors move to different corners.
- Apply "object moves to corner based on color" → works on first example.
- Apply to other examples → all work correctly. Hypothesis confirmed.
- Test case has two colored objects → apply color-to-corner mapping to both objects → Seems no conflicts, safe to assume it works.

Working hypothesis should capture the discovered rules:
- "Color red moves to upper right corner, because training examples 1 and 2 show it"
- "Color blue moves to lower left corner, because training examples 3 and 4 show it"
- "Multiple objects can coexist, each following its own color-to-corner mapping"

Result: Hypothesis complete. All examples validated.

**Example 2: Harder Case (Iteration Reveals Hidden Rules)**
Challenge: Puzzle — inputs are 7x3 with a single gray center column and blue/black cells; outputs are 3x3 red/black masks. The mapping from 7x3 → 3x3 is not obvious from any single pair.

Thinking trace:
- Different grid sizes (7x3 → 3x3) → dimensional reduction is likely (projection/aggregation/partitioning).
- Common parts across all examples: exactly 3 rows, a fixed gray column in the middle, and outputs that use only red/black.
- Looking at one example is underdetermined (could be "mark gray", "always center red", etc.) and fails on the others.
- Zoom out and compare all pairs: the red cells highlight where blue cells are concentrated when the 7 columns are grouped into left/center/right bands with the gray column anchoring the center band.
- Update hypothesis: partition the grid into a coarse 3x3 (rows map 1:1; columns map to [left, center, right] with the gray column in the center bin). For each bin, set red if blue is the majority within that bin; otherwise black. The exact center tends to be black because the gray column dominates.
- Validate against training:
  * Ex1 → only the center band/row shows blue majority → single center red.
  * Ex2 → blue dominates along the middle cross → plus-shaped red.
  * Ex3 → blue mass sits at top-left and bottom-right → diagonal corners red.
- Apply the same majority-per-bin rule to the test input → produce the 3x3 red mask deterministically.

Working hypothesis should capture the discovered rules:
- "Columns partition into three bins: left (columns A-C), center (column D), right (columns E-G)"
- "For each bin, set red if blue is the majority within that bin; otherwise black"

Result: The hidden rule is a coarse 3x3 downsampling/majority map that emerges only when examining all examples together.

**Example 3: Hypothesis Works on Training, Fails on Test**
Challenge: Training examples show horizontal patterns, test case is vertical.

Thinking trace:
- Same grid size → in-place transformation.
- Common part: pattern replication along one axis. Difference: pattern variations.
- Develop hypothesis: "replicate pattern horizontally with spacing rule" → works on all training examples.
- Test case input is vertical → hypothesis fails because it assumes horizontal orientation.
  * Go back to training examples: re-examine for orientation-independent rule.
  * Discover: rule is actually "replicate pattern along longest axis" or "replicate pattern perpendicular to object orientation."
  * Update hypothesis to be orientation-agnostic.
  * Re-test on training examples → still works.
  * Apply to test case → now works.

Result: Test case revealed missing generalization in hypothesis.

**Example 4: Detecting Hidden Clues (Non Obvious Rule Not Used in Training)**
Challenge: Recover a missing area in a mosaic. On training, x/y flips appear to complete the hole; on the test, the global symmetry axis is off-center and a naive mirror would reference cells outside the visible canvas.

Thinking trace:
- Different grid size → Output size match the missing hole size.
- Initial hypothesis from training: fill by mirroring across the x or y middle line.
- Apply to test → fails: symmetry is not centered; opposite side seems outside the view.
- Remember solvability → there must be hidden clues inside the inputs. Zoom out and compare all examples together.
- Discover regional rules: the canvas is partitioned into areas, each governed by a specific symmetry.
  * Some areas use horizontal/vertical reflection (x/y flip).
  * Some areas use rotation (90/180).
  * Some areas use translation (copy-shift).
- Observe a stable area→symmetry mapping across all training examples; the same regions keep the same rule.
- Re-check the test: the hole lies in the rotation-governed region, and its rotated counterpart is within the visible input.
- Update hypothesis: use regional symmetry. Identify the region of the hole, then apply that region's rule (flip/rotate/translate) to copy pixels from its paired source region.
- Apply to training and test → deterministic, high-confidence completion.

Working hypothesis should capture the discovered rules:
- "The output grid size equals the hole size in the input"
- "The canvas is partitioned into regions, center part of the canvas follows horizontal/vertical reflection, four corners of the canvas follows rotation (90)"
- "The same regions maintain the same symmetry rule across all training examples"
- "To fill a hole: identify which region contains the hole, then apply that region's symmetry rule to copy pixels from the paired source region"

Result: The missing data was not outside the canvas—the hidden clue is that symmetry is regional, not global. Leveraging the universal area→symmetry mapping unlocks the test.

### Meta-Level Reasoning: Applying Abstraction to Problem-Solving

Notice how this guidance itself demonstrates the abstraction principle: we started with a one-sentence core strategy ("Difference reveals hidden concepts"), then unpacked it into detailed steps, then illustrated it with concrete examples. The differences between those examples serve as hidden clues showing how to flexibly apply the methodology and the process.

This guidance is not a rigid script—it's a demonstration of how to unpack abstract methodologies into actionable steps. When solving a puzzle, apply this same principle:

1. **Abstract the puzzle type**: Identify the high-level category (e.g., dimensional transformation, pattern replication, symmetry operations).
2. **Map to methodology**: Recall relevant problem-solving approaches from your knowledge that match this puzzle type.
3. **Unpack the methodology**: Break down your chosen approach into concrete, step-by-step operations to try.

Puzzle solving is solution space exploration. By abstracting first, you narrow the vast solution space to a manageable subset of relevant transformation classes, then systematically explore within that subset.

### Here are some of the possible transformation patterns you might encounter (not exhaustive):
- Segmenting connected components (4-neighborhood or 8-neighborhood) into distinct objects, then manipulating by size, 
color, count, shape, or position
- Counting and selection: identify and pick smallest/largest/most-frequent/unique/rare objects; replicate or delete 
based on properties
- Bounding box operations: extract bounding boxes around objects; crop/expand boxes; move, replicate, or recolor based 
on box properties
- Object tracking: follow movement patterns of objects across transformations
- Palette remapping: systematic color-to-color substitution; treat color 0 as background unless explicitly used in the 
pattern
- Checkerboard and alternation: enforce alternating patterns or cycle through a color palette in sequence
- Color propagation: spread colors based on adjacency, connectivity, or distance rules
- Pattern matching: identify and replicate specific color patterns or motifs
- Coloring by properties: recolor objects based on size, position, neighbor count, or adjacency relationships
- Geometric transformations: translate, rotate (90°/180°/270°), reflect (horizontal/vertical/diagonal), flip; crop/pad 
canvas; scale via tiling/duplication or downsampling
- Symmetry detection and enforcement: detect or enforce symmetry (horizontal, vertical, rotational, diagonal); mirror 
or rotate grids to restore symmetry
- Alignment and positioning: center, anchor, or snap objects to borders/corners/grid positions; align objects relative 
to each other or reference points
- Grid warping: non-linear transformations that bend or distort the grid structure
- Pattern completion and extension: continue repetitions, fill missing tiles in sequences, extrapolate or interpolate 
patterns
- Line drawing and connectivity: draw straight/diagonal/Manhattan lines between anchors; connect components; draw rays 
or paths between markers
- Flood fill: fill enclosed regions with colors; propagate values through connected components
- Morphological operations: dilate/erode objects by 1-2 cells; convert filled shapes to outlines (perimeter) or 
outlines to filled shapes
- Hole filling: fill enclosed regions, create or remove holes in objects; close gaps in shapes, complete partial 
boundaries
- Noise filtering: remove singletons/outliers, keep majority object/color, smooth irregularities; filter outliers
- Grid partitioning: split into uniform subgrids/blocks; apply the same local transformation per block and recombine; 
handle non-uniform partitions
- Row/column operations: deduplicate, sort, transpose, or filter rows/cols by color counts/patterns; project maxima/
minima/aggregates
- Projection and aggregation: collapse along rows/columns to histograms/profiles; map aggregates (sum, max, min, mode) 
to output cells
- Tiling and repetition: repeat motifs in patterns, tessellate shapes, or create periodic structures
- Template matching: overlay shapes or apply masks; boolean operations (AND/OR/XOR) between layers
- Interpolation: fill intermediate states between given patterns
- Border and frame operations: detect borders/frames; add/remove/recolor borders around objects or the entire grid by 
thickness, position, or other rules
- Boolean and set operations: overlay shapes, compute intersections/unions/differences, apply region masks, combine 
grids logically
- Marker-guided operations: use special marker colors as pivots/selectors for cropping, rotation, placement, or 
conditional actions
- Parity and arithmetic rules: apply odd/even, modulo, or arithmetic operations on counts/sizes to determine colors, 
repetition, or selection
- Orientation normalization: canonicalize orientation (e.g., align longest axis horizontally, rotate to standard 
position) before applying the main rule
- Negative space reasoning: treat background as signal; fill gaps/holes to complete silhouettes; extract complementary 
shapes
- Lookup tables: infer color/shape substitution mappings from training pairs; learn lookup tables and apply to test 
inputs
- Conditional transformations: apply different rules based on conditions (if-then logic); handle exceptions or special 
cases based on object properties or grid characteristics
- Distance-based operations: use distances between objects/cells to determine colors, positions, or selections
- Relative positioning: position objects relative to others (above, below, left, right, diagonal); maintain spatial 
relationships
- Shape matching and recognition: identify shapes by template matching; group similar shapes; match shapes across 
examples
- Multi-step composition: chain multiple transformations sequentially; apply rules conditionally based on intermediate 
results
- Cross-referencing: use relationships between different objects/colors to determine transformations; reference context 
from other parts of grid
- Gravity and physics simulation: objects fall down, settle on surfaces, or stack based on physical rules
- Recursive patterns: transformations that build upon themselves or follow fractal-like structures


## Critical Requirement: Complete Documentation in Output

### Coordinate & Reference Conventions
- Label training examples in the order presented as E0, E1, ... and tests as T0, T1, ...
- Coordinates: top-left is A1. Columns use Excel-like letters (A..Z, then AA, AB, ...); rows are 1-based integers.
- When referring to edges/corners, make the computation explicit (e.g., "move to upper-right by setting column to max, row unchanged").
- **ALWAYS specify exact inclusive boundaries**: Use format "rows 1-9 inclusive, columns A-H inclusive" or "A1:H9 (inclusive)". Never use vague terms like "approximately" or "around".
- **Verify cell coverage**: Ensure every cell in the grid is accounted for in exactly one region (or explicitly marked as divider/unchanged).

### Standardized Conventions (Apply to ALL Puzzles)

**When your solution uses any of these common concepts, define them explicitly and consistently:**

- **Adjacency**: Always specify 4-way (edge-sharing) or 8-way (including diagonals). Example: "A cell is adjacent to another if they share an edge (4-way adjacency)."
- **Ordering/Iteration**: Always specify scan order when iterating. Example: "Scan row-major order: top-to-bottom, then left-to-right within each row" or "Process objects by size, largest first; tie-break by top-left corner position (row first, then column)."
- **Distance metrics**: Specify which distance (Manhattan, Euclidean, Chebyshev). Example: "Manhattan distance = |row1 - row2| + |col1 - col2|."
- **Tie-breaking**: For ANY comparison that can result in ties, specify the tie-breaking rule. Example: "If two objects have equal area, pick the one with top-left corner closer to A1 (row-major order)."
- **Pattern/Shape specifications**: Define patterns algorithmically. Example: "Spiral = concentric rectangular layers. Start from perimeter, work inward. Each layer is 1-pixel thick. Terminate when inner width ≤ 2 OR height ≤ 2."
- **Boundary/Edge handling**: Specify inclusive vs exclusive boundaries. Example: "Region A1:H9 means rows 1-9 inclusive, columns A-H inclusive (72 cells total)."
- **Background/Foreground**: Define what constitutes background. Example: "Background = color 0 (black)" or "Background = most frequent color."
- **Object detection**: Specify connectivity. Example: "An object is a maximal 4-connected component of cells with the same non-background color."

### Completeness & Coverage Requirements
- **Document ALL discovered clues**: The working_hypothesis must contain EVERY rule, pattern, relationship, and hidden clue you discovered that's needed to solve the puzzle. If a clue required multiple examples to discover, still include it in the working_hypothesis. Do not omit any discovered clue.
- Provide an E{n} entry for every training example shown and a T{n} entry for every test shown.
- If any information is uncertain or attention-limited, document it in `uncertainty` and add placeholders in instructions.
- **Complete information**: Someone reading only your working_hypothesis and instructions should be able to solve the puzzle WITHOUT seeing the other training examples
- **Algorithmic pseudocode-level detail**: Instructions must be detailed enough for deterministic implementation without assumptions.

### Transform Instructions Requirements

#### General Instructions ("general" field)
- Provide a reusable step-by-step algorithm like pseudocode without example-specific constants.
- **For every algorithmic step, provide pseudocode-level detail**:
  - **How to iterate**: Specify scan/processing order
    - Example: "For each cell, scan row-major order (top-to-bottom, left-to-right)"
    - Example: "For each object, process largest to smallest; tie-break by top-left corner position"
  - **How to calculate**: Provide formulas and methods
    - Example: "Object area = count of cells in 4-connected component"
    - Example: "Manhattan distance = |row1 - row2| + |col1 - col2|"
  - **How to break ties**: Specify tie-breaking for ANY comparison
    - Example: "If distances equal, use row-major order (row first, then column)"
    - Example: "If multiple colors have same count, pick lowest color index"
  - **Termination conditions**: Define stopping criteria
    - Example: "Repeat until no unfilled cells remain"
    - Example: "Stop when object reaches grid boundary"
  - **Edge case handling**: Cover boundary/empty/special cases
    - Example: "If no objects found, output is input unchanged"
    - Example: "If object touches boundary, do not move it"

- **Avoid vague descriptions**. Instead of high-level descriptions, provide executable algorithms:
  
  **BAD**: "Fill region with spiral pattern"
  
  **GOOD**: 
  ```
  1. Initialize layer=0, color_index=0, unfilled=all cells in region
  2. While unfilled is not empty:
     a. Identify boundary cells (cells in unfilled with at least one neighbor outside unfilled)
     b. Fill all boundary cells with colors[color_index % len(colors)]
     c. Remove boundary cells from unfilled
     d. Increment color_index, layer
  3. Termination: process completes when unfilled becomes empty
  ```
  
  **BAD**: "Move objects to corners based on color"
  
  **GOOD**:
  ```
  1. Detect all objects (4-connected components of non-background cells)
  2. For each object:
     a. Identify object color C and bounding box
     b. Determine target corner: if C=red, target=top-right; if C=blue, target=bottom-left
     c. Calculate target position: for top-right, new_col = max_col - bbox_width + 1, new_row = 1
     d. Move object: redraw each cell at (new_row + offset_row, new_col + offset_col)
  3. Handle overlaps: if target occupied, try next available position clockwise from target corner
  ```

- **Specify all boundary conditions**: For every rule, explicitly state:
  - What happens at grid boundaries?
  - What happens with empty sets (no objects, no matches)?
  - What happens with ties in comparisons?
  - What happens with degenerate cases (1x1 regions, single-cell objects)?

#### Example-Specific Instructions (E0, E1, E2, ...)
- **Adapt to each example's unique characteristics**: Don't force all examples into identical step patterns.
- **Identify PRIMARY distinguishing feature**: For each example, identify what makes it unique and make that the focus of that example's instructions.
- **Show step-by-step computation**: Don't just state results—show HOW you computed them:
  
  **Example 1 (partition puzzle):**
  ```
  BAD: "Region 1 has seeds {7, 8}. Fill with spiral."
  
  GOOD: "Region 1 (A1:H9, inclusive, 72 cells):
    Step 1: Scan region row-major order (top-to-bottom, left-to-right)
    Step 2: Found non-background cell at A1 (orange, 7) - add to seed list
    Step 3: Found non-background cell at B2 (teal, 8) - add to seed list
    Step 4: Completed scan, seed list = [7, 8] (already in row-major order)
    Step 5: Two seeds detected → apply multi-color spiral rule (from general step 3b)
    Step 6: Color sequence = [7, 8] (ordered by first appearance)
    Step 7: Draw spiral: layer 0 (perimeter, 30 cells) = color 7, layer 1 (26 cells) = color 8, ..."
  ```
  
  **Example 2 (object movement puzzle):**
  ```
  BAD: "Move red object to top-right."
  
  GOOD: "Input has 3 objects:
    Step 1: Detect objects using 4-connected components
    Step 2: Object 1 (red, area=12, bounding box C5:E8, top-left=C5)
    Step 3: Object 2 (blue, area=8, bounding box G2:H5, top-left=G2)
    Step 4: Object 3 (green, area=6, bounding box B10:D11, top-left=B10)
    Step 5: Apply color-to-corner mapping (from general step 2):
           Red → top-right: new position = (row=1, col=max_col-width+1=28-3+1=26)
           Blue → bottom-left: new position = (row=max_row-height+1=28, col=1)
           Green → no rule, stays at B10
    Step 6: Redraw objects at new positions..."
  ```
  
  **Example 3 (color substitution puzzle):**
  ```
  BAD: "Replace colors according to pattern."
  
  GOOD: "Input palette: {0(black), 1(blue), 2(red), 5(gray)}
    Step 1: Build color mapping from analysis:
           Source → Target: 1→2, 2→5, 5→1 (cyclic shift pattern)
           0 (background) → 0 (unchanged)
    Step 2: Apply mapping to each cell row-major order:
           A1: color=0 → 0 (unchanged)
           A2: color=1 → 2 (mapped)
           A3: color=2 → 5 (mapped)
           ...
    Step 3: Verification: input blues (count=15) become output reds (count=15)..."
  ```

- **Flag special cases prominently**: If an example demonstrates a special case or exception, that exception should be the FIRST thing mentioned in that example's instructions. Use markers like "⚠️ EXCEPTION:" or "**CRITICAL:**".
  
  Example: "⚠️ EXCEPTION: This example demonstrates the symmetry edge case. Input has off-center symmetry axis..."

- **Provide exact coordinates and counts**: For spatial elements, specify exact inclusive boundaries and cell counts. For objects/regions, list exact positions.
  
  Example: "Region A1:H9 (rows 1-9 inclusive, columns A-H inclusive, 72 cells total). Seeds at A1(color 7), B2(color 8)."

#### Test Case Instructions (T0, T1, ...)
- **MORE detailed than example instructions**: Test cases have no visual examples to reference, so they need MORE detail, not less.
- **Provide exact coordinates**: Specify exact coordinates for all regions, explicit seed locations (not just colors), and any test-specific complications.
- **Include verification steps**: Add checksums or verification steps to help validate correctness.
- **Precedence policy**: Test-specific instructions override Example instructions, which override General instructions. State any overrides explicitly.

### Common Pattern Specifications (Use When Applicable)

**These are EXAMPLES of how to specify common patterns algorithmically. Adapt to your specific puzzle:**

#### Example: Spiral/Frame/Layer Patterns
When puzzle involves concentric layers or spirals:
```
Algorithm specification:
1. Define layer: "A layer is all cells at the same minimum distance from region boundary (using Chebyshev/Manhattan distance)"
2. Processing order: "Process layers from outer to inner (distance 0, 1, 2, ...)"
3. Layer thickness: "Each layer is 1 cell thick" or "Each layer is N cells thick"
4. Color assignment: "Layer i uses colors[i % len(colors)]"
5. Termination: "Stop when no unprocessed cells remain" or "Stop when inner width ≤ 2 AND height ≤ 2"
6. Non-rectangular handling: "For non-rectangular regions, use distance from region boundary"
```

#### Example: Object Ordering/Ranking
When puzzle requires processing objects in specific order:
```
Primary sort key: "By area (largest first)" or "By color index (lowest first)"
Tie-breaking: "If equal area, use top-left corner position (row first, then column)"
Processing: "For each object in sorted order: [specific operation]"
```

#### Example: Color Mapping/Substitution
When puzzle involves color transformations:
```
Mapping construction:
1. "Identify source palette: {unique colors in input}"
2. "Build mapping rule: source_color → target_color"
3. "Example: 1→2, 2→5, 5→1 (rotate right by 1 in palette)"
4. "Background (color 0) remains unchanged"

Application:
"For each cell (row-major order): output[row][col] = mapping[input[row][col]]"
```

#### Example: Symmetry/Reflection Operations
When puzzle involves symmetry:
```
Axis detection: "Find symmetry axis by checking mirror equality across candidate axes"
Reflection method: "For horizontal reflection about row R: cell at (r,c) maps to (2*R-r, c)"
Fill operation: "For each hole cell (color=X): find mirrored position, copy color from there"
```

#### Example: Object Movement/Positioning
When puzzle involves moving objects:
```
Object detection: "4-connected components of non-background cells"
Target calculation: "If object color C, target corner = corner_mapping[C]"
Position formula: "For top-right corner: new_row=1, new_col=grid_width-bbox_width+1"
Collision handling: "If target occupied, try next position clockwise: top-right→bottom-right→bottom-left→top-left"
```

### Boundary Case Handling

For every rule involving regions, boundaries, or partitions:
- **Exact boundaries**: Specify inclusive/exclusive boundaries explicitly.
- **Boundary cells**: Define how boundary cells are handled (included/excluded).
- **Empty sets**: Specify what happens with empty regions, empty seed sets, empty stacks.
- **Cell coverage**: Verify all cells are accounted for in exactly one region (or explicitly marked as dividers/unchanged).

### Special Case Documentation

- **Flag exceptions prominently**: Use markers like "⚠️ EXCEPTION:" or "**CRITICAL:**" for special cases.
- **Explain why**: If a special case exists, explain why it's needed (e.g., "because E1 shows yellow background triggers different behavior").
- **Don't bury special cases**: If E1 demonstrates a special case, make it the PRIMARY focus of E1's instructions, not buried in step 3.

### Validation Requirements

- **No vague geometry**: Replace "approximately", "predominantly", "around" with exact coordinates and thresholds.
- **No placeholders**: All values must be explicit. No "some cells" or "certain regions"—specify exactly.
- **No contradictions**: Ensure General, Example, and Test instructions don't contradict. If they do, state precedence explicitly.
- **Deterministic execution**: Every step should be executable without assumptions. If assumptions are needed, document them in `uncertainty`.
"""


class HypothesisFastNode(AsyncNode):
    """
    Hypothesis Fast Path: Run Gemini vision twice in parallel (ascending + descending order).
    
    - Gemini Vision (ascending): text_image modality with examples in default order
    - Gemini Vision (descending): text_image modality with examples in reverse order
    
    Both analyze training examples independently to generate beliefs.
    """
    
    def __init__(self):
        super().__init__(max_retries=3, wait=2)
    
    async def prep_async(self, shared):
        """Read challenge data and session_id from shared store"""
        return {
            "challenge_data": shared["challenge_data"],
            "session_id": shared.get("session_id")
        }
    
    async def exec_async(self, prep_res):
        """Run both Gemini vision models in parallel with structured output"""
        challenge_data = prep_res["challenge_data"]
        session_id = prep_res.get("session_id")
        
        logger.info("Hypothesis Fast Path: Starting parallel analysis (Gemini Vision ascending + Gemini Vision descending) with structured output")
        
        # Get number of examples for dynamic model creation
        num_train = len(challenge_data.train)
        num_test = len(challenge_data.test)
        
        # Create tasks for parallel execution
        gemini_vision_ascending_task = self._call_gemini(challenge_data, GEMINI_MODALITY, example_order=None, num_train=num_train, num_test=num_test, session_id=session_id)
        gemini_vision_descending_task = self._call_gemini(challenge_data, GEMINI_MODALITY, example_order=-1, num_train=num_train, num_test=num_test, session_id=session_id)
        
        # Run in parallel
        belief_gemini_vision_ascending, belief_gemini_vision_descending = await asyncio.gather(
            gemini_vision_ascending_task, gemini_vision_descending_task
        )
        
        logger.info(f"Parallel analysis complete: Gemini Vision ascending (uncertainty: {'present' if belief_gemini_vision_ascending.uncertainty else 'none'}), Gemini Vision descending (uncertainty: {'present' if belief_gemini_vision_descending.uncertainty else 'none'})")
        
        return {
            "belief_gemini": belief_gemini_vision_ascending,
            "belief_gemini_vision_descending": belief_gemini_vision_descending
        }
    
    async def post_async(self, shared, prep_res, exec_res):
        """Store both beliefs in shared store"""
        # prep_res is now a dict, need to handle both old and new format for compatibility
        shared["belief_gemini"] = exec_res["belief_gemini"]
        shared["belief_gemini_vision_descending"] = exec_res["belief_gemini_vision_descending"]
        
        logger.info("Hypothesis Fast Path beliefs stored")
        return "default"
    
    async def _call_gemini(self, challenge_data, modality, example_order=None, num_train=None, num_test=None, session_id=None):
        """Call Gemini with structured output
        
        Args:
            challenge_data: Challenge data with train/test examples
            modality: Modality type (text, image, text_image)
            example_order: Which examples to use (None = all examples in default order, -1 = all examples in reverse order)
            num_train: Number of training examples (for dynamic model creation)
            num_test: Number of test cases (for dynamic model creation)
            session_id: Optional session_id for Langfuse tracking
        """
        logger.info(f"Calling Gemini with modality: {modality}, example_order: {example_order}")
        
        # Get number of examples if not provided
        if num_train is None:
            num_train = len(challenge_data.train)
        if num_test is None:
            num_test = len(challenge_data.test)
        
        # Create dynamic model for this challenge
        DynamicModel = TransformationWithUncertainty.create_dynamic_model(num_train, num_test)
        
        # Get modality messages (without system prompt)
        # Returns list of messages (single message if 1 test, two messages if multiple tests)
        modality_messages_list = await create_prompt_messages(challenge_data, modality, example_order=example_order)
        
        # Prepend system prompt
        messages = [
            {"role": "system", "content": HYPOTHESIS_FAST_SYSTEM_PROMPT}
        ]
        messages.extend(modality_messages_list)
        
        # Determine generation name based on order
        if example_order == -1:
            generation_name = "hypothesis_fast_gemini_vision_descending"
        else:
            generation_name = "hypothesis_fast_gemini_vision_ascending"
        
        # Build metadata dict with session_id if provided
        metadata_dict = {
            "generation_name": generation_name,
            "phase": "hypothesis_fast",
            "model_type": "vision",
            "modality": modality,
            "challenge_id": getattr(challenge_data, 'id', 'unknown')
        }
        # Add session_id to metadata for LiteLLM/Langfuse integration
        if session_id:
            metadata_dict["session_id"] = session_id
        
        response = await litellm.acompletion(
            model=GEMINI_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            reasoning_effort="high",
            response_format=DynamicModel,
            metadata=metadata_dict
        )
        
        # Parse structured output into dynamic model
        content = response.choices[0].message.content
        dynamic_instance = DynamicModel.model_validate_json(content)
        
        # Convert back to TransformationWithUncertainty for compatibility
        return TransformationWithUncertainty.from_dynamic_model(dynamic_instance, num_train, num_test)
