Author: GPT-5 High

I reviewed the experiment results and all hypothesis.json sets across modalities, plus the hypothesis-generation and follow-instructions code. Below is a concise problem report highlighting ambiguity sources that can break follow_instructions and lead to errors.

### Where our code expects determinism but hypotheses fall short

```246:259:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/src/nodes/hypothesis_fast_nodes.py
## Critical Requirement: Complete Documentation in Output
...
- **Complete information**: Someone reading only your working_hypothesis and instructions should be able to solve the puzzle WITHOUT seeing the other training examples
- **5-year-old level of detail**: You should document the puzzle in a way that a 5-year-old can understand and solve the puzzle without trying to discover any thing by themselves.
```

```156:161:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/src/nodes/models.py
transform_instructions: Dict[str, str] = Field(
    description="A dictionary with keys: 'general' = WHAT to do ... 'E{N}' and 'T{N}' = HOW ... Use numbered steps, no placeholders, and make all values explicit. ..."
)
```

```210:213:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/src/utils/follow_instructions.py
Output your response in JSON format with:
- "grid" field: The output grid as a list of lists of integers
- "uncertainty" field: A string ...
```

And we saw repeated structured-output failures in results (rectangular-grid violations):

```24:31:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/results.json
"error-fixed": "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (20,) + inhomogeneous part."
```

### Summary of ambiguity/fragility patterns found

- Seed ordering rules conflict across hypotheses
  - Row-major vs column-first vs Manhattan/closest-corner ordering, and sometimes special rules by background proximity. This inconsistency guarantees different color sequences for spirals/frames across files.
    - Column-first sorting used here:

```2:9:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_image_descending/hypothesis.json
... N>1 ... colors ... sorted ... (sorted by column, then row).
```

    - Row-major used elsewhere:

```4:5:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_image_ascending/hypothesis.json
... Sort them by their first appearance (row-major order) ...
```

- Special-case “yellow background” varies by modality
  - Single-seed on yellow sometimes becomes alternating spiral; in others the spiral is “drawn on top” leaving gaps; elsewhere it’s area-threshold gated.
    - Yellow exception:

```2:12:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_only_ascending/hypothesis.json
... exception ... if ... one seed color and the region's background is yellow (4) ...
```

    - “On yellow leave gaps”:

```1:7:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_image_descending/hypothesis.json
... if the background color is yellow (4) ... spiral lines are drawn on top ... leaving the space between the spiral lines yellow.
```

    - Area threshold (40 cells) determining solid vs spiral:

```1:11:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_image_ascending/hypothesis.json
... If the area is greater than 40 cells, a spiral ... If the area is 40 cells or less, flood-fill ...
```

- Boundary/separator/static-color detection is inconsistent
  - “Largest connected non-background”, “first predominantly single-color column”, “red/magenta are always static.”
    - “Largest connected non-background”:

```2:6:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_image_descending/hypothesis.json
... separator color is the color of the largest connected non-background object ...
```

    - “First predominant single-color column”:

```3:5:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_image_ascending/hypothesis.json
... find the first column that is predominantly filled with a single, non-background color. This is the separator ...
```

    - “Static=red/magenta” assumption:

```2:4:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/col_only_ascending/hypothesis.json
... Red (2) and magenta (6) are static colors that act as barriers ...
```

- Spiral specification is under-defined and varies
  - No single canonical: “concentric rectangles”, “gapped over yellow”, “recursive frames”, termination not standardized (when to fill the center; gapped vs full).
    - Gapped on yellow:

```1:7:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_image_descending/hypothesis.json
... if the background color is yellow (4), the spiral lines are drawn on top ... leaving ... yellow.
```

    - Recursion left undefined for >2 seeds:

```11:12:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/col_image_descending/hypothesis.json
... more than two seed colors, a complex recursive filling occurs, which is not fully determined by the examples ...
```

- Empty-region inheritance is not unified
  - Some copy “from above,” others “from symmetric counterpart,” others “remain unchanged.”
    - Symmetry-based inheritance:

```1:9:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_image_ascending/hypothesis.json
... If symmetric ... partitions without seeds will inherit ... Otherwise remain unchanged.
```

    - Copy from nearest non-empty above:

```1:11:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_image_descending/hypothesis.json
... empty region copies from the first non-empty region directly above it ...
```

- Letters/glyphs under-specified
  - “Assume standard, centered, block letters” or “1 vs H” in narrow regions; stencil/scaling/position rules missing → non-deterministic outcomes.
    - Assumed stencils:

```11:12:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_image_ascending/hypothesis.json
... assuming standard, centered, block-style capital letters that scale to fit ...
```

    - Inferred glyph rules:

```11:11:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_descending/hypothesis.json
... exact rules for drawing the number glyphs ... inferred ... assumed standard block-style ...
```

- Conflicting precedence between “general” vs “test” vs “example” instructions
  - Many hypotheses explicitly override general rules in E/T steps, without a single global precedence contract; follow_instructions is told to “apply test precisely” but test texts still depend on general rules in places → model can choose inconsistently.

- Geometric vagueness and placeholders
  - “Approximate” regions, “conceptual cross,” “predominantly filled column,” bounding boxes misaligned to barriers.
    - “Approx.” region:

```8:9:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/row_col_ascending/hypothesis.json
... teal (8) outline of a 'B' (approx. R8:X15) ...
```

- Mode heuristics that are ad-hoc and hard to reproduce
  - E.g., image_only_ascending “Mode A/B/C” with new color generation (max+1) and “displacer” seeds; hard to follow deterministically.

```1:6:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/image_only_ascending/hypothesis.json
... Mode C ... Mode A ... Mode B ... generate new colors ... displacer seeds ...
```

- 4-way vs 8-way adjacency ambiguities for seed detection; not standardized
  - Some say “adjacent 8-way”, others treat seeds as any non-background cell.

- Tie-breakers missing or vary
  - Some define row-major; others Manhattan-distance-based outer color; others corner proximity; many leave ties unspecified in uncertainties.

- Tiling/resize semantics unspecified
  - When “copy pattern” or “tile bounding box” to differently sized regions, cropping/tiling direction/phase is left unspecified.

### Observed score pattern (E2 > E0 > E1 and test < held-out): why it tracks these issues

- E2 examples tend to be simpler and fit more of the ad-hoc rules (e.g., fewer seeds, clear partitions), so underspecified pieces don’t break as often.
- E1 heavily exercises yellow special-casing and mixed rules (gapped spirals, multi-seed complexities), the most inconsistent area across hypotheses; thus worse performance.
- Test cases deviate more (partial separators, unusual seeds, shape/glyph uncertainties), exposing missing “explicitness,” so test < held-out.

### Impact on follow_instructions

- Non-determinism from the above forces the model to “assume,” often producing grids with inconsistent row lengths or partially specified regions, causing structured-output validation to fail:

```439:477:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/src/nodes/models.py
... Grid has inconsistent row lengths ... All rows must have the same length.
```

And observed in results:

```24:31:/home/bwen/.cursor/worktrees/arc-aif-agi/Ar4kA/modality_experiment_results/13e47133/20251105_1142/results.json
... inhomogeneous shape after 1 dimensions ...
```

### Concrete examples per modality (representative)

- Row-only: “inheritance in columns” vs “special-case 7 outermost frame,” plus yellow special-case. Conflicts with other modalities’ seed ordering and outer-color choices.
- Row-image: single-seed adjacency → flood vs spiral; elsewhere single-seed yellow special-case; conflicting sorting (column-first vs row-first).
- Row-col-image: area-threshold vs inheritance vs symmetric copying; spiral “on top” vs full; no unified tie-breakers.
- Col-only / Image-only / Col-image: Voronoi-like rules vs seed-ranking numerics vs recursive frames; “complex case” undefined; inconsistent “boundary/static” definitions.

### Minimal remediation to remove ambiguity (focused, actionable)

- Unify global glossary and detection rules (and enforce across all modalities)
  - Separator/boundary/static: choose one term; define detection deterministically (e.g., any full-span 1-pixel-wide line; if multiple, pick the color with maximum total span; tie-break by color index).
  - Seed detection: choose 4-way or 8-way, and say so everywhere.
  - Seed ordering: row-major position order only, always. No “column-first” or “corner-closest” variants.

- Canonical spiral spec (one and only one)
  - Shape: concentric 1-pixel rings (no gaps). Termination: when inner w≤2 or h≤2, fill remainder with current color. Outer color = first color in ordered palette. No background “gaps on yellow.”
  - If you need “gapped-on-yellow”, make it a per-problem flag emitted explicitly, never inferred from BG=4.

- One precedence policy
  - “Specific Test instructions override Example, which override General,” and require hypotheses to state each override explicitly in the T steps.

- Ban vague geometry
  - Replace “approx.” / “predominantly column” with exact coordinates and thresholds; if auto-detection is required, state the algorithm and tie-breakers.

- No “complex case” placeholders
  - If >2 seeds: specify the exact frame recursion (layer order, colors, termination). Otherwise explicitly say “not supported” and fall back to a deterministic simpler rule.

- Letters/glyphs
  - Either: provide explicit stencils and placement (width/height, padding, stroke thickness) or remove the letter system and rewrite instructions into pattern primitives (frames/spirals). “Assume standard letters” is not acceptable for deterministic generation.

- Tiling/cropping policy
  - If “copy/tiling pattern” into a different-sized region, specify modulo tiling in both axes, top-left anchored, with right/bottom cropping.

- Make E/T steps fully explicit
  - Every E{n}/T{n} must include:
    - separator_color, background_color
    - region list with exact boxes
    - seed list (coords, colors)
    - chosen palette (ordered)
    - spiral start color and termination
    - inheritance/copy policy
    - ties resolved by row-major order only
  - No contradictions (e.g., do not mix “conceptual cross” with literal partitions).

These changes align with the code’s structured-output expectations and the “no placeholders, all values explicit” contract.

Brief status: I read all hypothesis.jsons, plus the generation/evaluation code. Next I can propose a tightened template for transform_instructions if you want me to implement it in the prompt model descriptions.

- E2 scores higher because its cases fit simpler, more consistently documented rules; E1 stresses the mismatches (yellow background and multi-seed complexities).
- The main fix is to standardize definitions (separator, seeds, ordering), spiral spec, tie-breakers, and eliminate vagueness/overrides in E/T steps so follow_instructions can apply them deterministically.