# Modality Vision Experiment

This directory contains experiments that test the model's ability to "see" and describe puzzle grids under different input modalities.

## Purpose

The modality vision experiment tests how well the LLM can perceive and describe visual information when presented in different formats:
- **Text formats**: Row-wise, column-wise, ASCII, JSON
- **Image formats**: Various resolutions (14x14, 15x15, 16x16, 17x17, 24x24, 768x768) with and without coordinate annotations

For each modality, the model is asked to provide a detailed description of what it sees, including:
- Objects and shapes
- Locations (using spreadsheet notation like A1, B2)
- Colors and their numeric mappings
- Spatial relationships
- Patterns and symmetries
- Grid structure

## Data Structure

Results are organized by timestamp directories:

- `data/modality_vision_{timestamp}/` - Each experiment run creates a new timestamped directory
  - `summary.json` - Overall experiment summary with metadata
  - `{challenge_id}/` - Per-challenge results
    - `result.json` - Detailed descriptions for input and output grids in all tested modalities

## Running the Experiment

### Basic Usage

```bash
uv run python run_modality_vision_experiment.py --challenge-ids 13e47133 0934a4d8 135a2760 136b0064 142ca369 --rpm 60
```

### Full Options
If you want to try with different challenges:

```bash
uv run python run_modality_vision_experiment.py \
  --challenge-ids <id1> <id2> ... \
  --output-dir data \
  --challenges-file <path_to_challenges.json> \
  --model <model_name> \
  --temperature <temperature> \
  --rpm <rpm_limit> \
  --modality-types <modality1> <modality2> ...
```

### Arguments

- `--challenge-ids`: **Required**. One or more challenge IDs to test
- `--output-dir`: Base output directory (default: `data`). Results saved to `{output_dir}/modality_vision_{timestamp}/`
- `--challenges-file`: Path to challenges JSON file (default: uses `DEFAULT_CHALLENGES_FILE` from data_loader)
- `--model`: Model to use (default: `gemini/gemini-2.5-pro`)
- `--temperature`: Temperature setting (default: `0.3`)
- `--rpm`: Maximum requests per minute for rate limiting (recommended: 60-120 RPM)
- `--modality-types`: Specific modalities to test (default: all). Example: `--modality-types image_768x768 image_16x16`
- `--existing-experiment-dir`: Path to existing experiment directory to append results to (instead of creating new one)

### Supported Modality Types

- `row_only` - Row-wise text format
- `col_only` - Column-wise text format
- `ascii` - ASCII format (space-separated integers per row)
- `json` - Raw JSON format (list of lists)
- `image_14x14` - Image with 14x14 pixels per cell (with coordinates)
- `image_15x15` - Image with 15x15 pixels per cell (with coordinates)
- `image_16x16` - Image with 16x16 pixels per cell (with coordinates)
- `image_17x17` - Image with 17x17 pixels per cell (with coordinates)
- `image_24x24` - Image with 24x24 pixels per cell (with coordinates)
- `image_768x768` - Image optimized for Gemini 768x768 patch (with coordinates)

### Examples

**Test all modalities for 5 challenges:**
```bash
uv run python run_modality_vision_experiment.py \
  --challenge-ids 13e47133 0934a4d8 135a2760 136b0064 142ca369 \
  --output-dir data \
  --rpm 60
```

**Test only high-resolution images:**
```bash
uv run python run_modality_vision_experiment.py \
  --challenge-ids 13e47133 0934a4d8 \
  --modality-types image_768x768 image_24x24 \
  --output-dir data \
  --rpm 60
```

**Append results to existing experiment:**
```bash
uv run python run_modality_vision_experiment.py \
  --challenge-ids 142ca369 \
  --existing-experiment-dir data/modality_vision_20251108_1148 \
  --modality-types image_768x768 \
  --output-dir data \
  --rpm 60
```

## Output Format

Each challenge result (`{challenge_id}/result.json`) contains:

```json
{
  "challenge_id": "13e47133",
  "input_grid": [[...]],
  "output_grid": [[...]],
  "grid_dimensions": {
    "input": {"rows": 10, "cols": 10},
    "output": {"rows": 10, "cols": 10}
  },
  "input_descriptions": {
    "row_only": {"description": "..."},
    "col_only": {"description": "..."},
    "image_16x16": {"description": "..."},
    ...
  },
  "output_descriptions": {
    "row_only": {"description": "..."},
    "col_only": {"description": "..."},
    "image_16x16": {"description": "..."},
    ...
  }
}
```

## Calculating Perception Scores

After running the modality vision experiment, calculating perception accuracy scores involves several steps:

### Overview

The scoring process follows this workflow:

1. **Human creates Ground Truth and Scoring Guide**: For each challenge, a human analyst examines the puzzle input and constructs:
   - **Ground Truth Features**: A detailed specification of all critical features that must be identified (e.g., colored dots, shapes, divider lines, background colors)
   - **Scoring Guide**: Point values for each feature and how penalties are calculated

2. **Extract claims from experiment results**: Use an LLM-assisted script to extract structured claims from the free-text descriptions generated by the experiment. See `example_extract_scoring_data.py` for an example implementation. [IMPORTANT] But we didn't actually use this script in our experiment because creating structured output for each challenge is quite labor-intensive. We actually did the data analysis in Cursor IDE, prompting `Sonnet 4.5` with the following but customize a little bit for each challenge:

```prompt
@appendix2.tex review the Accuracy Analysis part carefully. use @print_modality_description.py to loop through @modality_vision_20251107_2355 (all other)@modality_vision_20251108_1148 (image_24x24-1148)@modality_vision_20251108_1205 (image_24x24-1205)
to print out the description of each modality for each challenge one at a time and read the print out, this is the raw data. use them to review the Accuracy Analysis, don't worry about the scoring yet.  there are a lot of raw data, so you need to do iteratively one by one. don't try to grep/filter/regex/head, just read the raw complete output, and then fill out the appendix, otherwise you will miss info and make mistakes. example call: cd data/modality_vision_20251108_1205 && python3 print_modality_description.py 13e47133 image_24x24 input

carefully review the results of 142ca369, read one and write one, instead of reading everything and then write the document. The current entry all read similar, which is suspicious, check all the features
```

Run this in Cursor a few times alternating between `Sonnet 4.5` and other models with cross exam the results and provide an highly accurate output.

3. **Human review**: Review the extracted claims for accuracy

4. **Calculate final scores**: Copy the reviewed claims into the `modalities_data` field in the scoring notebook `calculate_perception_scores.ipynb` and compute final scores
