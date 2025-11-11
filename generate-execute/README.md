# Modality Experiment Results

This directory contains the experiment results used to generate the violin plot in the paper.

## Data Structure

The experiment results are organized by timestamp directories:

- `20251105_1142/` - Cross-order experiment testing all 7 modalities with both ascending and descending orders
  - Contains `results.json` with all experiment results
  - Contains subdirectories for each modality+order combination (e.g., `row_only_ascending/`, `row_col_image_descending/`)
  - Each subdirectory contains:
    - `hypothesis.json` - Generated hypothesis and reasoning
    - `grids.json` - All input/output grids for held-out and test cases
    - `results.json` - Similarity scores for all cases
    - `grids_reduced.json`, `results_reduced.json` - Reduced context versions

- `20251108_1435/` - JSON-only modality experiment (ascending order)
- `20251108_1444/` - Row+Column+JSON modality experiment (ascending order)
- `20251108_1455/` - Row+Column+JSON+Image modality experiment (ascending order)
- `20251108_1504/` - Image+JSON modality experiment (ascending order)
- `20251108_2058/` - Repeat sampling experiment (4 trials with temperature 0.7) (we didn't report this result in the paper)

## Reproducing the Experiments

**⚠️ Important**: Before running any experiment, use the `--dry-run` flag to preview the number of API calls and understand the experiment configuration. This helps you plan your experiments carefully and avoid unexpected costs.

### Reproducing Paper Results (All Modalities, Both Orders, With Reduced)

To reproduce the complete paper results matching the original experiments:

```bash
# First, preview the experiment with --dry-run
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order both \
  --include-reduced \
  --dry-run

# Then run the actual experiment
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order both \
  --include-reduced \
  --rpm 60
```

This command tests all 11 modality types (row_only, col_only, image_only, row_col, row_image, col_image, row_col_image, json_only, row_col_json, row_col_json_image, image_json) with both ascending and descending example orders, including reduced context versions.

**Note**: 
- This experiment requires significant API calls (242 API calls for challenge 13e47133 with 3 training examples and 1 test case: 11 modalities × 2 orders × (1 hypothesis + 3 held-out + 2 test + 3 held-out reduced + 2 test reduced) = 11 × 2 × 11 = 242 calls). Use `--dry-run` to understand the API calls before running the real experiment.
- Use the `--rpm` flag to enable rate limiting to avoid hitting API rate limits.
- Results are saved to `data/experiment_results/{challenge_id}/{timestamp}/` to match the existing data structure.

### Simplified Version (All Modalities, Ascending Only, No Reduced)

For a faster experiment that tests all modalities but only with ascending order and without reduced versions:

```bash
# First, preview the experiment with --dry-run
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order ascending \
  --dry-run

# Then run the actual experiment
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order ascending \
  --rpm 60
```

This reduces the API calls significantly (66 API calls for challenge 13e47133: 11 modalities × 1 order × (1 hypothesis + 3 held-out + 2 test) = 11 × 6 = 66 calls). Use `--dry-run` to understand the API calls before running the real experiment.

### Testing Specific Modalities

To test only specific modality types (useful for faster iteration or focused experiments):

```bash
# First, preview the experiment with --dry-run
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --modality-types row_only col_only \
  --dry-run

# Then run the actual experiment
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --modality-types row_only col_only \
  --rpm 60
```

This reduces API calls significantly (12 API calls for challenge 13e47133: 2 modalities × 1 order × (1 hypothesis + 3 held-out + 2 test) = 2 × 6 = 12 calls). Use `--dry-run` to understand the API calls before running the real experiment.

**Available modality types**: `row_only`, `col_only`, `image_only`, `row_col`, `row_image`, `col_image`, `row_col_image`, `json_only`, `row_col_json`, `row_col_json_image`, `image_json`

### Cross-Order Experiment (20251105_1142)

This experiment was run using `run_double_modality_experiment.py` with the original 7 modalities:

```bash
# First, preview the experiment with --dry-run
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order both \
  --include-reduced \
  --modality-types row_only col_only image_only row_col row_image col_image row_col_image\
  --dry-run

# Then run the actual experiment
uv run python run_double_modality_experiment.py \
  --challenge-id 13e47133 \
  --output-dir data/experiment_results \
  --order both \
  --include-reduced \
  --modality-types row_only col_only image_only row_col row_image col_image row_col_image\
  --rpm 60
```

### JSON-Based Experiments (20251108_1435, 20251108_1444, 20251108_1455, 20251108_1504)

**Note**: These JSON-based experiments are now included in the comprehensive `run_double_modality_experiment.py` script. However, if you want to run them separately using `run_single_modality_experiment.py`:

```bash
# Preview first with --dry-run, then run all JSON-based modalities sequentially
for modality in json_only row_col_json row_col_json_image image_json; do
  uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type $modality --output-dir data/experiment_results --dry-run
  uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type $modality --output-dir data/experiment_results --rpm 60
done
```

Or run them individually:

```bash
# json_only
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type json_only --output-dir data/experiment_results --rpm 60

# row_col_json
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json --output-dir data/experiment_results --rpm 60

# row_col_json_image
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --output-dir data/experiment_results --rpm 60

# image_json
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type image_json --output-dir data/experiment_results --rpm 60
```

### Repeat Sampling Experiment

This experiment tested output diversity with multiple trials:

```bash
# First, preview the experiment with --dry-run
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --num-trials 4 --temperature 0.7 --output-dir data/experiment_results --dry-run

# Then run the actual experiment
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --num-trials 4 --temperature 0.7 --output-dir data/experiment_results --rpm 60
```

## Reproducing the Violin Plot

Use the `reproduce_violin_plot.ipynb` notebook to generate the violin plot from the paper. The notebook loads results from the experiment directories and creates the plot showing score distributions by modality.

## Notes

- The cross-order experiment (20251105_1142) tested both ascending and descending orders to investigate order effects
- After concluding that order doesn't significantly affect quality, subsequent experiments (20251108_*) only used ascending order
- **`run_double_modality_experiment.py`** now includes all 11 modalities (7 original + 4 JSON) and supports flexible configuration:
  - `--order`: Choose "ascending" (default), "descending", or "both" to test different example orders
  - `--include-reduced`: Add this flag to include reduced context versions (no training examples in context). Default: False
  - `--modality-types`: Specify specific modality types to test (default: all). Example: `--modality-types row_only col_only image_only`
  - Example: `--order both --include-reduced` reproduces the full paper experiment
  - Example: `--order ascending` (default, no `--include-reduced`) runs a simplified version with all modalities
  - Example: `--modality-types row_only col_only` tests only row-only and column-only modalities
- **`run_single_modality_experiment.py`** tests one modality at a time (default: `row_col_image`). Use `--modality-type` to specify a different modality, or use a bash loop to test multiple modalities sequentially. Supports `--num-trials` for multiple sampling runs.
- **Output Directory**: Both scripts default to saving results in `modality_experiment_results/`, but to match the existing data structure in this repository, use `--output-dir data/experiment_results` when running experiments. Results will be saved to `data/experiment_results/{challenge_id}/{timestamp}/`.
- **Dataset Selection**: By default, scripts use the ARC Prize 2025 evaluation dataset. To use other datasets (e.g., ARC Prize 2024, training sets, or ARC-1), use the `--challenges-file` argument:
  ```bash
  # Use ARC Prize 2024 dataset
  --challenges-file data/arc-prize-2024/arc-agi_evaluation_challenges.json
  
  # Use training set
  --challenges-file data/arc-prize-2025/arc-agi_training_challenges.json
  ```
  If a solutions file exists in the same directory (e.g., `arc-agi_evaluation_solutions.json`), it will be automatically loaded to provide ground truth outputs for test cases.

