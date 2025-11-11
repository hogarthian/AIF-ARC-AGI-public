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
- `20251108_2058/` - Repeat sampling experiment (4 trials with temperature 0.7)

## Reproducing the Experiments

### Cross-Order Experiment (20251105_1142)

This experiment was run using `run_double_modality_experiment.py`:

```bash
uv run python run_double_modality_experiment.py --challenge-id 13e47133 --output-dir data/experiment_results --rpm 60
```

This script tests all 7 modality types (row_only, col_only, image_only, row_col, row_image, col_image, row_col_image) with both ascending and descending example orders. It generates hypotheses, runs held-out validation, and tests on test cases for each combination.

**Note**: 
- This experiment requires ~294 API calls (for challenge 13e47133 with 3 training examples and 1 test case). Use the `--rpm` flag to enable rate limiting to avoid hitting API rate limits.
- Results are saved to `data/experiment_results/{challenge_id}/{timestamp}/` to match the existing data structure. 

### JSON-Based Experiments (20251108_1435, 20251108_1444, 20251108_1455, 20251108_1504)

These experiments can be reproduced using `run_single_modality_experiment.py`:

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

### Repeat Sampling Experiment (20251108_2058)

This experiment tested output diversity with multiple trials:

```bash
uv run python run_single_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --num-trials 4 --temperature 0.7 --output-dir data/experiment_results --rpm 60
```

## Reproducing the Violin Plot

Use the `reproduce_violin_plot.ipynb` notebook to generate the violin plot from the paper. The notebook loads results from the experiment directories and creates the plot showing score distributions by modality.

## Notes

- The cross-order experiment (20251105_1142) tested both ascending and descending orders to investigate order effects
- After concluding that order doesn't significantly affect quality, subsequent experiments (20251108_*) only used ascending order
- The simplified experiment script (`run_single_modality_experiment.py`) was created to streamline future experiments
- **Output Directory**: Both scripts default to saving results in `modality_experiment_results/`, but to match the existing data structure in this repository, use `--output-dir data/experiment_results` when running experiments. Results will be saved to `data/experiment_results/{challenge_id}/{timestamp}/`.

