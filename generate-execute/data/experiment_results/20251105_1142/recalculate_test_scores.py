"""
Script to recalculate test case and held-out example scores from saved grids using ground truth outputs.

This script fixes the bug where ground truth outputs were not loaded during the original
experiment, causing all test cases to show has_ground_truth: false and scores of 0.0.
It also recalculates held-out example scores, which is useful after regenerating grids
with fix_invalid_grids.py.

This script:
1. Loads challenge data WITH solutions (now fixed)
2. For each experiment folder:
   - Loads grids.json and grids_reduced.json (contains generated grids)
   - Loads results.json and results_reduced.json (contains scores)
   - For each held-out example (E0, E1, etc.):
     - Gets ground truth from challenge_data.train[held_out_idx].output
     - Compares grids.json[E{idx}].ascending/descending with ground truth
     - Recalculates similarity scores
   - For each test case (T0, T1, etc.):
     - Gets ground truth from challenge_data.test[test_idx].output
     - Compares grids.json[T{idx}].ascending/descending with ground truth
     - Recalculates similarity scores
   - Updates results.json and results_reduced.json with corrected scores
   - Updates main results.json with corrected held-out and test case results

Usage:
    uv run python recalculate_test_scores.py --experiment-dir <path_to_experiment_dir> --challenge-id <challenge_id> [--challenges-file <challenges_file>]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src import logger, get_run_log_file
from src.utils.data_loader import (
    load_challenges_from_arc_prize_json,
    DEFAULT_CHALLENGES_FILE,
    DEFAULT_SOLUTIONS_FILE
)
from src.utils.scoring_engine import get_grid_similarity

# List of pre-fix experiments from PRE_FIX_STATE.md
PRE_FIX_EXPERIMENTS = [
    ("col_only", "ascending"),
    ("col_only", "descending"),
    ("image_only", "ascending"),
    ("image_only", "descending"),
    ("row_col", "ascending"),
    ("row_col", "descending"),
    ("row_image", "ascending"),
    ("row_image", "descending"),
    ("row_only", "ascending"),
    ("row_only", "descending"),
]


def is_valid_grid(grid: List[List[int]]) -> bool:
    """
    Check if a grid has consistent row lengths (all rows must have the same length).
    
    Args:
        grid: Grid to validate
        
    Returns:
        True if grid is valid (consistent row lengths), False otherwise
    """
    if not grid:
        return False
    
    if not isinstance(grid, list):
        return False
    
    if len(grid) == 0:
        return False
    
    # Check if all rows are lists and have the same length
    first_row_length = None
    for row in grid:
        if not isinstance(row, list):
            return False
        if first_row_length is None:
            first_row_length = len(row)
        elif len(row) != first_row_length:
            return False
    
    return True


def recalculate_test_scores_for_experiment(
    experiment_dir: Path,
    challenge_data: Any
) -> bool:
    """
    Recalculate held-out example and test case scores for a single experiment.
    
    Args:
        experiment_dir: Directory containing grids.json, results.json, etc.
        challenge_data: Challenge data with ground truth outputs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Recalculating scores for: {experiment_dir.name}")
    
    results_file = experiment_dir / "results.json"
    results_reduced_file = experiment_dir / "results_reduced.json"
    grids_file = experiment_dir / "grids.json"
    grids_reduced_file = experiment_dir / "grids_reduced.json"
    
    # Check that required files exist
    if not results_file.exists() or not results_reduced_file.exists():
        logger.warning(f"  Missing results files: {results_file} or {results_reduced_file}")
        return False
    
    if not grids_file.exists() or not grids_reduced_file.exists():
        logger.warning(f"  Missing grids files: {grids_file} or {grids_reduced_file}")
        return False
    
    # Load existing results
    with open(results_file, "r") as f:
        existing_results = json.load(f)
    
    with open(results_reduced_file, "r") as f:
        existing_results_reduced = json.load(f)
    
    # Load grids
    with open(grids_file, "r") as f:
        grids = json.load(f)
    
    with open(grids_reduced_file, "r") as f:
        grids_reduced = json.load(f)
    
    # Create corrected results (recalculate both held-out and test cases)
    corrected_results = {}
    corrected_results_reduced = {}
    
    # Copy all backup entries (e.g., T0-bug-fix, E0-invalid-grid)
    for key, value in existing_results.items():
        if "-" in key:
            corrected_results[key] = value
    
    for key, value in existing_results_reduced.items():
        if "-" in key:
            corrected_results_reduced[key] = value
    
    # Recalculate held-out example scores (normal)
    for held_out_idx, held_out_example in enumerate(challenge_data.train):
        held_out_key = f"E{held_out_idx}"
        
        # Skip if held-out example not in grids
        if held_out_key not in grids:
            logger.warning(f"  Held-out example {held_out_key} not found in grids.json, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results:
                corrected_results[held_out_key] = existing_results[held_out_key]
            continue
        
        grid_data = grids[held_out_key]
        
        # Get ground truth output (from training example)
        expected_grid = held_out_example.output
        if expected_grid is None:
            logger.warning(f"  Held-out example {held_out_key} has no ground truth output, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results:
                corrected_results[held_out_key] = existing_results[held_out_key]
            continue
        
        # Get generated grids
        asc_grid = grid_data.get("ascending")
        desc_grid = grid_data.get("descending")
        
        if asc_grid is None or desc_grid is None:
            logger.warning(f"  Held-out example {held_out_key} missing ascending/descending grids, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results:
                corrected_results[held_out_key] = existing_results[held_out_key]
            continue
        
        # Validate grids before calculating similarity
        if not is_valid_grid(asc_grid) or not is_valid_grid(desc_grid):
            row_lengths_asc = [len(row) for row in asc_grid if isinstance(row, list)] if asc_grid else []
            row_lengths_desc = [len(row) for row in desc_grid if isinstance(row, list)] if desc_grid else []
            logger.warning(
                f"  Held-out example {held_out_key} has invalid grids (asc row lengths: {set(row_lengths_asc)}, "
                f"desc row lengths: {set(row_lengths_desc)}), skipping score calculation"
            )
            # Preserve existing result if present
            if held_out_key in existing_results:
                corrected_results[held_out_key] = existing_results[held_out_key]
            continue
        
        # Calculate similarity scores
        asc_similarity = get_grid_similarity(expected_grid, asc_grid)
        desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        best_similarity = max(asc_similarity, desc_similarity)
        
        # Get existing result to preserve metadata (uncertainty, reasoning_content)
        existing_result = existing_results.get(held_out_key, {})
        
        # Create updated result
        corrected_result = {
            "held_out_idx": held_out_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": existing_result.get("ascending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("ascending", {}).get("reasoning_content")
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": existing_result.get("descending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("descending", {}).get("reasoning_content")
            },
            "best_similarity": best_similarity
        }
        
        # Preserve error if present
        if "error" in existing_result:
            corrected_result["error"] = existing_result["error"]
        
        # Preserve skipped flag if present
        if "skipped" in existing_result:
            corrected_result["skipped"] = existing_result["skipped"]
        
        corrected_results[held_out_key] = corrected_result
        logger.info(f"  Updated {held_out_key}: ascending={asc_similarity:.4f}, descending={desc_similarity:.4f}, best={best_similarity:.4f}")
    
    # Recalculate test case scores (normal)
    for test_idx, test_case in enumerate(challenge_data.test):
        test_key = f"T{test_idx}"
        
        # Skip if test case not in grids
        if test_key not in grids:
            logger.warning(f"  Test case {test_key} not found in grids.json, skipping")
            # Preserve existing result if present
            if test_key in existing_results:
                corrected_results[test_key] = existing_results[test_key]
            continue
        
        grid_data = grids[test_key]
        
        # Get ground truth output
        expected_grid = test_case.output
        if expected_grid is None:
            logger.warning(f"  Test case {test_key} has no ground truth output, skipping")
            # Preserve existing result if present
            if test_key in existing_results:
                corrected_results[test_key] = existing_results[test_key]
            continue
        
        # Get generated grids
        asc_grid = grid_data.get("ascending")
        desc_grid = grid_data.get("descending")
        
        if asc_grid is None or desc_grid is None:
            logger.warning(f"  Test case {test_key} missing ascending/descending grids, skipping")
            # Preserve existing result if present
            if test_key in existing_results:
                corrected_results[test_key] = existing_results[test_key]
            continue
        
        # Validate grids before calculating similarity
        if not is_valid_grid(asc_grid) or not is_valid_grid(desc_grid):
            row_lengths_asc = [len(row) for row in asc_grid if isinstance(row, list)] if asc_grid else []
            row_lengths_desc = [len(row) for row in desc_grid if isinstance(row, list)] if desc_grid else []
            logger.warning(
                f"  Test case {test_key} has invalid grids (asc row lengths: {set(row_lengths_asc)}, "
                f"desc row lengths: {set(row_lengths_desc)}), skipping score calculation"
            )
            # Preserve existing result if present
            if test_key in existing_results:
                corrected_results[test_key] = existing_results[test_key]
            continue
        
        # Calculate similarity scores
        asc_similarity = get_grid_similarity(expected_grid, asc_grid)
        desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        best_similarity = max(asc_similarity, desc_similarity)
        
        # Get existing result to preserve metadata (uncertainty, reasoning_content)
        existing_result = existing_results.get(test_key, {})
        
        # Create updated result
        corrected_result = {
            "test_idx": test_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": existing_result.get("ascending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("ascending", {}).get("reasoning_content")
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": existing_result.get("descending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("descending", {}).get("reasoning_content")
            },
            "best_similarity": best_similarity,
            "has_ground_truth": True
        }
        
        # Preserve error if present
        if "error" in existing_result:
            corrected_result["error"] = existing_result["error"]
        
        corrected_results[test_key] = corrected_result
        logger.info(f"  Updated {test_key}: ascending={asc_similarity:.4f}, descending={desc_similarity:.4f}, best={best_similarity:.4f}")
    
    # Recalculate held-out example scores (reduced)
    for held_out_idx, held_out_example in enumerate(challenge_data.train):
        held_out_key = f"E{held_out_idx}"
        
        # Skip if held-out example not in grids_reduced
        if held_out_key not in grids_reduced:
            logger.warning(f"  Held-out example {held_out_key} not found in grids_reduced.json, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results_reduced:
                corrected_results_reduced[held_out_key] = existing_results_reduced[held_out_key]
            continue
        
        grid_data = grids_reduced[held_out_key]
        
        # Get ground truth output (from training example)
        expected_grid = held_out_example.output
        if expected_grid is None:
            logger.warning(f"  Held-out example {held_out_key} has no ground truth output, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results_reduced:
                corrected_results_reduced[held_out_key] = existing_results_reduced[held_out_key]
            continue
        
        # Get generated grids
        asc_grid = grid_data.get("ascending")
        desc_grid = grid_data.get("descending")
        
        if asc_grid is None or desc_grid is None:
            logger.warning(f"  Held-out example {held_out_key} missing ascending/descending grids, skipping")
            # Preserve existing result if present
            if held_out_key in existing_results_reduced:
                corrected_results_reduced[held_out_key] = existing_results_reduced[held_out_key]
            continue
        
        # Validate grids before calculating similarity
        if not is_valid_grid(asc_grid) or not is_valid_grid(desc_grid):
            row_lengths_asc = [len(row) for row in asc_grid if isinstance(row, list)] if asc_grid else []
            row_lengths_desc = [len(row) for row in desc_grid if isinstance(row, list)] if desc_grid else []
            logger.warning(
                f"  Held-out example {held_out_key} (reduced) has invalid grids (asc row lengths: {set(row_lengths_asc)}, "
                f"desc row lengths: {set(row_lengths_desc)}), skipping score calculation"
            )
            # Preserve existing result if present
            if held_out_key in existing_results_reduced:
                corrected_results_reduced[held_out_key] = existing_results_reduced[held_out_key]
            continue
        
        # Calculate similarity scores
        asc_similarity = get_grid_similarity(expected_grid, asc_grid)
        desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        best_similarity = max(asc_similarity, desc_similarity)
        
        # Get existing result to preserve metadata (uncertainty, reasoning_content)
        existing_result = existing_results_reduced.get(held_out_key, {})
        
        # Create updated result
        corrected_result = {
            "held_out_idx": held_out_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": existing_result.get("ascending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("ascending", {}).get("reasoning_content")
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": existing_result.get("descending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("descending", {}).get("reasoning_content")
            },
            "best_similarity": best_similarity
        }
        
        # Preserve error if present
        if "error" in existing_result:
            corrected_result["error"] = existing_result["error"]
        
        # Preserve skipped flag if present
        if "skipped" in existing_result:
            corrected_result["skipped"] = existing_result["skipped"]
        
        corrected_results_reduced[held_out_key] = corrected_result
        logger.info(f"  Updated {held_out_key} (reduced): ascending={asc_similarity:.4f}, descending={desc_similarity:.4f}, best={best_similarity:.4f}")
    
    # Recalculate test case scores (reduced)
    for test_idx, test_case in enumerate(challenge_data.test):
        test_key = f"T{test_idx}"
        
        # Skip if test case not in grids_reduced
        if test_key not in grids_reduced:
            logger.warning(f"  Test case {test_key} not found in grids_reduced.json, skipping")
            # Preserve existing result if present
            if test_key in existing_results_reduced:
                corrected_results_reduced[test_key] = existing_results_reduced[test_key]
            continue
        
        grid_data = grids_reduced[test_key]
        
        # Get ground truth output
        expected_grid = test_case.output
        if expected_grid is None:
            logger.warning(f"  Test case {test_key} has no ground truth output, skipping")
            # Preserve existing result if present
            if test_key in existing_results_reduced:
                corrected_results_reduced[test_key] = existing_results_reduced[test_key]
            continue
        
        # Get generated grids
        asc_grid = grid_data.get("ascending")
        desc_grid = grid_data.get("descending")
        
        if asc_grid is None or desc_grid is None:
            logger.warning(f"  Test case {test_key} missing ascending/descending grids, skipping")
            # Preserve existing result if present
            if test_key in existing_results_reduced:
                corrected_results_reduced[test_key] = existing_results_reduced[test_key]
            continue
        
        # Validate grids before calculating similarity
        if not is_valid_grid(asc_grid) or not is_valid_grid(desc_grid):
            row_lengths_asc = [len(row) for row in asc_grid if isinstance(row, list)] if asc_grid else []
            row_lengths_desc = [len(row) for row in desc_grid if isinstance(row, list)] if desc_grid else []
            logger.warning(
                f"  Test case {test_key} (reduced) has invalid grids (asc row lengths: {set(row_lengths_asc)}, "
                f"desc row lengths: {set(row_lengths_desc)}), skipping score calculation"
            )
            # Preserve existing result if present
            if test_key in existing_results_reduced:
                corrected_results_reduced[test_key] = existing_results_reduced[test_key]
            continue
        
        # Calculate similarity scores
        asc_similarity = get_grid_similarity(expected_grid, asc_grid)
        desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        best_similarity = max(asc_similarity, desc_similarity)
        
        # Get existing result to preserve metadata (uncertainty, reasoning_content)
        existing_result = existing_results_reduced.get(test_key, {})
        
        # Create updated result
        corrected_result = {
            "test_idx": test_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": existing_result.get("ascending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("ascending", {}).get("reasoning_content")
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": existing_result.get("descending", {}).get("uncertainty", ""),
                "reasoning_content": existing_result.get("descending", {}).get("reasoning_content")
            },
            "best_similarity": best_similarity,
            "has_ground_truth": True
        }
        
        # Preserve error if present
        if "error" in existing_result:
            corrected_result["error"] = existing_result["error"]
        
        corrected_results_reduced[test_key] = corrected_result
        logger.info(f"  Updated {test_key} (reduced): ascending={asc_similarity:.4f}, descending={desc_similarity:.4f}, best={best_similarity:.4f}")
    
    # Save corrected results
    with open(results_file, "w") as f:
        json.dump(corrected_results, f, indent=2)
    
    with open(results_reduced_file, "w") as f:
        json.dump(corrected_results_reduced, f, indent=2)
    
    logger.info(f"  Successfully recalculated scores for {experiment_dir.name}")
    return True


def update_main_results(
    main_results_file: Path,
    challenge_data: Any
):
    """
    Update main results.json with corrected held-out example and test case results.
    
    Args:
        main_results_file: Path to main results.json
        challenge_data: Challenge data
    """
    if not main_results_file.exists():
        logger.warning(f"  Main results.json not found: {main_results_file}")
        return
    
    with open(main_results_file, "r") as f:
        main_results = json.load(f)
    
    # Update test results for each experiment
    results_list = main_results.get("results", [])
    updated_count = 0
    
    for result in results_list:
        modality_type = result.get("modality_type")
        order_name = result.get("example_order")
        
        # Find corresponding experiment directory
        experiment_dir = main_results_file.parent / f"{modality_type}_{order_name}"
        
        if not experiment_dir.exists():
            continue
        
        # Load corrected results
        results_file = experiment_dir / "results.json"
        results_reduced_file = experiment_dir / "results_reduced.json"
        
        if not results_file.exists() or not results_reduced_file.exists():
            continue
        
        with open(results_file, "r") as f:
            corrected_results = json.load(f)
        
        with open(results_reduced_file, "r") as f:
            corrected_results_reduced = json.load(f)
        
        # Extract held-out results (sort by held_out_idx to ensure correct order)
        held_out_results = []
        held_out_results_reduced = []
        
        # Collect and sort by held_out_idx
        held_out_results_dict = {}
        held_out_results_reduced_dict = {}
        
        for key in corrected_results.keys():
            if key.startswith("E") and "-" not in key:  # Exclude backup entries
                held_out_results_dict[key] = corrected_results[key]
        
        for key in corrected_results_reduced.keys():
            if key.startswith("E") and "-" not in key:  # Exclude backup entries
                held_out_results_reduced_dict[key] = corrected_results_reduced[key]
        
        # Sort by held_out_idx
        held_out_results = sorted(
            held_out_results_dict.values(),
            key=lambda x: x.get("held_out_idx", 0)
        )
        held_out_results_reduced = sorted(
            held_out_results_reduced_dict.values(),
            key=lambda x: x.get("held_out_idx", 0)
        )
        
        # Extract test results (sort by test_idx to ensure correct order)
        test_results = []
        test_results_reduced = []
        
        # Collect and sort by test_idx
        test_results_dict = {}
        test_results_reduced_dict = {}
        
        for key in corrected_results.keys():
            if key.startswith("T") and "-" not in key:  # Exclude backup entries
                test_results_dict[key] = corrected_results[key]
        
        for key in corrected_results_reduced.keys():
            if key.startswith("T") and "-" not in key:  # Exclude backup entries
                test_results_reduced_dict[key] = corrected_results_reduced[key]
        
        # Sort by test_idx
        test_results = sorted(
            test_results_dict.values(),
            key=lambda x: x.get("test_idx", 0)
        )
        test_results_reduced = sorted(
            test_results_reduced_dict.values(),
            key=lambda x: x.get("test_idx", 0)
        )
        
        # Update result entry
        result["held_out_results"] = held_out_results
        result["held_out_results_reduced"] = held_out_results_reduced
        result["test_results"] = test_results
        result["test_results_reduced"] = test_results_reduced
        updated_count += 1
        logger.info(f"  Updated main results.json for {modality_type} ({order_name})")
    
    # Save updated main results
    with open(main_results_file, "w") as f:
        json.dump(main_results, f, indent=2)
    
    logger.info(f"  Updated main results.json: {updated_count} experiments")


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate held-out example and test case scores from saved grids using ground truth outputs"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory (e.g., modality_experiment_results/13e47133/20251105_1142)"
    )
    parser.add_argument(
        "--challenge-id",
        type=str,
        required=True,
        help="Challenge ID (e.g., 13e47133)"
    )
    parser.add_argument(
        "--challenges-file",
        type=Path,
        default=None,
        help=f"Path to challenges JSON file (default: {DEFAULT_CHALLENGES_FILE})"
    )
    
    args = parser.parse_args()
    
    # Show log file location
    run_log_file = get_run_log_file()
    if run_log_file:
        print(f"📝 Log file: {run_log_file}")
        print()
    
    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Load challenge data WITH solutions (the fix)
    logger.info(f"Loading challenge: {args.challenge_id}")
    if args.challenges_file:
        # Determine solutions file path
        solutions_path = args.challenges_file.parent / "arc-agi_evaluation_solutions.json"
        if not solutions_path.exists():
            solutions_path = None
            logger.warning(f"Solutions file not found at {solutions_path}, test cases will not have ground truth")
        
        challenges = load_challenges_from_arc_prize_json(
            args.challenges_file, challenge_ids={args.challenge_id}, solutions_path=solutions_path
        )
        if args.challenge_id not in challenges:
            raise ValueError(f"Challenge {args.challenge_id} not found")
        challenge_data = challenges[args.challenge_id]
    else:
        # Use default challenges file
        challenges_file = DEFAULT_CHALLENGES_FILE
        solutions_file = DEFAULT_SOLUTIONS_FILE
        if challenges_file.exists():
            solutions_path = solutions_file if solutions_file.exists() else None
            if not solutions_path:
                logger.warning(f"Solutions file not found at {solutions_file}, test cases will not have ground truth")
            
            challenges = load_challenges_from_arc_prize_json(
                challenges_file, challenge_ids={args.challenge_id}, solutions_path=solutions_path
            )
            if args.challenge_id not in challenges:
                raise ValueError(f"Challenge {args.challenge_id} not found in {challenges_file}")
            challenge_data = challenges[args.challenge_id]
        else:
            raise FileNotFoundError(f"Challenges file not found: {challenges_file}. Please specify --challenges-file")
    
    logger.info(f"Challenge loaded: {len(challenge_data.train)} train, {len(challenge_data.test)} test")
    
    # Check if test cases have ground truth
    test_cases_with_ground_truth = sum(1 for t in challenge_data.test if t.output is not None)
    logger.info(f"Test cases with ground truth: {test_cases_with_ground_truth}/{len(challenge_data.test)}")
    
    if test_cases_with_ground_truth == 0:
        logger.error("No test cases have ground truth outputs. Cannot recalculate scores.")
        return
    
    # Process each experiment
    main_results_file = experiment_dir / "results.json"
    
    success_count = 0
    fail_count = 0
    
    # Process all experiments (not just pre-fix ones, in case others need fixing too)
    for modality_order_dir in experiment_dir.iterdir():
        if not modality_order_dir.is_dir():
            continue
        
        # Skip plots directory and other non-experiment directories
        if modality_order_dir.name in ["plots"]:
            continue
        
        # Check if this is an experiment directory (has results.json)
        if not (modality_order_dir / "results.json").exists():
            logger.debug(f"Skipping {modality_order_dir.name}: no results.json")
            continue
        
        success = recalculate_test_scores_for_experiment(
            modality_order_dir, challenge_data
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Update main results.json
    update_main_results(main_results_file, challenge_data)
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Successfully recalculated: {success_count}")
    logger.info(f"Failed: {fail_count}")
    
    if success_count > 0:
        logger.info(f"\nHeld-out example and test case scores have been recalculated using ground truth outputs.")
        logger.info(f"Main results.json has been updated with corrected held-out and test case scores.")


if __name__ == "__main__":
    main()

