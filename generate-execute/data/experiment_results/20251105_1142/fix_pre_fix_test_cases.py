"""
Script to fix test case results for pre-fix experiments that were affected by the bug.

The bug: Test case instructions (T0, T1) were mismatched with input grids due to incorrect 
labeling in follow_instructions.py. This caused all test cases to score 0.0.

This script:
1. Identifies pre-fix experiments (the 10 completed before the bug fix)
2. For each experiment:
   - Loads hypothesis.json and reconstructs the belief
   - Loads challenge data
   - Preserves held-out validation results (they're valid)
   - Backs up old T0/T1 results to T0-bug-fix/T1-bug-fix
   - Re-runs test cases (normal + reduced) using the fixed code
   - Updates results.json and results_reduced.json with corrected test case scores
   - Updates main results.json with corrected test case results

Usage:
    uv run python fix_pre_fix_test_cases.py --experiment-dir <path_to_experiment_dir> --challenge-id <challenge_id> [--rpm <rpm>]
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import sys

# Import from test_modality_experiment.py
from src import logger, get_run_log_file
from src.utils.data_loader import (
    load_challenge,
    load_challenges_from_arc_prize_json,
    DEFAULT_CHALLENGES_FILE,
    DEFAULT_SOLUTIONS_FILE
)
from src.utils.follow_instructions import follow_instructions_twice
from src.utils.scoring_engine import get_grid_similarity
from src.nodes.models import TransformationWithUncertainty
from src.nodes.hypothesis_fast_nodes import GEMINI_MODEL, TEMPERATURE

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


def load_hypothesis_and_reconstruct_belief(hypothesis_file: Path, challenge_data: Any) -> TransformationWithUncertainty:
    """
    Load hypothesis.json and reconstruct TransformationWithUncertainty belief object.
    
    Args:
        hypothesis_file: Path to hypothesis.json
        challenge_data: Challenge data (needed to determine num_train and num_test)
        
    Returns:
        TransformationWithUncertainty belief object
    """
    with open(hypothesis_file, "r") as f:
        hypothesis_data = json.load(f)
    
    # Reconstruct belief from hypothesis_data
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    
    belief = TransformationWithUncertainty(
        working_hypothesis=hypothesis_data.get("working_hypothesis", ""),
        transform_instructions=hypothesis_data.get("transform_instructions", {}),
        uncertainty=hypothesis_data.get("uncertainty", ""),
        notebook=hypothesis_data.get("notebook", "")
    )
    
    return belief


async def process_single_test_case(
    test_idx: int,
    test_case: Any,
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    session_id: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single test case (used for parallelization).
    
    Returns:
        Tuple of (result_dict, grid_dict_entry)
    """
    logger.info(f"    Processing test {test_idx}...")
    
    # Build instructions dict for this test:
    # - Include general + all example instructions (E0, E1, ...) as context
    # - Include only the specific test instruction (T{test_idx})
    test_instructions = {}
    # Copy general and example instructions
    if "general" in belief.transform_instructions:
        test_instructions["general"] = belief.transform_instructions["general"]
    for key in belief.transform_instructions.keys():
        if key.startswith("E"):
            test_instructions[key] = belief.transform_instructions[key]
    
    # Add test instruction
    test_key = f"T{test_idx}"
    grid_key = test_key  # Use T0, T1, etc. as keys
    if test_key in belief.transform_instructions:
        test_instructions[test_key] = belief.transform_instructions[test_key]
    else:
        logger.warning(f"Test instruction {test_key} not found, using general instructions only")
    
    try:
        # Apply transform_instructions twice: ascending and descending order
        asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
            instructions=test_instructions,
            training_examples=challenge_data.train,
            test_input_grid=test_case.input,
            challenge_data=challenge_data,
            is_held_out=False,
            test_idx=test_idx,  # IMPORTANT: Pass test_idx for correct labeling
            working_hypothesis=belief.working_hypothesis,
            modality_type=modality_type,
            session_id=session_id
        )
        
        # Calculate scores (if we have ground truth)
        asc_similarity = 0.0
        desc_similarity = 0.0
        expected_grid = None
        
        if hasattr(test_case, 'output') and test_case.output is not None:
            expected_grid = test_case.output
            asc_similarity = get_grid_similarity(expected_grid, asc_grid)
            desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        
        # Use best score
        best_similarity = max(asc_similarity, desc_similarity)
        
        result = {
            "test_idx": test_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": asc_uncertainty,
                "reasoning_content": asc_reasoning
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": desc_uncertainty,
                "reasoning_content": desc_reasoning
            },
            "best_similarity": best_similarity,
            "has_ground_truth": hasattr(test_case, 'output') and test_case.output is not None
        }
        
        # Store grids
        grid_data = {
            "input": test_case.input,
            "ascending": asc_grid,
            "descending": desc_grid
        }
        if expected_grid is not None:
            grid_data["expected"] = expected_grid
        
        return result, {grid_key: grid_data}
        
    except Exception as e:
        logger.error(f"    Error processing test {test_idx}: {e}")
        error_result = {
            "test_idx": test_idx,
            "error": str(e),
            "ascending": {"similarity": 0.0},
            "descending": {"similarity": 0.0},
            "best_similarity": 0.0
        }
        error_grid = {grid_key: {"input": test_case.input, "error": str(e)}}
        return error_result, error_grid


async def rerun_test_cases(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-run test cases for a modality + order combination (normal version).
    
    Runs all test cases in parallel using asyncio.gather, respecting RPM limit.
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per test case
        - grids_dict: Dict with keys T0, T1, ... containing grids
    """
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Re-running test cases: {modality_type} ({order_name})")
    
    # Create tasks for all test cases to run in parallel
    tasks = [
        process_single_test_case(
            test_idx, test_case, challenge_data, belief, modality_type, session_id
        )
        for test_idx, test_case in enumerate(challenge_data.test)
    ]
    
    # Run all test cases in parallel
    results = await asyncio.gather(*tasks)
    
    # Unpack results
    test_results = []
    grids_dict = {}
    
    for result, grid_entry in results:
        test_results.append(result)
        grids_dict.update(grid_entry)
    
    # Sort results by test_idx to ensure consistent ordering
    test_results.sort(key=lambda x: x.get("test_idx", 0))
    
    return test_results, grids_dict


async def process_single_test_case_reduced(
    test_idx: int,
    test_case: Any,
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    session_id: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process a single test case with reduced context (used for parallelization).
    
    Returns:
        Tuple of (result_dict, grid_dict_entry)
    """
    logger.info(f"    Processing test {test_idx} (reduced)...")
    
    # Build instructions dict for this test (reduced: only general + test instruction)
    test_instructions = {}
    # Copy general instruction only
    if "general" in belief.transform_instructions:
        test_instructions["general"] = belief.transform_instructions["general"]
    
    # Add test instruction
    test_key = f"T{test_idx}"
    grid_key = test_key  # Use T0, T1, etc. as keys
    if test_key in belief.transform_instructions:
        test_instructions[test_key] = belief.transform_instructions[test_key]
    else:
        logger.warning(f"Test instruction {test_key} not found, using general instructions only")
    
    try:
        # Apply transform_instructions twice with reduced context (no training examples)
        asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
            instructions=test_instructions,
            training_examples=challenge_data.train,  # Passed but not used in modality
            test_input_grid=test_case.input,
            challenge_data=challenge_data,
            is_held_out=False,
            test_idx=test_idx,  # IMPORTANT: Pass test_idx for correct labeling
            working_hypothesis=belief.working_hypothesis,
            modality_type=modality_type,
            include_training_examples=False,  # Reduced mode: no training examples
            session_id=session_id
        )
        
        # Calculate scores (if we have ground truth)
        asc_similarity = 0.0
        desc_similarity = 0.0
        expected_grid = None
        
        if hasattr(test_case, 'output') and test_case.output is not None:
            expected_grid = test_case.output
            asc_similarity = get_grid_similarity(expected_grid, asc_grid)
            desc_similarity = get_grid_similarity(expected_grid, desc_grid)
        
        # Use best score
        best_similarity = max(asc_similarity, desc_similarity)
        
        result = {
            "test_idx": test_idx,
            "ascending": {
                "similarity": asc_similarity,
                "uncertainty": asc_uncertainty,
                "reasoning_content": asc_reasoning
            },
            "descending": {
                "similarity": desc_similarity,
                "uncertainty": desc_uncertainty,
                "reasoning_content": desc_reasoning
            },
            "best_similarity": best_similarity,
            "has_ground_truth": hasattr(test_case, 'output') and test_case.output is not None
        }
        
        # Store grids
        grid_data = {
            "input": test_case.input,
            "ascending": asc_grid,
            "descending": desc_grid
        }
        if expected_grid is not None:
            grid_data["expected"] = expected_grid
        
        return result, {grid_key: grid_data}
        
    except Exception as e:
        logger.error(f"    Error processing test {test_idx} (reduced): {e}")
        error_result = {
            "test_idx": test_idx,
            "error": str(e),
            "ascending": {"similarity": 0.0},
            "descending": {"similarity": 0.0},
            "best_similarity": 0.0
        }
        error_grid = {grid_key: {"input": test_case.input, "error": str(e)}}
        return error_result, error_grid


async def rerun_test_cases_reduced(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Re-run test cases for a modality + order combination with reduced context (no training examples).
    
    Runs all test cases in parallel using asyncio.gather, respecting RPM limit.
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per test case
        - grids_dict: Dict with keys T0, T1, ... containing grids
    """
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Re-running test cases (reduced): {modality_type} ({order_name})")
    
    # Create tasks for all test cases to run in parallel
    tasks = [
        process_single_test_case_reduced(
            test_idx, test_case, challenge_data, belief, modality_type, session_id
        )
        for test_idx, test_case in enumerate(challenge_data.test)
    ]
    
    # Run all test cases in parallel
    results = await asyncio.gather(*tasks)
    
    # Unpack results
    test_results = []
    grids_dict = {}
    
    for result, grid_entry in results:
        test_results.append(result)
        grids_dict.update(grid_entry)
    
    # Sort results by test_idx to ensure consistent ordering
    test_results.sort(key=lambda x: x.get("test_idx", 0))
    
    return test_results, grids_dict


async def fix_experiment(
    experiment_dir: Path,
    modality_type: str,
    order_name: str,
    challenge_data: Any,
    session_id: Optional[str] = None
) -> bool:
    """
    Fix test case results for a single experiment.
    
    Args:
        experiment_dir: Directory containing hypothesis.json, results.json, etc.
        modality_type: Modality type (e.g., "row_col")
        order_name: Order name ("ascending" or "descending")
        challenge_data: Challenge data
        session_id: Optional session ID for API tracking
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Fixing experiment: {modality_type} ({order_name})")
    
    hypothesis_file = experiment_dir / "hypothesis.json"
    results_file = experiment_dir / "results.json"
    results_reduced_file = experiment_dir / "results_reduced.json"
    grids_file = experiment_dir / "grids.json"
    grids_reduced_file = experiment_dir / "grids_reduced.json"
    
    # Check that required files exist
    if not hypothesis_file.exists():
        logger.error(f"  Missing hypothesis.json: {hypothesis_file}")
        return False
    
    if not results_file.exists() or not results_reduced_file.exists():
        logger.error(f"  Missing results files: {results_file} or {results_reduced_file}")
        return False
    
    # Load hypothesis and reconstruct belief
    try:
        belief = load_hypothesis_and_reconstruct_belief(hypothesis_file, challenge_data)
    except Exception as e:
        logger.error(f"  Failed to load hypothesis: {e}")
        return False
    
    # Load existing results
    with open(results_file, "r") as f:
        existing_results = json.load(f)
    
    with open(results_reduced_file, "r") as f:
        existing_results_reduced = json.load(f)
    
    # Load existing grids if they exist
    existing_grids = {}
    if grids_file.exists():
        with open(grids_file, "r") as f:
            existing_grids = json.load(f)
    
    existing_grids_reduced = {}
    if grids_reduced_file.exists():
        with open(grids_reduced_file, "r") as f:
            existing_grids_reduced = json.load(f)
    
    # Backup old test case results
    bug_backup_key = "bug-fix"
    logger.info(f"  Backing up old test case results with key '{bug_backup_key}'")
    
    # Backup normal results
    for test_key in ["T0", "T1"]:
        if test_key in existing_results:
            backup_key = f"{test_key}-{bug_backup_key}"
            existing_results[backup_key] = existing_results[test_key]
            logger.info(f"    Backed up {test_key} -> {backup_key}")
        
        if test_key in existing_grids:
            backup_key = f"{test_key}-{bug_backup_key}"
            existing_grids[backup_key] = existing_grids[test_key]
    
    # Backup reduced results
    for test_key in ["T0", "T1"]:
        if test_key in existing_results_reduced:
            backup_key = f"{test_key}-{bug_backup_key}"
            existing_results_reduced[backup_key] = existing_results_reduced[test_key]
        
        if test_key in existing_grids_reduced:
            backup_key = f"{test_key}-{bug_backup_key}"
            existing_grids_reduced[backup_key] = existing_grids_reduced[test_key]
    
    # Determine example_order integer from order_name
    example_order = None if order_name == "ascending" else -1
    
    # Re-run test cases in parallel (normal + reduced)
    logger.info(f"  Re-running test cases (normal + reduced) in parallel...")
    try:
        (test_results, test_grids), (test_results_reduced, test_grids_reduced) = await asyncio.gather(
            rerun_test_cases(
                challenge_data, belief, modality_type, example_order, session_id=session_id
            ),
            rerun_test_cases_reduced(
                challenge_data, belief, modality_type, example_order, session_id=session_id
            )
        )
    except Exception as e:
        logger.error(f"  Failed to re-run test cases: {e}")
        return False
    
    # Preserve held-out validation results (they're valid)
    # Only update test case results
    corrected_results = {}
    corrected_results_reduced = {}
    corrected_grids = {}
    corrected_grids_reduced = {}
    
    # Copy all held-out results (E0, E1, etc.)
    for key, value in existing_results.items():
        if key.startswith("E"):
            corrected_results[key] = value
    
    for key, value in existing_results_reduced.items():
        if key.startswith("E"):
            corrected_results_reduced[key] = value
    
    # Copy held-out grids
    for key, value in existing_grids.items():
        if key.startswith("E"):
            corrected_grids[key] = value
    
    for key, value in existing_grids_reduced.items():
        if key.startswith("E"):
            corrected_grids_reduced[key] = value
    
    # Add corrected test case results
    for result in test_results:
        test_key = f"T{result.get('test_idx', 0)}"
        corrected_results[test_key] = result
    
    for result in test_results_reduced:
        test_key = f"T{result.get('test_idx', 0)}"
        corrected_results_reduced[test_key] = result
    
    # Add corrected test case grids
    corrected_grids.update(test_grids)
    corrected_grids_reduced.update(test_grids_reduced)
    
    # Keep all backup entries
    for key, value in existing_results.items():
        if "-" in key:  # Backup entries have "-" in key (e.g., "T0-bug-fix")
            corrected_results[key] = value
    
    for key, value in existing_results_reduced.items():
        if "-" in key:
            corrected_results_reduced[key] = value
    
    for key, value in existing_grids.items():
        if "-" in key:
            corrected_grids[key] = value
    
    for key, value in existing_grids_reduced.items():
        if "-" in key:
            corrected_grids_reduced[key] = value
    
    # Save corrected results
    with open(results_file, "w") as f:
        json.dump(corrected_results, f, indent=2)
    
    with open(results_reduced_file, "w") as f:
        json.dump(corrected_results_reduced, f, indent=2)
    
    # Save grids (always save if we have data, even if file didn't exist before)
    if corrected_grids or existing_grids:
        with open(grids_file, "w") as f:
            json.dump(corrected_grids, f, indent=2)
    
    if corrected_grids_reduced or existing_grids_reduced:
        with open(grids_reduced_file, "w") as f:
            json.dump(corrected_grids_reduced, f, indent=2)
    
    logger.info(f"  Successfully fixed {modality_type} ({order_name})")
    return True


async def update_main_results(
    main_results_file: Path,
    modality_type: str,
    order_name: str,
    test_results: List[Dict[str, Any]],
    test_results_reduced: List[Dict[str, Any]]
):
    """
    Update main results.json with corrected test case results.
    
    Args:
        main_results_file: Path to main results.json
        modality_type: Modality type
        order_name: Order name
        test_results: Corrected test results (normal)
        test_results_reduced: Corrected test results (reduced)
    """
    if not main_results_file.exists():
        logger.warning(f"  Main results.json not found: {main_results_file}")
        return
    
    with open(main_results_file, "r") as f:
        main_results = json.load(f)
    
    # Find the corresponding result entry
    results_list = main_results.get("results", [])
    for i, result in enumerate(results_list):
        if result.get("modality_type") == modality_type and result.get("example_order") == order_name:
            # Update test_results
            result["test_results"] = test_results
            result["test_results_reduced"] = test_results_reduced
            logger.info(f"  Updated main results.json for {modality_type} ({order_name})")
            break
    
    # Save updated main results
    with open(main_results_file, "w") as f:
        json.dump(main_results, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="Fix test case results for pre-fix experiments"
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
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Maximum requests per minute (RPM) for rate limiting"
    )
    
    args = parser.parse_args()
    
    # Show log file location
    run_log_file = get_run_log_file()
    if run_log_file:
        print(f"📝 Log file: {run_log_file}")
        print()
    
    # Suppress console logging (keep file logging)
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            root_logger.removeHandler(handler)
    
    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Load challenge data (same as test_modality_experiment.py)
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
        # Use default challenges file (same as test_modality_experiment.py)
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
    
    # Setup rate limiting if needed (same as test_modality_experiment.py)
    if args.rpm:
        import litellm
        from collections import deque
        import time
        
        # Define RateLimiter locally (same as test_modality_experiment.py)
        class RateLimiter:
            def __init__(self, rpm: Optional[int] = None):
                self.rpm = rpm
                self.call_times: deque = deque()
                self.lock = asyncio.Lock()
            
            async def wait_if_needed(self):
                if self.rpm is None:
                    return
                async with self.lock:
                    now = time.time()
                    minute_ago = now - 60.0
                    while self.call_times and self.call_times[0] < minute_ago:
                        self.call_times.popleft()
                    if len(self.call_times) >= self.rpm:
                        oldest_time = self.call_times[0]
                        wait_time = 60.0 - (now - oldest_time) + 0.1
                        if wait_time > 0:
                            logger.info(f"Rate limiter: Waiting {wait_time:.2f}s to stay within {self.rpm} RPM limit")
                            await asyncio.sleep(wait_time)
                            now = time.time()
                            while self.call_times and self.call_times[0] < now - 60.0:
                                self.call_times.popleft()
                    self.call_times.append(now)
        
        rate_limiter = RateLimiter(args.rpm)
        logger.info(f"Rate limiting enabled: {args.rpm} RPM")
        
        _original_acompletion = litellm.acompletion
        
        async def patched_acompletion(*args, **kwargs):
            await rate_limiter.wait_if_needed()
            return await _original_acompletion(*args, **kwargs)
        
        litellm.acompletion = patched_acompletion
        
        # Also patch in the follow_instructions module
        import src.utils.follow_instructions as follow_instructions_module
        follow_instructions_module.litellm.acompletion = patched_acompletion
    
    # Create session ID
    session_id = f"{args.challenge_id}-fix_pre_fix_test_cases"
    
    # Process each pre-fix experiment
    main_results_file = experiment_dir / "results.json"
    
    success_count = 0
    fail_count = 0
    
    for modality_type, order_name in PRE_FIX_EXPERIMENTS:
        exp_dir = experiment_dir / f"{modality_type}_{order_name}"
        
        if not exp_dir.exists():
            logger.warning(f"Skipping {modality_type} ({order_name}): directory not found")
            continue
        
        success = await fix_experiment(
            exp_dir, modality_type, order_name, challenge_data, session_id=session_id
        )
        
        if success:
            success_count += 1
            # Update main results.json
            # Load corrected results to update main file
            results_file = exp_dir / "results.json"
            results_reduced_file = exp_dir / "results_reduced.json"
            
            with open(results_file, "r") as f:
                corrected_results = json.load(f)
            
            with open(results_reduced_file, "r") as f:
                corrected_results_reduced = json.load(f)
            
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
            
            await update_main_results(
                main_results_file, modality_type, order_name,
                test_results, test_results_reduced
            )
        else:
            fail_count += 1
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Successfully fixed: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Total: {len(PRE_FIX_EXPERIMENTS)}")
    
    if success_count > 0:
        logger.info(f"\nFixed experiments have been updated. Old test case results are backed up with '-bug-fix' suffix.")
        logger.info(f"Main results.json has been updated with corrected test case scores.")


if __name__ == "__main__":
    asyncio.run(main())

