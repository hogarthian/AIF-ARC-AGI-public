"""
Script to fix grids with inconsistent row lengths or inhomogeneous errors by regenerating them.

This script:
1. Scans all grids.json and grids_reduced.json files in experiment directories
2. Detects grids with:
   - Inconsistent row lengths (LLM generation failures)
   - Entries with "error" fields containing "inhomogeneous part" messages
3. For each invalid grid:
   - Loads hypothesis.json to get instructions
   - Determines test case index and whether it's normal/reduced, ascending/descending
   - Re-runs LLM call to regenerate the grid
   - Backs up old grid with "-invalid-grid" suffix
   - Removes error field if regeneration succeeds
   - Saves new grid

Usage:
    uv run python fix_invalid_grids.py --experiment-dir <path_to_experiment_dir> --challenge-id <challenge_id> [--rpm <rpm>] [--dry-run]
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

from src import logger, get_run_log_file
from src.utils.data_loader import (
    load_challenges_from_arc_prize_json,
    DEFAULT_CHALLENGES_FILE,
    DEFAULT_SOLUTIONS_FILE
)
from src.utils.follow_instructions import follow_instructions_to_generate_grid
from src.nodes.models import TransformationWithUncertainty


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


def find_invalid_grids(grids_dict: Dict[str, Any], grid_type: str) -> List[Tuple[str, str]]:
    """
    Find grids with inconsistent row lengths or entries with inhomogeneous errors.
    
    Args:
        grids_dict: Dictionary containing grids (keys like T0, T1, E0, etc.)
        grid_type: Type indicator for logging (e.g., "normal", "reduced")
        
    Returns:
        List of tuples (grid_key, invalid_field) where invalid_field is "ascending", "descending", or "both"
    """
    invalid_grids = []
    
    for grid_key, grid_data in grids_dict.items():
        if not isinstance(grid_data, dict):
            continue
        
        # Check for inhomogeneous errors
        if "error" in grid_data:
            error_msg = str(grid_data.get("error", ""))
            if "inhomogeneous" in error_msg.lower() or "inhomogeneous part" in error_msg.lower():
                # For test cases (T0, T1, etc.), regenerate both ascending and descending
                # For held-out examples (E0, E1, etc.), also regenerate both
                if grid_key.startswith("T") or grid_key.startswith("E"):
                    invalid_grids.append((grid_key, "both"))
                    logger.warning(
                        f"  Found entry with inhomogeneous error: {grid_key} ({grid_type}) - "
                        f"error: {error_msg[:100]}"
                    )
                continue
        
        asc_grid = grid_data.get("ascending")
        desc_grid = grid_data.get("descending")
        
        asc_valid = is_valid_grid(asc_grid) if asc_grid is not None else True  # None means not present, skip
        desc_valid = is_valid_grid(desc_grid) if desc_grid is not None else True
        
        if not asc_valid or not desc_valid:
            if not asc_valid and not desc_valid:
                invalid_field = "both"
            elif not asc_valid:
                invalid_field = "ascending"
            else:
                invalid_field = "descending"
            
            invalid_grids.append((grid_key, invalid_field))
            
            row_lengths_asc = []
            row_lengths_desc = []
            if asc_grid:
                row_lengths_asc = [len(row) for row in asc_grid if isinstance(row, list)]
            if desc_grid:
                row_lengths_desc = [len(row) for row in desc_grid if isinstance(row, list)]
            
            logger.warning(
                f"  Found invalid grid: {grid_key} ({grid_type}) - {invalid_field} invalid. "
                f"Asc row lengths: {set(row_lengths_asc) if row_lengths_asc else 'N/A'}, "
                f"Desc row lengths: {set(row_lengths_desc) if row_lengths_desc else 'N/A'}"
            )
    
    return invalid_grids


def load_hypothesis_and_reconstruct_belief(hypothesis_file: Path, challenge_data: Any) -> TransformationWithUncertainty:
    """Load hypothesis.json and reconstruct TransformationWithUncertainty belief object."""
    with open(hypothesis_file, "r") as f:
        hypothesis_data = json.load(f)
    
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    
    belief = TransformationWithUncertainty(
        working_hypothesis=hypothesis_data.get("working_hypothesis", ""),
        transform_instructions=hypothesis_data.get("transform_instructions", {}),
        uncertainty=hypothesis_data.get("uncertainty", ""),
        notebook=hypothesis_data.get("notebook", "")
    )
    
    return belief


async def regenerate_held_out_grid(
    held_out_idx: int,
    held_out_example: Any,
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    is_reduced: bool,
    is_ascending: bool,
    session_id: Optional[str] = None
) -> Tuple[List[List[int]], str]:
    """
    Regenerate a single grid (ascending or descending) for a held-out example.
    
    Args:
        held_out_idx: Held-out example index
        held_out_example: Held-out example object with input/output
        challenge_data: Full challenge data
        belief: TransformationWithUncertainty belief
        modality_type: Modality type
        is_reduced: If True, use reduced context (no training examples)
        is_ascending: If True, regenerate ascending grid (example_order=None), else descending (example_order=-1)
        session_id: Optional session ID
        
    Returns:
        Tuple of (grid, uncertainty)
    """
    # Build instructions for held-out validation
    held_out_instructions = {}
    
    if is_reduced:
        # Reduced: only general instruction
        if "general" in belief.transform_instructions:
            held_out_instructions["general"] = belief.transform_instructions["general"]
    else:
        # Normal: general + context example instructions (all except held-out)
        if "general" in belief.transform_instructions:
            held_out_instructions["general"] = belief.transform_instructions["general"]
        for key in belief.transform_instructions.keys():
            if key.startswith("E") and key != f"E{held_out_idx}":
                held_out_instructions[key] = belief.transform_instructions[key]
    
    # Use held-out example's instruction as test instruction
    held_out_key = f"E{held_out_idx}"
    if held_out_key in belief.transform_instructions:
        held_out_instructions["T0"] = belief.transform_instructions[held_out_key]
    else:
        logger.warning(f"Held-out instruction {held_out_key} not found, using general instructions only")
    
    # Set example_order based on is_ascending
    example_order = None if is_ascending else -1
    
    # Get context examples (all except held-out)
    context_examples = challenge_data.train[:held_out_idx] + challenge_data.train[held_out_idx+1:]
    
    # Regenerate the specific grid using follow_instructions_to_generate_grid
    try:
        grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
            instructions=held_out_instructions,
            training_examples=context_examples,
            test_input_grid=held_out_example.input,
            challenge_data=challenge_data,
            is_held_out=True,
            example_order=example_order,
            test_idx=0,  # Use 0 for held-out validation
            working_hypothesis=belief.working_hypothesis,
            modality_type=modality_type,
            include_training_examples=not is_reduced,
            session_id=session_id
        )
        
        return grid, uncertainty
        
    except Exception as e:
        logger.error(f"    Error regenerating grid for held-out {held_out_idx} ({'ascending' if is_ascending else 'descending'}): {e}")
        raise


async def regenerate_grid(
    test_idx: int,
    test_case: Any,
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    is_reduced: bool,
    is_ascending: bool,
    session_id: Optional[str] = None
) -> Tuple[List[List[int]], str]:
    """
    Regenerate a single grid (ascending or descending) for a test case.
    
    Args:
        test_idx: Test case index
        test_case: Test case object with input
        challenge_data: Full challenge data
        belief: TransformationWithUncertainty belief
        modality_type: Modality type
        is_reduced: If True, use reduced context (no training examples)
        is_ascending: If True, regenerate ascending grid (example_order=None), else descending (example_order=-1)
        session_id: Optional session ID
        
    Returns:
        Tuple of (grid, uncertainty)
    """
    # Build instructions dict
    test_instructions = {}
    
    if is_reduced:
        # Reduced: only general + test instruction
        if "general" in belief.transform_instructions:
            test_instructions["general"] = belief.transform_instructions["general"]
    else:
        # Normal: general + all example instructions + test instruction
        if "general" in belief.transform_instructions:
            test_instructions["general"] = belief.transform_instructions["general"]
        for key in belief.transform_instructions.keys():
            if key.startswith("E"):
                test_instructions[key] = belief.transform_instructions[key]
    
    # Add test instruction
    test_key = f"T{test_idx}"
    if test_key in belief.transform_instructions:
        test_instructions[test_key] = belief.transform_instructions[test_key]
    else:
        logger.warning(f"Test instruction {test_key} not found, using general instructions only")
    
    # Set example_order based on is_ascending
    # When is_reduced=True, example_order doesn't matter, but we still need to set it correctly
    example_order = None if is_ascending else -1
    
    # Regenerate the specific grid using follow_instructions_to_generate_grid
    try:
        grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
            instructions=test_instructions,
            training_examples=challenge_data.train,
            test_input_grid=test_case.input,
            challenge_data=challenge_data,
            is_held_out=False,
            example_order=example_order,
            test_idx=test_idx,
            working_hypothesis=belief.working_hypothesis,
            modality_type=modality_type,
            include_training_examples=not is_reduced,
            session_id=session_id
        )
        
        return grid, uncertainty
        
    except Exception as e:
        logger.error(f"    Error regenerating grid for test {test_idx} ({'ascending' if is_ascending else 'descending'}): {e}")
        raise


async def fix_invalid_grids_in_experiment(
    experiment_dir: Path,
    challenge_data: Any,
    dry_run: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fix invalid grids in a single experiment directory.
    
    Args:
        experiment_dir: Experiment directory containing grids.json, grids_reduced.json, hypothesis.json
        challenge_data: Challenge data
        dry_run: If True, only detect invalid grids without regenerating
        session_id: Optional session ID
        
    Returns:
        Dictionary with statistics about fixes
    """
    logger.info(f"Processing experiment: {experiment_dir.name}")
    
    hypothesis_file = experiment_dir / "hypothesis.json"
    grids_file = experiment_dir / "grids.json"
    grids_reduced_file = experiment_dir / "grids_reduced.json"
    
    # Check required files
    if not hypothesis_file.exists():
        logger.warning(f"  Missing hypothesis.json, skipping")
        return {"skipped": True, "reason": "missing_hypothesis"}
    
    if not grids_file.exists() and not grids_reduced_file.exists():
        logger.warning(f"  Missing grids files, skipping")
        return {"skipped": True, "reason": "missing_grids"}
    
    # Load hypothesis
    try:
        belief = load_hypothesis_and_reconstruct_belief(hypothesis_file, challenge_data)
    except Exception as e:
        logger.error(f"  Failed to load hypothesis: {e}")
        return {"skipped": True, "reason": "failed_to_load_hypothesis"}
    
    # Get modality type from directory name or hypothesis
    modality_type = None
    if "modality_type" in json.load(open(hypothesis_file)):
        modality_type = json.load(open(hypothesis_file)).get("modality_type")
    else:
        # Try to infer from directory name
        dir_name = experiment_dir.name
        for mt in ["row_only", "col_only", "image_only", "row_col", "row_image", "col_image", "row_col_image"]:
            if dir_name.startswith(mt):
                modality_type = mt
                break
    
    if not modality_type:
        logger.warning(f"  Could not determine modality type, skipping")
        return {"skipped": True, "reason": "unknown_modality"}
    
    stats = {
        "modality_type": modality_type,
        "fixed_normal": 0,
        "fixed_reduced": 0,
        "regenerated": [],
        "invalid_details": []  # For dry-run summary
    }
    
    # Process normal grids
    if grids_file.exists():
        with open(grids_file, "r") as f:
            grids = json.load(f)
        
        invalid_grids = find_invalid_grids(grids, "normal")
        
        if invalid_grids:
            logger.info(f"  Found {len(invalid_grids)} invalid grid entries in grids.json")
            
            if dry_run:
                # Collect details for dry-run summary
                for grid_key, invalid_field in invalid_grids:
                    if "-" in grid_key:  # Skip backup entries
                        continue
                    grid_data = grids.get(grid_key, {})
                    details = {
                        "grid_key": grid_key,
                        "type": "normal",
                        "invalid_field": invalid_field
                    }
                    
                    # Add row length details if available
                    if invalid_field in ["ascending", "both"]:
                        asc_grid = grid_data.get("ascending")
                        if asc_grid:
                            row_lengths = [len(row) for row in asc_grid if isinstance(row, list)]
                            details["asc_row_lengths"] = set(row_lengths) if row_lengths else None
                    
                    if invalid_field in ["descending", "both"]:
                        desc_grid = grid_data.get("descending")
                        if desc_grid:
                            row_lengths = [len(row) for row in desc_grid if isinstance(row, list)]
                            details["desc_row_lengths"] = set(row_lengths) if row_lengths else None
                    
                    # Check for error messages
                    if "error" in grid_data:
                        error_msg = str(grid_data.get("error", ""))
                        details["error"] = error_msg[:200]  # Truncate long errors
                    
                    stats["invalid_details"].append(details)
            
            for grid_key, invalid_field in invalid_grids:
                # Skip backup entries (keys with "-" in them)
                if "-" in grid_key:
                    continue
                
                # Handle test cases (T0, T1, etc.)
                if grid_key.startswith("T"):
                    test_idx = int(grid_key[1:])
                    if test_idx >= len(challenge_data.test):
                        logger.warning(f"    Test index {test_idx} out of range (max: {len(challenge_data.test)-1})")
                        continue
                    
                    test_case = challenge_data.test[test_idx]
                    
                    if dry_run:
                        logger.info(f"    [DRY RUN] Would regenerate {grid_key} ({invalid_field})")
                        stats["regenerated"].append({
                            "grid_key": grid_key,
                            "type": "normal",
                            "invalid_field": invalid_field,
                            "dry_run": True
                        })
                    else:
                        logger.info(f"    Regenerating {grid_key} ({invalid_field})")
                        
                        # Backup old grids (if entry exists)
                        backup_key = f"{grid_key}-invalid-grid"
                        if grid_key in grids and backup_key not in grids:
                            grids[backup_key] = grids[grid_key].copy()
                        
                        # Ensure entry exists
                        if grid_key not in grids:
                            grids[grid_key] = {}
                        
                        # Track if any regeneration succeeded
                        regeneration_success = False
                        
                        # Regenerate ascending if needed
                        if invalid_field in ["ascending", "both"]:
                            try:
                                asc_new, asc_unc = await regenerate_grid(
                                    test_idx=test_idx,
                                    test_case=test_case,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=False,
                                    is_ascending=True,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(asc_new):
                                    row_lengths = [len(row) for row in asc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated ascending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids[grid_key]["ascending"] = asc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids[grid_key]:
                                        error_msg = grids[grid_key]["error"]
                                        grids[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids[grid_key]["error"]
                                    logger.info(f"      Regenerated ascending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate ascending grid: {e}")
                        
                        # Regenerate descending if needed
                        if invalid_field in ["descending", "both"]:
                            try:
                                desc_new, desc_unc = await regenerate_grid(
                                    test_idx=test_idx,
                                    test_case=test_case,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=False,
                                    is_ascending=False,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(desc_new):
                                    row_lengths = [len(row) for row in desc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated descending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids[grid_key]["descending"] = desc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids[grid_key]:
                                        error_msg = grids[grid_key]["error"]
                                        grids[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids[grid_key]["error"]
                                    logger.info(f"      Regenerated descending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate descending grid: {e}")
                        
                        if regeneration_success:
                            stats["fixed_normal"] += 1
                            stats["regenerated"].append({
                                "grid_key": grid_key,
                                "type": "normal",
                                "invalid_field": invalid_field
                            })
                
                # Handle held-out examples (E0, E1, etc.)
                elif grid_key.startswith("E"):
                    held_out_idx = int(grid_key[1:])
                    if held_out_idx >= len(challenge_data.train):
                        logger.warning(f"    Held-out index {held_out_idx} out of range (max: {len(challenge_data.train)-1})")
                        continue
                    
                    held_out_example = challenge_data.train[held_out_idx]
                    
                    if dry_run:
                        logger.info(f"    [DRY RUN] Would regenerate {grid_key} ({invalid_field})")
                        stats["regenerated"].append({
                            "grid_key": grid_key,
                            "type": "normal",
                            "invalid_field": invalid_field,
                            "dry_run": True
                        })
                    else:
                        logger.info(f"    Regenerating {grid_key} ({invalid_field})")
                        
                        # Backup old grids (if entry exists)
                        backup_key = f"{grid_key}-invalid-grid"
                        if grid_key in grids and backup_key not in grids:
                            grids[backup_key] = grids[grid_key].copy()
                        
                        # Ensure entry exists
                        if grid_key not in grids:
                            grids[grid_key] = {}
                        
                        # Track if any regeneration succeeded
                        regeneration_success = False
                        
                        # Regenerate ascending if needed
                        if invalid_field in ["ascending", "both"]:
                            try:
                                asc_new, asc_unc = await regenerate_held_out_grid(
                                    held_out_idx=held_out_idx,
                                    held_out_example=held_out_example,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=False,
                                    is_ascending=True,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(asc_new):
                                    row_lengths = [len(row) for row in asc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated ascending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids[grid_key]["ascending"] = asc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids[grid_key]:
                                        error_msg = grids[grid_key]["error"]
                                        grids[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids[grid_key]["error"]
                                    logger.info(f"      Regenerated ascending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate ascending grid: {e}")
                        
                        # Regenerate descending if needed
                        if invalid_field in ["descending", "both"]:
                            try:
                                desc_new, desc_unc = await regenerate_held_out_grid(
                                    held_out_idx=held_out_idx,
                                    held_out_example=held_out_example,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=False,
                                    is_ascending=False,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(desc_new):
                                    row_lengths = [len(row) for row in desc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated descending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids[grid_key]["descending"] = desc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids[grid_key]:
                                        error_msg = grids[grid_key]["error"]
                                        grids[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids[grid_key]["error"]
                                    logger.info(f"      Regenerated descending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate descending grid: {e}")
                        
                        if regeneration_success:
                            stats["fixed_normal"] += 1
                            stats["regenerated"].append({
                                "grid_key": grid_key,
                                "type": "normal",
                                "invalid_field": invalid_field
                            })
                
                else:
                    logger.warning(f"    Skipping {grid_key}: not a test case or held-out example")
                    continue
            
            # Save updated grids
            if not dry_run:
                with open(grids_file, "w") as f:
                    json.dump(grids, f, indent=2)
                logger.info(f"  Saved updated grids.json")
    
    # Process reduced grids
    if grids_reduced_file.exists():
        with open(grids_reduced_file, "r") as f:
            grids_reduced = json.load(f)
        
        invalid_grids_reduced = find_invalid_grids(grids_reduced, "reduced")
        
        if invalid_grids_reduced:
            logger.info(f"  Found {len(invalid_grids_reduced)} invalid grid entries in grids_reduced.json")
            
            if dry_run:
                # Collect details for dry-run summary
                for grid_key, invalid_field in invalid_grids_reduced:
                    if "-" in grid_key:  # Skip backup entries
                        continue
                    grid_data = grids_reduced.get(grid_key, {})
                    details = {
                        "grid_key": grid_key,
                        "type": "reduced",
                        "invalid_field": invalid_field
                    }
                    
                    # Add row length details if available
                    if invalid_field in ["ascending", "both"]:
                        asc_grid = grid_data.get("ascending")
                        if asc_grid:
                            row_lengths = [len(row) for row in asc_grid if isinstance(row, list)]
                            details["asc_row_lengths"] = set(row_lengths) if row_lengths else None
                    
                    if invalid_field in ["descending", "both"]:
                        desc_grid = grid_data.get("descending")
                        if desc_grid:
                            row_lengths = [len(row) for row in desc_grid if isinstance(row, list)]
                            details["desc_row_lengths"] = set(row_lengths) if row_lengths else None
                    
                    # Check for error messages
                    if "error" in grid_data:
                        error_msg = str(grid_data.get("error", ""))
                        details["error"] = error_msg[:200]  # Truncate long errors
                    
                    stats["invalid_details"].append(details)
            
            for grid_key, invalid_field in invalid_grids_reduced:
                # Skip backup entries (keys with "-" in them)
                if "-" in grid_key:
                    continue
                
                # Handle test cases (T0, T1, etc.)
                if grid_key.startswith("T"):
                    test_idx = int(grid_key[1:])
                    if test_idx >= len(challenge_data.test):
                        logger.warning(f"    Test index {test_idx} out of range (max: {len(challenge_data.test)-1})")
                        continue
                    
                    test_case = challenge_data.test[test_idx]
                    
                    if dry_run:
                        logger.info(f"    [DRY RUN] Would regenerate {grid_key} reduced ({invalid_field})")
                        stats["regenerated"].append({
                            "grid_key": grid_key,
                            "type": "reduced",
                            "invalid_field": invalid_field,
                            "dry_run": True
                        })
                    else:
                        logger.info(f"    Regenerating {grid_key} reduced ({invalid_field})")
                        
                        # Backup old grids (if entry exists)
                        backup_key = f"{grid_key}-invalid-grid"
                        if grid_key in grids_reduced and backup_key not in grids_reduced:
                            grids_reduced[backup_key] = grids_reduced[grid_key].copy()
                        
                        # Ensure entry exists
                        if grid_key not in grids_reduced:
                            grids_reduced[grid_key] = {}
                        
                        # Track if any regeneration succeeded
                        regeneration_success = False
                        
                        # Regenerate ascending if needed
                        if invalid_field in ["ascending", "both"]:
                            try:
                                asc_new, asc_unc = await regenerate_grid(
                                    test_idx=test_idx,
                                    test_case=test_case,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=True,
                                    is_ascending=True,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(asc_new):
                                    row_lengths = [len(row) for row in asc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated ascending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids_reduced[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids_reduced[grid_key]["ascending"] = asc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids_reduced[grid_key]:
                                        error_msg = grids_reduced[grid_key]["error"]
                                        grids_reduced[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids_reduced[grid_key]["error"]
                                    logger.info(f"      Regenerated ascending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate ascending grid: {e}")
                        
                        # Regenerate descending if needed
                        if invalid_field in ["descending", "both"]:
                            try:
                                desc_new, desc_unc = await regenerate_grid(
                                    test_idx=test_idx,
                                    test_case=test_case,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=True,
                                    is_ascending=False,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(desc_new):
                                    row_lengths = [len(row) for row in desc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated descending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids_reduced[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids_reduced[grid_key]["descending"] = desc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids_reduced[grid_key]:
                                        error_msg = grids_reduced[grid_key]["error"]
                                        grids_reduced[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids_reduced[grid_key]["error"]
                                    logger.info(f"      Regenerated descending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate descending grid: {e}")
                        
                        if regeneration_success:
                            stats["fixed_reduced"] += 1
                            stats["regenerated"].append({
                                "grid_key": grid_key,
                                "type": "reduced",
                                "invalid_field": invalid_field
                            })
                
                # Handle held-out examples (E0, E1, etc.)
                elif grid_key.startswith("E"):
                    held_out_idx = int(grid_key[1:])
                    if held_out_idx >= len(challenge_data.train):
                        logger.warning(f"    Held-out index {held_out_idx} out of range (max: {len(challenge_data.train)-1})")
                        continue
                    
                    held_out_example = challenge_data.train[held_out_idx]
                    
                    if dry_run:
                        logger.info(f"    [DRY RUN] Would regenerate {grid_key} reduced ({invalid_field})")
                        stats["regenerated"].append({
                            "grid_key": grid_key,
                            "type": "reduced",
                            "invalid_field": invalid_field,
                            "dry_run": True
                        })
                    else:
                        logger.info(f"    Regenerating {grid_key} reduced ({invalid_field})")
                        
                        # Backup old grids (if entry exists)
                        backup_key = f"{grid_key}-invalid-grid"
                        if grid_key in grids_reduced and backup_key not in grids_reduced:
                            grids_reduced[backup_key] = grids_reduced[grid_key].copy()
                        
                        # Ensure entry exists
                        if grid_key not in grids_reduced:
                            grids_reduced[grid_key] = {}
                        
                        # Track if any regeneration succeeded
                        regeneration_success = False
                        
                        # Regenerate ascending if needed
                        if invalid_field in ["ascending", "both"]:
                            try:
                                asc_new, asc_unc = await regenerate_held_out_grid(
                                    held_out_idx=held_out_idx,
                                    held_out_example=held_out_example,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=True,
                                    is_ascending=True,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(asc_new):
                                    row_lengths = [len(row) for row in asc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated ascending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids_reduced[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids_reduced[grid_key]["ascending"] = asc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids_reduced[grid_key]:
                                        error_msg = grids_reduced[grid_key]["error"]
                                        grids_reduced[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids_reduced[grid_key]["error"]
                                    logger.info(f"      Regenerated ascending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate ascending grid: {e}")
                        
                        # Regenerate descending if needed
                        if invalid_field in ["descending", "both"]:
                            try:
                                desc_new, desc_unc = await regenerate_held_out_grid(
                                    held_out_idx=held_out_idx,
                                    held_out_example=held_out_example,
                                    challenge_data=challenge_data,
                                    belief=belief,
                                    modality_type=modality_type,
                                    is_reduced=True,
                                    is_ascending=False,
                                    session_id=session_id
                                )
                                # Validate regenerated grid before saving
                                if not is_valid_grid(desc_new):
                                    row_lengths = [len(row) for row in desc_new if isinstance(row, list)]
                                    logger.error(
                                        f"      Regenerated descending grid is still invalid (row lengths: {set(row_lengths)}). "
                                        f"Not saving invalid grid."
                                    )
                                    grids_reduced[grid_key]["error"] = f"Regeneration produced invalid grid with inconsistent row lengths: {set(row_lengths)}"
                                else:
                                    grids_reduced[grid_key]["descending"] = desc_new
                                    # Rename error field to error-fixed with appended message if it exists
                                    if "error" in grids_reduced[grid_key]:
                                        error_msg = grids_reduced[grid_key]["error"]
                                        grids_reduced[grid_key]["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                        del grids_reduced[grid_key]["error"]
                                    logger.info(f"      Regenerated descending grid (validated)")
                                    regeneration_success = True
                            except Exception as e:
                                logger.error(f"      Failed to regenerate descending grid: {e}")
                        
                        if regeneration_success:
                            stats["fixed_reduced"] += 1
                            stats["regenerated"].append({
                                "grid_key": grid_key,
                                "type": "reduced",
                                "invalid_field": invalid_field
                            })
                
                else:
                    logger.warning(f"    Skipping {grid_key}: not a test case or held-out example")
                    continue
            
            # Save updated grids
            if not dry_run:
                with open(grids_reduced_file, "w") as f:
                    json.dump(grids_reduced, f, indent=2)
                logger.info(f"  Saved updated grids_reduced.json")
    
    return stats


def update_results_files(experiment_dir: Path, dry_run: bool = False):
    """
    Update results.json and results_reduced.json files to rename error fields to error-fixed.
    
    Args:
        experiment_dir: Experiment directory containing results.json, results_reduced.json
        dry_run: If True, only detect errors without updating
    """
    results_file = experiment_dir / "results.json"
    results_reduced_file = experiment_dir / "results_reduced.json"
    
    updated_count = 0
    
    # Update results.json (main aggregated file)
    if results_file.exists():
        with open(results_file, "r") as f:
            results_data = json.load(f)
        
        # Check if it's the main aggregated results file structure
        if "results" in results_data and isinstance(results_data["results"], list):
            for result_group in results_data["results"]:
                # Update held_out_results
                for entry in result_group.get("held_out_results", []):
                    if "error" in entry:
                        error_msg = str(entry["error"])
                        if "inhomogeneous" in error_msg.lower():
                            if not dry_run:
                                entry["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                del entry["error"]
                            updated_count += 1
                
                # Update test_results
                for entry in result_group.get("test_results", []):
                    if "error" in entry:
                        error_msg = str(entry["error"])
                        if "inhomogeneous" in error_msg.lower():
                            if not dry_run:
                                entry["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                del entry["error"]
                            updated_count += 1
                
                # Update held_out_results_reduced
                for entry in result_group.get("held_out_results_reduced", []):
                    if "error" in entry:
                        error_msg = str(entry["error"])
                        if "inhomogeneous" in error_msg.lower():
                            if not dry_run:
                                entry["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                                del entry["error"]
                            updated_count += 1
        
        if updated_count > 0 and not dry_run:
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"  Updated {updated_count} error entries in results.json")
    
    # Update results_reduced.json (per-experiment file)
    if results_reduced_file.exists():
        with open(results_reduced_file, "r") as f:
            results_reduced_data = json.load(f)
        
        updated_reduced = 0
        
        # Handle dict structure (E0, E1, etc. as keys)
        if isinstance(results_reduced_data, dict):
            for key, entry in results_reduced_data.items():
                if isinstance(entry, dict) and "error" in entry:
                    error_msg = str(entry["error"])
                    if "inhomogeneous" in error_msg.lower():
                        if not dry_run:
                            entry["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                            del entry["error"]
                        updated_reduced += 1
        
        if updated_reduced > 0 and not dry_run:
            with open(results_reduced_file, "w") as f:
                json.dump(results_reduced_data, f, indent=2)
            logger.info(f"  Updated {updated_reduced} error entries in results_reduced.json")
    
    # Also check per-experiment results.json (if different from main results.json)
    results_file_per_exp = experiment_dir / "results.json"
    if results_file_per_exp.exists() and results_file_per_exp != results_file:
        with open(results_file_per_exp, "r") as f:
            results_per_exp_data = json.load(f)
        
        updated_per_exp = 0
        
        # Handle dict structure (E0, E1, etc. as keys)
        if isinstance(results_per_exp_data, dict):
            for key, entry in results_per_exp_data.items():
                if isinstance(entry, dict) and "error" in entry:
                    error_msg = str(entry["error"])
                    if "inhomogeneous" in error_msg.lower():
                        if not dry_run:
                            entry["error-fixed"] = f"{error_msg} (this experiment has been rerun and added to above output)"
                            del entry["error"]
                        updated_per_exp += 1
        
        if updated_per_exp > 0 and not dry_run:
            with open(results_file_per_exp, "w") as f:
                json.dump(results_per_exp_data, f, indent=2)
            logger.info(f"  Updated {updated_per_exp} error entries in per-experiment results.json")


async def main():
    parser = argparse.ArgumentParser(
        description="Fix grids with inconsistent row lengths by regenerating them"
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect invalid grids without regenerating them"
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
    
    # Load challenge data
    logger.info(f"Loading challenge: {args.challenge_id}")
    if args.challenges_file:
        solutions_path = args.challenges_file.parent / "arc-agi_evaluation_solutions.json"
        if not solutions_path.exists():
            solutions_path = None
            logger.warning(f"Solutions file not found, test cases will not have ground truth")
        
        challenges = load_challenges_from_arc_prize_json(
            args.challenges_file, challenge_ids={args.challenge_id}, solutions_path=solutions_path
        )
        if args.challenge_id not in challenges:
            raise ValueError(f"Challenge {args.challenge_id} not found")
        challenge_data = challenges[args.challenge_id]
    else:
        challenges_file = DEFAULT_CHALLENGES_FILE
        solutions_file = DEFAULT_SOLUTIONS_FILE
        if challenges_file.exists():
            solutions_path = solutions_file if solutions_file.exists() else None
            if not solutions_path:
                logger.warning(f"Solutions file not found, test cases will not have ground truth")
            
            challenges = load_challenges_from_arc_prize_json(
                challenges_file, challenge_ids={args.challenge_id}, solutions_path=solutions_path
            )
            if args.challenge_id not in challenges:
                raise ValueError(f"Challenge {args.challenge_id} not found in {challenges_file}")
            challenge_data = challenges[args.challenge_id]
        else:
            raise FileNotFoundError(f"Challenges file not found: {challenges_file}. Please specify --challenges-file")
    
    logger.info(f"Challenge loaded: {len(challenge_data.train)} train, {len(challenge_data.test)} test")
    
    # Setup rate limiting if needed
    if args.rpm:
        import litellm
        from collections import deque
        import time
        
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
    session_id = f"{args.challenge_id}-fix_invalid_grids"
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE: Only detecting invalid grids, not regenerating ===")
    
    # Process all experiment subdirectories
    all_stats = []
    
    for subdir in experiment_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Skip non-experiment directories
        if subdir.name in ["plots"]:
            continue
        
        # Check if this is an experiment directory (has hypothesis.json)
        if not (subdir / "hypothesis.json").exists():
            continue
        
        stats = await fix_invalid_grids_in_experiment(
            subdir, challenge_data, dry_run=args.dry_run, session_id=session_id
        )
        
        if not stats.get("skipped"):
            all_stats.append(stats)
        
        # Update results.json files for this experiment
        update_results_files(subdir, dry_run=args.dry_run)
    
    # Also update main results.json file
    main_results_file = experiment_dir / "results.json"
    if main_results_file.exists():
        update_results_files(experiment_dir, dry_run=args.dry_run)
    
    # Summary
    logger.info(f"\n=== Summary ===")
    total_fixed_normal = sum(s.get("fixed_normal", 0) for s in all_stats)
    total_fixed_reduced = sum(s.get("fixed_reduced", 0) for s in all_stats)
    total_regenerated = sum(len(s.get("regenerated", [])) for s in all_stats)
    
    logger.info(f"Experiments processed: {len(all_stats)}")
    logger.info(f"Invalid grids detected: {total_regenerated}")
    logger.info(f"  Normal grids fixed: {total_fixed_normal}")
    logger.info(f"  Reduced grids fixed: {total_fixed_reduced}")
    
    if args.dry_run:
        logger.info(f"\n{'='*80}")
        logger.info(f"DRY RUN SUMMARY: Grids that would be regenerated")
        logger.info(f"{'='*80}")
        
        # Group by experiment
        for stats in all_stats:
            if not stats.get("invalid_details"):
                continue
            
            modality = stats.get("modality_type", "unknown")
            logger.info(f"\nExperiment: {modality}")
            logger.info(f"{'-'*80}")
            
            # Group by type (normal/reduced)
            normal_details = [d for d in stats["invalid_details"] if d["type"] == "normal"]
            reduced_details = [d for d in stats["invalid_details"] if d["type"] == "reduced"]
            
            if normal_details:
                logger.info(f"  Normal grids ({len(normal_details)}):")
                for detail in normal_details:
                    grid_key = detail["grid_key"]
                    invalid_field = detail["invalid_field"]
                    info_parts = [f"    {grid_key}: {invalid_field}"]
                    
                    if "asc_row_lengths" in detail and detail["asc_row_lengths"]:
                        info_parts.append(f"asc row lengths: {detail['asc_row_lengths']}")
                    if "desc_row_lengths" in detail and detail["desc_row_lengths"]:
                        info_parts.append(f"desc row lengths: {detail['desc_row_lengths']}")
                    if "error" in detail:
                        info_parts.append(f"error: {detail['error'][:100]}")
                    
                    logger.info(" | ".join(info_parts))
            
            if reduced_details:
                logger.info(f"  Reduced grids ({len(reduced_details)}):")
                for detail in reduced_details:
                    grid_key = detail["grid_key"]
                    invalid_field = detail["invalid_field"]
                    info_parts = [f"    {grid_key}: {invalid_field}"]
                    
                    if "asc_row_lengths" in detail and detail["asc_row_lengths"]:
                        info_parts.append(f"asc row lengths: {detail['asc_row_lengths']}")
                    if "desc_row_lengths" in detail and detail["desc_row_lengths"]:
                        info_parts.append(f"desc row lengths: {detail['desc_row_lengths']}")
                    if "error" in detail:
                        info_parts.append(f"error: {detail['error'][:100]}")
                    
                    logger.info(" | ".join(info_parts))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DRY RUN: No grids were regenerated. Run without --dry-run to fix invalid grids.")
        logger.info(f"{'='*80}")
    else:
        logger.info(f"\nInvalid grids have been regenerated. Old grids are backed up with '-invalid-grid' suffix.")


if __name__ == "__main__":
    asyncio.run(main())

