"""
Scoring engine utilities for Leave-One-Out validation.

Uses modality_encoder format standards for grid diff and formatting.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from src.utils.modality_encoder import (
    format_grid_row_wise,
    format_grid_column_wise,
    format_grid_diff_simplified
)
from src import logger


def get_grid_similarity(
    ground_truth_grid: List[List[int]], 
    sample_grid: List[List[int]]
) -> float:
    """
    Calculate similarity as the percentage of cells that match exactly.
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).
    
    Args:
        ground_truth_grid: Expected output grid
        sample_grid: Generated output grid
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if not ground_truth_grid or not sample_grid:
        return 0.0
    
    # Convert to numpy arrays for easier comparison
    gt_array = np.array(ground_truth_grid)
    sample_array = np.array(sample_grid)
    
    # Check if grids have the same dimensions
    if gt_array.shape != sample_array.shape:
        logger.warning(f"Grid shape mismatch: expected {gt_array.shape}, got {sample_array.shape}")
        return 0.0
    
    # Calculate matching cells
    total_cells = gt_array.size
    matching_cells = np.sum(gt_array == sample_array)
    
    similarity = matching_cells / total_cells
    
    return float(similarity)


def generate_grid_diff(
    expected_grid: List[List[int]], 
    actual_grid: List[List[int]],
    example_prefix: str = "Diff"
) -> Tuple[str, str]:
    """
    Generate diff notation using modality_encoder format (with | separators).
    
    Uses format_grid_diff_simplified from modality_encoder directly.
    Returns both the row-wise diff text and the spreadsheet notation list.
    
    Args:
        expected_grid: Expected output grid
        actual_grid: Generated output grid
        example_prefix: Prefix for the diff (e.g., "E0" for example 0)
        
    Returns:
        Tuple of (diff_text, notation_list) where:
        - diff_text: Row-wise diff format with | separators (modality_encoder standard)
        - notation_list: Spreadsheet notation list grouped by color changes
    """
    if not expected_grid or not actual_grid:
        return "Error: Empty grid(s)", ""
    
    # Convert to numpy arrays for dimension check
    expected_array = np.array(expected_grid)
    actual_array = np.array(actual_grid)
    
    # Handle dimension mismatch
    if expected_array.shape != actual_array.shape:
        return f"Error: Shape mismatch - expected {expected_array.shape}, got {actual_array.shape}", ""
    
    # Use modality_encoder's format_grid_diff_simplified directly
    # This returns format like "E01: 00000|F1: 3->2|000300" with | separators
    # Keep the original format as-is (modality_encoder standard)
    diff_text, notation_list = format_grid_diff_simplified(
        expected_grid, actual_grid, example_prefix
    )
    
    return diff_text, notation_list


def format_grid_for_prompt(grid: List[List[int]], prefix: str = "Grid") -> str:
    """
    Format grid using modality_encoder's row-wise and column-wise format.
    
    Args:
        grid: Grid to format
        prefix: Prefix for the grid (e.g., "E0i" for example 0 input)
        
    Returns:
        Formatted string with both row-wise and column-wise views
    """
    row_wise = format_grid_row_wise(grid, prefix)
    col_wise = format_grid_column_wise(grid, prefix)
    return f"Row-wise:\n{row_wise}\n\nColumn-wise:\n{col_wise}"


def get_failure_details(
    expected_grid: List[List[int]],
    actual_grid: List[List[int]],
    example_idx: int
) -> Dict[str, Any]:
    """
    Get detailed failure information for a single example.
    
    Uses modality_encoder format standards.
    
    Args:
        expected_grid: Expected output grid
        actual_grid: Generated output grid
        example_idx: Index of the example (for reference)
        
    Returns:
        Dictionary with failure details including diff text and notation list
    """
    similarity = get_grid_similarity(expected_grid, actual_grid)
    example_prefix = f"E{example_idx}"
    diff_text, notation_list = generate_grid_diff(expected_grid, actual_grid, example_prefix)
    
    # Count mismatches
    if expected_grid and actual_grid:
        expected_array = np.array(expected_grid)
        actual_array = np.array(actual_grid)
        
        if expected_array.shape == actual_array.shape:
            mismatches = np.sum(expected_array != actual_array)
            total_cells = expected_array.size
        else:
            mismatches = -1  # Shape mismatch
            total_cells = -1
    else:
        mismatches = -1
        total_cells = -1
    
    return {
        "example_idx": example_idx,
        "similarity": similarity,
        "mismatches": mismatches,
        "total_cells": total_cells,
        "diff_text": diff_text,
        "notation_list": notation_list,
        "expected_grid": expected_grid,
        "actual_grid": actual_grid
    }

