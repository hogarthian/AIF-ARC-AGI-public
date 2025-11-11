#!/usr/bin/env python3
"""
Shared utility functions for scoring calculations.
"""

import re
from typing import Set, Tuple, Optional

def col_to_num(col: str) -> int:
    """Convert column letter(s) to number: A=1, B=2, ..., Z=26, AA=27, etc."""
    result = 0
    for char in col:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result

def num_to_col(num: int) -> str:
    """Convert number to column letter(s): 1=A, 2=B, ..., 26=Z, 27=AA, etc."""
    result = ""
    num -= 1
    while num >= 0:
        result = chr(ord('A') + (num % 26)) + result
        num = num // 26 - 1
    return result

def parse_coordinate(coord: str) -> Optional[Tuple[str, int]]:
    """Parse coordinate like 'A1' into ('A', 1)"""
    match = re.match(r'([A-Z]+)(\d+)', coord.strip())
    if match:
        return (match.group(1), int(match.group(2)))
    return None

def parse_range(start: str, end: str) -> Set[str]:
    """Parse a range like 'I1-I20' into set of coordinates"""
    start_col, start_row = parse_coordinate(start)
    end_col, end_row = parse_coordinate(end)
    
    if not start_col or not end_col:
        return set()
    
    start_col_num = col_to_num(start_col)
    end_col_num = col_to_num(end_col)
    
    # Normalize reversed ranges for both rows and columns
    if start_row > end_row:
        start_row, end_row = end_row, start_row
    if start_col_num > end_col_num:
        start_col_num, end_col_num = end_col_num, start_col_num
    
    coords = set()
    for row in range(start_row, end_row + 1):
        for col_num in range(start_col_num, end_col_num + 1):
            col_str = num_to_col(col_num)
            coords.add(f"{col_str}{row}")
    
    return coords

def parse_coordinate_list(coords_str: str) -> Set[str]:
    """Parse various coordinate formats into a set"""
    coords = set()
    if not coords_str:
        return coords
    
    # Handle ranges like "I1-I20" or "I1:I20"
    range_pattern = r'([A-Z]+\d+)[-:]([A-Z]+\d+)'
    for match in re.finditer(range_pattern, coords_str):
        start, end = match.groups()
        coords.update(parse_range(start, end))
        # Remove the range from string to avoid double parsing
        coords_str = coords_str.replace(match.group(0), '')
    
    # Handle individual coordinates
    coord_pattern = r'([A-Z]+\d+)'
    for match in re.finditer(coord_pattern, coords_str):
        coord = match.group(1)
        parsed = parse_coordinate(coord)
        if parsed:
            coords.add(coord)
    
    return coords

def parse_divider_claims(divider_str: str) -> Set[str]:
    """Parse divider claims from various formats"""
    if not divider_str:
        return set()
    # Support both ranges and single coordinates in one pass
    # Example inputs: "I1-I20; J10-Q10; B3; L3" or multi-line strings
    return parse_coordinate_list(divider_str)

def calculate_diff(gt: Set[str], claimed: Set[str]) -> float:
    r"""Calculate |diff| = |(F_gt \ F_claimed) ∪ (F_claimed \ F_gt)| / |F_gt|
    
    Special case: When gt is empty (feature doesn't exist), any claimed pixels
    are hallucinations. Return 1.0 (full penalty) if there are any claims, 0.0 if none.
    """
    if not gt:
        return 1.0 if claimed else 0.0
    
    missing = gt - claimed  # Points in GT but not in claimed
    extra = claimed - gt   # Points in claimed but not in GT
    
    diff_count = len(missing) + len(extra)
    return diff_count / len(gt)

def calculate_feature_score(points_feature: float, diff: float) -> float:
    """Calculate penalty for a feature: (points_feature) × |diff|"""
    return points_feature * diff

def format_feature_score_md(feature_name: str, points: float, gt_set: Set[str], claimed_set: Set[str], is_binary: bool = False) -> Tuple[str, float]:
    """Format detailed scoring for a feature in markdown"""
    lines = []
    
    lines.append(f"**{feature_name}** ({points} points):")
    lines.append("")
    
    if is_binary:
        correct = len(claimed_set) > 0 if len(gt_set) > 0 else len(claimed_set) == 0
        penalty = 0.0 if correct else points
        lines.append(f"- Binary decision: {'Correct' if correct else 'Wrong/Missing'}")
        lines.append(f"- Penalty: {penalty:.4f}")
        return "\n".join(lines), penalty
    
    overlap = gt_set & claimed_set
    missing = gt_set - claimed_set
    extra = claimed_set - gt_set
    
    diff_value = calculate_diff(gt_set, claimed_set)
    penalty = calculate_feature_score(points, diff_value)
    
    lines.append(f"- Ground truth: {len(gt_set)} pixels - {sorted(gt_set)}")
    lines.append(f"- Claimed: {len(claimed_set)} pixels - {sorted(claimed_set)}")
    lines.append(f"- Overlap: {len(overlap)} pixels - {sorted(overlap)}")
    lines.append(f"- Missing (GT - Claimed): {len(missing)} pixels - {sorted(missing)}")
    lines.append(f"- Extra (Claimed - GT): {len(extra)} pixels - {sorted(extra)}")
    lines.append(f"- Total pixels (GT): {len(gt_set)}")
    lines.append(f"- |diff|: {diff_value:.4f}")
    lines.append(f"- Penalty: {points} × {diff_value:.4f} = {penalty:.4f}")
    
    return "\n".join(lines), penalty

