"""Data loading utilities for ARC-AGI competition format.

Supports loading from arc-prize competition JSON files:
- arc-agi_training_challenges.json
- arc-agi_evaluation_challenges.json
- arc-agi_test_challenges.json

Also supports legacy ARC-AGI format (individual JSON files per challenge).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)

# Data paths - relative to repository root
# Repository root is 3 levels up from this file: src/utils/data_loader.py -> repo root
REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
ARC_PRIZE_2024_DIR = DATA_DIR / "arc-prize-2024"
ARC_PRIZE_2025_DIR = DATA_DIR / "arc-prize-2025"

# Default challenge and solution files
DEFAULT_CHALLENGES_FILE = ARC_PRIZE_2025_DIR / "arc-agi_evaluation_challenges.json"
DEFAULT_SOLUTIONS_FILE = ARC_PRIZE_2025_DIR / "arc-agi_evaluation_solutions.json"


def resolve_challenges_path(path: Optional[Path | str], repo_root: Optional[Path] = None) -> Path:
    """
    Resolve a challenges file path relative to repository root.
    
    This function ensures that relative paths (like 'data/arc-prize-2024/arc-agi_evaluation_challenges.json')
    are resolved relative to the repository root, regardless of the current working directory.
    
    Args:
        path: Path to challenges file. Can be:
            - None: Returns default challenges file
            - Absolute path: Returns as-is
            - Relative path: Resolved relative to repo_root
        repo_root: Repository root directory. If None, uses REPO_ROOT from this module.
        
    Returns:
        Resolved absolute Path object
        
    Examples:
        >>> resolve_challenges_path('data/arc-prize-2024/arc-agi_evaluation_challenges.json')
        Path('/repo/root/data/arc-prize-2024/arc-agi_evaluation_challenges.json')
        
        >>> resolve_challenges_path('/absolute/path/to/file.json')
        Path('/absolute/path/to/file.json')
        
        >>> resolve_challenges_path(None)
        Path('/repo/root/data/arc-prize-2025/arc-agi_evaluation_challenges.json')
    """
    if repo_root is None:
        repo_root = REPO_ROOT
    
    if path is None:
        return DEFAULT_CHALLENGES_FILE
    
    path = Path(path)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # Resolve relative to repo root
    return repo_root / path


class Example(BaseModel):
    """Single training/test example"""
    input: list[list[int]]
    output: Optional[list[list[int]]] = None  # None for test examples without outputs
    
    @property
    def input_np(self) -> np.ndarray:
        return np.array(self.input)
    
    @property
    def output_np(self) -> np.ndarray:
        if self.output is None:
            raise ValueError("Output not available for test example")
        return np.array(self.output)


class Challenge(BaseModel):
    """ARC-AGI challenge with training and test examples"""
    id: str
    train: list[Example]
    test: list[Example]  # Test examples may not have output field
    
    @property
    def num_train(self) -> int:
        return len(self.train)
    
    @property
    def num_test(self) -> int:
        return len(self.test)


def load_challenges_from_arc_prize_json(
    challenges_path: Path,
    limit: Optional[int] = None,
    offset: int = 0,
    challenge_ids: Optional[set[str]] = None,
    solutions_path: Optional[Path] = None
) -> dict[str, Challenge]:
    """
    Load challenges from arc-prize competition JSON file.
    
    File format:
    {
        "challenge_id": {
            "train": [{"input": grid, "output": grid}, ...],
            "test": [{"input": grid}, ...]  # No output field
        },
        ...
    }
    
    Args:
        challenges_path: Path to JSON file (e.g., arc-agi_evaluation_challenges.json)
        limit: Maximum number of challenges to load (None for all)
        offset: Offset to start from (for pagination)
        challenge_ids: Optional set of specific challenge IDs to load
        solutions_path: Optional path to solutions JSON file (e.g., arc-agi_evaluation_solutions.json)
                       If provided, test outputs will be loaded from this file
        
    Returns:
        Dictionary mapping challenge_id to Challenge objects
    """
    if not challenges_path.exists():
        raise FileNotFoundError(f"Challenges file not found: {challenges_path}")
    
    logger.info(f"Loading challenges from {challenges_path}")
    
    with open(challenges_path) as f:
        raw_data = json.load(f)
    
    # Load solutions if provided
    solutions_data = {}
    if solutions_path and solutions_path.exists():
        logger.info(f"Loading solutions from {solutions_path}")
        with open(solutions_path) as f:
            solutions_data = json.load(f)
    elif solutions_path:
        logger.warning(f"Solutions file not found: {solutions_path}")
    
    challenges = {}
    
    # Filter by challenge_ids if provided
    if challenge_ids:
        raw_data = {k: v for k, v in raw_data.items() if k in challenge_ids}
    
    # Get list of challenge IDs
    challenge_id_list = list(raw_data.keys())
    logger.debug(f"Total challenges in file: {len(challenge_id_list)}")
    
    # Apply offset and limit
    if offset > 0:
        challenge_id_list = challenge_id_list[offset:]
        logger.debug(f"After offset {offset}: {len(challenge_id_list)} challenges")
    if limit:
        challenge_id_list = challenge_id_list[:limit]
        logger.debug(f"After limit {limit}: {len(challenge_id_list)} challenges")
    
    # Load each challenge
    for challenge_id in challenge_id_list:
        raw_challenge = raw_data[challenge_id]
        
        # Parse train examples (have input and output)
        train_examples = []
        for train_item in raw_challenge.get("train", []):
            train_examples.append(Example(
                input=train_item["input"],
                output=train_item["output"]
            ))
        
        # Parse test examples (may have outputs from solutions file)
        test_examples = []
        test_solutions = solutions_data.get(challenge_id, [])
        for test_idx, test_item in enumerate(raw_challenge.get("test", [])):
            # Get output from solutions file if available
            output = None
            if test_idx < len(test_solutions):
                output = test_solutions[test_idx]
            
            test_examples.append(Example(
                input=test_item["input"],
                output=output
            ))
        
        challenges[challenge_id] = Challenge(
            id=challenge_id,
            train=train_examples,
            test=test_examples
        )
    
    logger.info(f"Loaded {len(challenges)} challenges from {challenges_path}")
    if solutions_path and solutions_data:
        logger.info(f"  Loaded test outputs for {len([c for c in challenges.values() if any(t.output is not None for t in c.test)])} challenges")
    if limit:
        logger.info(f"  Limit: {limit}, Offset: {offset}")
    if challenge_ids:
        logger.info(f"  Filtered to {len(challenge_ids)} specific challenge IDs")
    
    return challenges


def load_challenge_from_file(file_path: Path) -> Challenge:
    """Load a single challenge from legacy JSON file format (individual file per challenge)"""
    with open(file_path) as f:
        data = json.load(f)
    
    # Add id from filename
    challenge_id = file_path.stem
    data["id"] = challenge_id
    
    # Convert to Example objects
    train_examples = [Example(**ex) for ex in data.get("train", [])]
    test_examples = [Example(**ex) for ex in data.get("test", [])]
    
    return Challenge(
        id=challenge_id,
        train=train_examples,
        test=test_examples
    )


def load_challenges_from_dir(dir_path: Path) -> dict[str, Challenge]:
    """Load all challenges from a directory (legacy format: individual JSON files)"""
    challenges = {}
    
    for file_path in sorted(dir_path.glob("*.json")):
        challenge = load_challenge_from_file(file_path)
        challenges[challenge.id] = challenge
    
    logger.info(f"Loaded {len(challenges)} challenges from {dir_path}")
    return challenges



def load_challenge(
    challenge_id: str,
    challenges_path: Optional[Path] = None
) -> Challenge:
    """
    Load a specific challenge by ID.
    
    Supports both legacy format and arc-prize JSON format.
    
    Args:
        challenge_id: Challenge ID (e.g., "00d62c1b")
        dataset: "training" or "evaluation" (legacy format only)
        version: 1 for ARC-AGI-1, 2 for ARC-AGI-2 (legacy format only)
        challenges_path: Path to arc-prize JSON file (if using competition format)
        
    Returns:
        Challenge object
    """
    if challenges_path:
        # Load from arc-prize JSON file
        challenges = load_challenges_from_arc_prize_json(
            challenges_path,
            challenge_ids={challenge_id}
        )
        if challenge_id not in challenges:
            raise ValueError(f"Challenge {challenge_id} not found in {challenges_path}")
        return challenges[challenge_id]



if __name__ == "__main__":
    # Test data loading
    print("=== Testing Data Loader ===")
    
    # Test loading from default arc-prize file
    print("\n1. Testing arc-prize format (default)...")
    if DEFAULT_CHALLENGES_FILE.exists():
        challenges = load_challenges_from_arc_prize_json(DEFAULT_CHALLENGES_FILE, limit=2)
        print(f"Loaded {len(challenges)} challenges from {DEFAULT_CHALLENGES_FILE}")
        for challenge_id, challenge in challenges.items():
            print(f"  {challenge_id}: {challenge.num_train} train, {challenge.num_test} test")
    else:
        print(f"  Default file not found: {DEFAULT_CHALLENGES_FILE}")
    