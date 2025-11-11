#!/usr/bin/env python3
"""
Simple script to load JSON files and print modality descriptions one at a time.

Usage:
    python print_modality_description.py <challenge_id> <modality> [input|output] [--experiment-dir <dir>]
    
Example:
    python print_modality_description.py 13e47133 row_only
    python print_modality_description.py 13e47133 col_only output
    python print_modality_description.py 13e47133 image_24x24 --experiment-dir data/modality_vision_20251108_1148
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add parent directory to path so we can import from src
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Challenge directories
CHALLENGES = [
    "13e47133",
    "135a2760",
    "136b0064",
    "142ca369",
    "0934a4d8"
]

# Available modalities
MODALITIES = [
    "row_only",
    "col_only",
    "ascii",
    "json",
    "image_14x14",
    "image_15x15",
    "image_16x16",
    "image_17x17",
    "image_24x24",
    "image_768x768"
]


def find_experiment_dirs(base_dir: Optional[Path] = None) -> List[Path]:
    """Find all modality_vision experiment directories."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "data"
    else:
        base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return []
    
    # Find all directories matching modality_vision_* pattern
    experiment_dirs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("modality_vision_")
    ]
    
    return sorted(experiment_dirs)


def load_challenge_data(challenge_id: str, experiment_dir: Optional[Path] = None) -> dict:
    """
    Load JSON data for a given challenge ID from experiment directory.
    
    Args:
        challenge_id: The challenge ID
        experiment_dir: Specific experiment directory to load from. If None, searches all experiment dirs.
    """
    base_dir = Path(__file__).parent
    
    # If specific experiment dir provided, use it
    if experiment_dir:
        experiment_dir = Path(experiment_dir)
        result_file = experiment_dir / challenge_id / "result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    # Otherwise, search in data directory
    data_dir = base_dir / "data"
    experiment_dirs = find_experiment_dirs(data_dir)
    
    # Try each experiment directory
    for exp_dir in experiment_dirs:
        result_file = exp_dir / challenge_id / "result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
    
    # Fallback: try local directory structure (for backward compatibility)
    result_file = base_dir / challenge_id / "result.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(
        f"Result file not found for challenge {challenge_id}. "
        f"Searched in: {[str(d) for d in experiment_dirs]}"
    )


def print_modality_description(
    challenge_id: str,
    modality: str,
    description_type: str = "input",
    experiment_dir: Optional[Path] = None
):
    """
    Print the description for a specific challenge, modality, and type.
    
    Args:
        challenge_id: The challenge ID (e.g., "13e47133")
        modality: The modality name (e.g., "row_only")
        description_type: Either "input" or "output" (default: "input")
        experiment_dir: Optional specific experiment directory to load from
    """
    try:
        data = load_challenge_data(challenge_id, experiment_dir)
        
        # Get the correct descriptions dict
        descriptions_key = f"{description_type}_descriptions"
        if descriptions_key not in data:
            print(f"Error: {descriptions_key} not found in challenge {challenge_id}")
            return
        
        descriptions = data[descriptions_key]
        
        if modality not in descriptions:
            print(f"Error: Modality '{modality}' not found in challenge {challenge_id}")
            print(f"Available modalities: {', '.join(descriptions.keys())}")
            return
        
        description = descriptions[modality].get("description", "")
        
        if not description:
            print(f"Error: No description found for {challenge_id} {modality}")
            return
        
        # Print in the requested format
        print(f"{challenge_id}{description_type}: {modality}: {description}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")


def list_available(experiment_dir: Optional[Path] = None):
    """List all available challenges and their modalities."""
    print("Available challenges and modalities:\n")
    
    for challenge_id in CHALLENGES:
        try:
            data = load_challenge_data(challenge_id, experiment_dir)
            input_modalities = list(data.get("input_descriptions", {}).keys())
            output_modalities = list(data.get("output_descriptions", {}).keys())
            
            print(f"{challenge_id}:")
            print(f"  Input modalities: {', '.join(input_modalities)}")
            print(f"  Output modalities: {', '.join(output_modalities)}")
            print()
        except Exception as e:
            print(f"{challenge_id}: Error loading - {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Print modality descriptions from experiment results"
    )
    parser.add_argument(
        "challenge_id",
        type=str,
        nargs="?",
        help="Challenge ID (e.g., 13e47133)"
    )
    parser.add_argument(
        "modality",
        type=str,
        nargs="?",
        help="Modality name (e.g., row_only)"
    )
    parser.add_argument(
        "description_type",
        type=str,
        nargs="?",
        default="input",
        choices=["input", "output"],
        help="Description type: input or output (default: input)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Specific experiment directory to load from (e.g., data/modality_vision_20251107_2355)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available challenges and modalities"
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None
    
    if args.list:
        list_available(experiment_dir)
        sys.exit(0)
    
    if not args.challenge_id or not args.modality:
        parser.print_help()
        print("\nExample:")
        print("  python print_modality_description.py 13e47133 row_only")
        print("  python print_modality_description.py 13e47133 col_only output")
        print("  python print_modality_description.py 13e47133 image_24x24 --experiment-dir data/modality_vision_20251108_1148")
        print("\nTo list all available challenges and modalities:")
        print("  python print_modality_description.py --list")
        sys.exit(1)
    
    print_modality_description(
        args.challenge_id,
        args.modality,
        args.description_type,
        experiment_dir
    )


if __name__ == "__main__":
    main()

