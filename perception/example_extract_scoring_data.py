#!/usr/bin/env python3
"""
Extract scoring data from modality vision experiment results.

This script uses an LLM to extract structured claims from free-text descriptions
and compare them with ground truth features to generate scoring data.

Usage:
    python extract_scoring_data.py <challenge_id> --experiment-dirs <dir1> <dir2> ... --ground-truth-file <gt_file.json>
    
Example:
    python extract_scoring_data.py 13e47133 \
        --experiment-dirs data/modality_vision_20251107_2355 data/modality_vision_20251108_1148 data/modality_vision_20251108_1205 \
        --ground-truth-file ground_truth_13e47133.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import litellm

# Add parent directory to path so we can import from src
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from langfuse import propagate_attributes
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    from contextlib import contextmanager
    @contextmanager
    def propagate_attributes(**kwargs):
        yield

from src import logger
from src.nodes.hypothesis_fast_nodes import GEMINI_MODEL, TEMPERATURE

# Setup litellm
litellm.drop_params = True

# Map experiment modality names to display names
MODALITY_NAME_MAP = {
    "row_only": "Row-only",
    "col_only": "Column-only",
    "ascii": "ASCII",
    "json": "JSON",
    "image_14x14": "Image 14×14",
    "image_15x15": "Image 15×15",
    "image_16x16": "Image 16×16",
    "image_17x17": "Image 17×17",
    "image_24x24": "24×24-1148",  # Will be differentiated by experiment dir
    "image_768x768": "Image 768×768",
}


def load_experiment_results(challenge_id: str, experiment_dirs: List[Path]) -> Dict[str, Dict[str, str]]:
    """
    Load descriptions from multiple experiment directories.
    
    Returns a dict mapping modality -> description_type -> description_text
    """
    results = {}
    
    for exp_dir in experiment_dirs:
        if not exp_dir.exists():
            logger.warning(f"Experiment directory does not exist: {exp_dir}")
            continue
        
        challenge_dir = exp_dir / challenge_id
        result_file = challenge_dir / "result.json"
        
        if not result_file.exists():
            logger.warning(f"Result file not found: {result_file}")
            continue
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Determine experiment identifier for 24x24 modalities
        exp_identifier = exp_dir.name.split('_')[-1] if '_' in exp_dir.name else ""
        
        # Load input descriptions
        for modality, desc_data in data.get("input_descriptions", {}).items():
            description = desc_data.get("description", "")
            if description:
                # Handle 24x24 special case
                if modality == "image_24x24":
                    display_name = f"24×24-{exp_identifier}"
                else:
                    display_name = MODALITY_NAME_MAP.get(modality, modality)
                
                if display_name not in results:
                    results[display_name] = {}
                results[display_name]["input"] = description
        
        # Load output descriptions (if needed)
        for modality, desc_data in data.get("output_descriptions", {}).items():
            description = desc_data.get("description", "")
            if description:
                if modality == "image_24x24":
                    display_name = f"24×24-{exp_identifier}"
                else:
                    display_name = MODALITY_NAME_MAP.get(modality, modality)
                
                if display_name not in results:
                    results[display_name] = {}
                results[display_name]["output"] = description
    
    return results


EXTRACTION_PROMPT_TEMPLATE = """You are an expert at analyzing visual descriptions and extracting structured claims.

**Ground Truth Features:**
{ground_truth}

**Scoring Guide:**
{scoring_guide}

**LLM Description:**
{description}

**Task:**
Extract the claims made by the LLM about the features described in the Ground Truth. For each feature type, extract:
1. What the LLM claims to see (coordinates, colors, shapes, etc.)
2. Any errors or discrepancies compared to ground truth

**Output Format:**
Provide a JSON object with the extracted claims. The structure should match the scoring guide's feature types.

Example output format:
{example_format}

Be precise and extract all claims, including incorrect ones. Use the exact coordinate notation from the description (spreadsheet notation like A1, B2, etc.)."""


async def extract_claims_with_llm(
    description: str,
    ground_truth: str,
    scoring_guide: str,
    example_format: str,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE
) -> Dict[str, Any]:
    """Use LLM to extract structured claims from free-text description."""
    
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        ground_truth=ground_truth,
        scoring_guide=scoring_guide,
        description=description,
        example_format=example_format
    )
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    metadata_dict = {
        "generation_name": "extract_scoring_data",
        "phase": "extraction",
        "challenge_id": "unknown"  # Will be set by caller
    }
    
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        metadata=metadata_dict,
        response_format={"type": "json_object"}
    )
    
    result_text = response.choices[0].message.content
    
    try:
        return json.loads(result_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response: {result_text}")
        return {"error": "Failed to parse JSON", "raw_response": result_text}


async def process_challenge(
    challenge_id: str,
    experiment_dirs: List[Path],
    ground_truth_file: Path,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Process a challenge and extract scoring data for all modalities."""
    
    # Load ground truth and scoring guide
    with open(ground_truth_file, 'r') as f:
        gt_data = json.load(f)
    
    ground_truth = gt_data.get("ground_truth_features", "")
    scoring_guide = gt_data.get("scoring_guide", "")
    example_format = gt_data.get("example_format", "{}")
    
    # Load experiment results
    logger.info(f"Loading experiment results for challenge {challenge_id}")
    experiment_results = load_experiment_results(challenge_id, experiment_dirs)
    
    if not experiment_results:
        logger.error(f"No experiment results found for challenge {challenge_id}")
        return {}
    
    logger.info(f"Found results for {len(experiment_results)} modalities")
    
    # Extract claims for each modality
    modalities_data = {}
    
    with propagate_attributes(session_id=f"extract_scoring-{challenge_id}"):
        for modality_name, descriptions in experiment_results.items():
            logger.info(f"Processing modality: {modality_name}")
            
            # Use input description (or output if input not available)
            description = descriptions.get("input", descriptions.get("output", ""))
            
            if not description:
                logger.warning(f"No description found for {modality_name}")
                continue
            
            # Extract claims using LLM
            claims = await extract_claims_with_llm(
                description=description,
                ground_truth=ground_truth,
                scoring_guide=scoring_guide,
                example_format=example_format,
                model=model,
                temperature=temperature
            )
            
            if "error" in claims:
                logger.error(f"Error extracting claims for {modality_name}: {claims.get('error')}")
                modalities_data[modality_name] = {"error": claims.get("error")}
            else:
                modalities_data[modality_name] = claims
    
    result = {
        "challenge_id": challenge_id,
        "modalities_data": modalities_data
    }
    
    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract scoring data from modality vision experiment results"
    )
    parser.add_argument(
        "challenge_id",
        type=str,
        help="Challenge ID (e.g., 13e47133)"
    )
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to experiment directories (e.g., data/modality_vision_20251107_2355)"
    )
    parser.add_argument(
        "--ground-truth-file",
        type=str,
        required=True,
        help="Path to JSON file containing ground truth features and scoring guide"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Model to use (default: {GEMINI_MODEL})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Temperature setting (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save extracted scoring data (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    experiment_dirs = [Path(d) for d in args.experiment_dirs]
    ground_truth_file = Path(args.ground_truth_file)
    
    if not ground_truth_file.exists():
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        sys.exit(1)
    
    output_file = Path(args.output_file) if args.output_file else None
    
    # Run extraction
    result = asyncio.run(process_challenge(
        challenge_id=args.challenge_id,
        experiment_dirs=experiment_dirs,
        ground_truth_file=ground_truth_file,
        model=args.model,
        temperature=args.temperature,
        output_file=output_file
    ))
    
    # Print result if not saving to file
    if not output_file:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

