"""
Modality Vision Experiment: Test model's ability to "see" things under different modalities.

This experiment:
1. Takes training example 1 (index 0) input and output from 5 challenges
2. Presents each puzzle in different single modalities:
   - row_only: Row-wise text format
   - col_only: Column-wise text format
   - ascii: ASCII format (space-separated integers per row, like arc-lang-public)
   - json: Raw JSON format (list of lists, like arc_agi)
   - image_14x14, image_15x15, image_16x16, image_17x17: Images with different resolutions
3. Asks the LLM to provide a very detailed description of what it sees
4. Saves descriptions for comparison

Usage:
uv run python test_modality_vision_experiment.py --challenge-ids <id1> <id2> ... --output-dir <output_dir> --model <model> --temperature <temperature> --rpm <rpm>

Example:
uv run python test_modality_vision_experiment.py --challenge-ids 13e47133 0934a4d8 135a2760 136b0064 142ca369 --rpm 60
"""

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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

from src import logger, get_run_log_file
from src.utils.data_loader import load_challenge, DEFAULT_CHALLENGES_FILE
from src.utils.modality_encoder import (
    format_grid_row_wise,
    format_grid_column_wise,
    create_16x16_image_with_coordinates,
    create_image_without_coordinates,
    encode_image_to_base64
)
from src.nodes.hypothesis_fast_nodes import (
    GEMINI_MODEL,
    TEMPERATURE,
)

# Setup litellm
litellm.drop_params = True

# Modality types to test
MODALITY_TYPES = [
    "row_only",
    "col_only",
    "ascii",
    "json",
    "image_14x14",
    "image_15x15",
    "image_16x16",
    "image_17x17",
    "image_768x768",
    "image_24x24",
    # Images without coordinate annotations (for comparison)
    "image_14x14_no_coords",
    "image_15x15_no_coords",
    "image_16x16_no_coords",
    "image_17x17_no_coords",
    "image_768x768_no_coords",
    "image_24x24_no_coords",
]

# Description prompt
DESCRIPTION_PROMPT = """You are an expert at visual analysis and pattern recognition. 

I will show you a puzzle grid in a specific format. Your task is to provide a VERY DETAILED description of what you see, including:

**Color Mapping (if numbers are shown):**
- 0 = black (empty/background)
- 1 = blue
- 2 = red
- 3 = green
- 4 = yellow
- 5 = gray
- 6 = magenta
- 7 = orange
- 8 = teal
- 9 = maroon

**Description Requirements:**

1. **Objects and Shapes**: What objects, shapes, or patterns do you see? Describe their exact forms, sizes, and boundaries.

2. **Locations**: Where are these objects located? Use specific coordinates or relative positions. Use spreadsheet notation (A1, B2, etc.) for all coordinates. Otherwise, describe positions clearly.

3. **Colors**: What colors are present? Map each color to its numeric value (0-9) if applicable. Describe object and pattern's colors, and their distribution.

4. **Relationships**: What relationships exist between objects?
   - Spatial relationships (above, below, left, right, adjacent, overlapping, distance)
   - Size relationships (larger, smaller, same size)
   - Color relationships (same color, different colors)
   - Structural relationships (connected, separated, nested, aligned)

5. **Patterns**: Are there any repeating patterns, symmetries, or regularities?

6. **Grid Structure**: Describe the overall grid structure:
   - Dimensions (rows × columns)
   - Empty vs filled regions
   - Background color
   - Any boundaries or frames

7. **Details**: Any other notable details that might be important for understanding the puzzle.

Be as thorough and precise as possible. Your description should be detailed enough that someone reading it could reconstruct the grid (or at least understand its key features).

Now, here is the puzzle grid:"""


class RateLimiter:
    """Rate limiter for API calls based on requests per minute (RPM)."""
    
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


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None
_original_acompletion = None
_progress_tracker = None


class ProgressTracker:
    """Track progress of API calls and experiment steps."""
    
    def __init__(self, total_api_calls: int):
        self.total_api_calls = total_api_calls
        self.completed_api_calls = 0
        self.current_step = "Initializing"
        self.lock = asyncio.Lock()
    
    async def increment(self, step_name: str = None):
        async with self.lock:
            self.completed_api_calls += 1
            if step_name:
                self.current_step = step_name
    
    def print_progress(self):
        remaining = self.total_api_calls - self.completed_api_calls
        progress_pct = (self.completed_api_calls / self.total_api_calls * 100) if self.total_api_calls > 0 else 0.0
        print(f"\r[Progress] {self.completed_api_calls}/{self.total_api_calls} API calls ({progress_pct:.1f}%) | Remaining: {remaining} | {self.current_step}", end="", flush=True)


def format_grid_ascii(grid: List[List[int]]) -> str:
    """Format grid as ASCII (space-separated integers per row, like arc-lang-public)."""
    return "\n".join(" ".join(str(val) for val in row) for row in grid)


def format_grid_json(grid: List[List[int]]) -> str:
    """Format grid as JSON (raw list of lists, like arc_agi)."""
    return json.dumps(grid)


async def create_modality_message(
    grid: List[List[int]],
    modality_type: str,
    grid_label: str = "Grid"
) -> List[Dict[str, Any]]:
    """Create a message with the grid in the specified modality format."""
    content_parts = []
    
    if modality_type == "row_only":
        # Format: R1: 000100300\n\nR2: 030030\n\n...
        row_wise = format_grid_row_wise(grid, prefix="R")
        content_parts.append({
            "type": "text",
            "text": f"{grid_label} (Row-wise format):\n{row_wise}"
        })
    
    elif modality_type == "col_only":
        # Format: CA: 000300\n\nCB: 030000\n\n...
        col_wise = format_grid_column_wise(grid, prefix="C")
        content_parts.append({
            "type": "text",
            "text": f"{grid_label} (Column-wise format):\n{col_wise}"
        })
    
    elif modality_type == "ascii":
        ascii_text = format_grid_ascii(grid)
        content_parts.append({
            "type": "text",
            "text": f"{grid_label} (ASCII format):\n{ascii_text}"
        })
    
    elif modality_type == "json":
        json_text = format_grid_json(grid)
        content_parts.append({
            "type": "text",
            "text": f"{grid_label} (JSON format):\n{json_text}"
        })
    
    elif modality_type.startswith("image_"):
        # Check if this is a "no_coords" variant
        has_coords = not modality_type.endswith("_no_coords")
        
        if modality_type == "image_768x768" or modality_type == "image_768x768_no_coords":
            # Calculate cell size to fill 768x768 space
            rows, cols = len(grid), len(grid[0]) if grid else 0
            if rows == 0 or cols == 0:
                resolution = 768
            else:
                # Calculate cell size to fit within 768x768
                # Use max dimension to ensure it fits
                max_dim = max(rows, cols)
                resolution = 768 // max_dim  # Integer division to get pixels per cell
            if has_coords:
                img = create_16x16_image_with_coordinates(grid, resolution=resolution)
            else:
                img = create_image_without_coordinates(grid, resolution=resolution)
            base64_img = await encode_image_to_base64(img)
            actual_width = cols * resolution if grid else resolution
            actual_height = rows * resolution if grid else resolution
            coords_note = "with coordinate annotations" if has_coords else "without coordinate annotations"
            content_parts.append({
                "type": "text",
                "text": f"{grid_label} (Image format, {resolution}x{resolution} pixels per cell, {actual_width}x{actual_height} total, optimized for Gemini 768x768 patch, {coords_note}):"
            })
        else:
            # Extract resolution (e.g., "image_16x16" -> 16, "image_16x16_no_coords" -> 16)
            base_type = modality_type.replace("_no_coords", "")
            resolution = int(base_type.split("_")[1].split("x")[0])
            if has_coords:
                img = create_16x16_image_with_coordinates(grid, resolution=resolution)
            else:
                img = create_image_without_coordinates(grid, resolution=resolution)
            base64_img = await encode_image_to_base64(img)
            coords_note = "with coordinate annotations" if has_coords else "without coordinate annotations"
            content_parts.append({
                "type": "text",
                "text": f"{grid_label} (Image format, {resolution}x{resolution} pixels per cell, {coords_note}):"
            })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
        })
    
    else:
        raise ValueError(f"Unknown modality type: {modality_type}")
    
    return content_parts


async def get_description(
    grid: List[List[int]],
    modality_type: str,
    grid_label: str,
    model: str,
    temperature: float,
    session_id: Optional[str] = None
) -> str:
    """Get detailed description from LLM for a grid in a specific modality."""
    global _rate_limiter, _progress_tracker
    
    # Create modality-specific content
    content_parts = []
    
    # Add description prompt
    content_parts.append({
        "type": "text",
        "text": DESCRIPTION_PROMPT
    })
    
    # Add grid in the specified modality
    modality_content = await create_modality_message(grid, modality_type, grid_label)
    content_parts.extend(modality_content)
    
    messages = [
        {
            "role": "user",
            "content": content_parts
        }
    ]
    
    metadata_dict = {
        "generation_name": "modality_vision_experiment",
        "phase": "description",
        "modality": modality_type,
        "grid_label": grid_label
    }
    if session_id:
        metadata_dict["session_id"] = session_id
    
    # Wait for rate limiter if needed
    if _rate_limiter:
        await _rate_limiter.wait_if_needed()
    
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        metadata=metadata_dict
    )
    
    description = response.choices[0].message.content
    
    # Update progress
    if _progress_tracker:
        await _progress_tracker.increment(f"{grid_label} - {modality_type}")
        _progress_tracker.print_progress()
    
    return description


async def process_challenge(
    challenge_id: str,
    challenge_data: Any,
    output_dir: Path,
    model: str,
    temperature: float,
    session_id: Optional[str] = None,
    modality_types: Optional[List[str]] = None,
    existing_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a single challenge: get descriptions for training example 1 in specified modalities."""
    logger.info(f"Processing challenge {challenge_id}")
    
    # Use provided modalities or default to all
    modalities_to_process = modality_types if modality_types else MODALITY_TYPES
    
    # Get training example 1 (index 0)
    if len(challenge_data.train) == 0:
        logger.warning(f"Challenge {challenge_id} has no training examples, skipping")
        return {"error": "No training examples"}
    
    example = challenge_data.train[0]
    input_grid = example.input
    output_grid = example.output
    
    # Start with existing result or create new one
    if existing_result:
        results = existing_result.copy()
        # Ensure descriptions dicts exist
        if "input_descriptions" not in results:
            results["input_descriptions"] = {}
        if "output_descriptions" not in results:
            results["output_descriptions"] = {}
    else:
        results = {
            "challenge_id": challenge_id,
            "input_descriptions": {},
            "output_descriptions": {}
        }
    
    # Get descriptions for input grid in specified modalities
    logger.info(f"  Getting input descriptions for {challenge_id} (modalities: {modalities_to_process})")
    input_tasks = []
    for modality_type in modalities_to_process:
        task = get_description(
            input_grid,
            modality_type,
            "Input",
            model,
            temperature,
            session_id=session_id
        )
        input_tasks.append((modality_type, task))
    
    # Run all input descriptions in parallel
    input_results = await asyncio.gather(*[task for _, task in input_tasks], return_exceptions=True)
    for (modality_type, _), description in zip(input_tasks, input_results):
        if isinstance(description, Exception):
            logger.error(f"    Error getting input description for {modality_type}: {description}")
            results["input_descriptions"][modality_type] = {"error": str(description)}
        else:
            results["input_descriptions"][modality_type] = {"description": description}
    
    # Get descriptions for output grid in specified modalities
    logger.info(f"  Getting output descriptions for {challenge_id}")
    output_tasks = []
    for modality_type in modalities_to_process:
        task = get_description(
            output_grid,
            modality_type,
            "Output",
            model,
            temperature,
            session_id=session_id
        )
        output_tasks.append((modality_type, task))
    
    # Run all output descriptions in parallel
    output_results = await asyncio.gather(*[task for _, task in output_tasks], return_exceptions=True)
    for (modality_type, _), description in zip(output_tasks, output_results):
        if isinstance(description, Exception):
            logger.error(f"    Error getting output description for {modality_type}: {description}")
            results["output_descriptions"][modality_type] = {"error": str(description)}
        else:
            results["output_descriptions"][modality_type] = {"description": description}
    
    # Save grid data for reference (only if not already present)
    if "input_grid" not in results:
        results["input_grid"] = input_grid
    if "output_grid" not in results:
        results["output_grid"] = output_grid
    if "grid_dimensions" not in results:
        results["grid_dimensions"] = {
            "input": {"rows": len(input_grid), "cols": len(input_grid[0]) if input_grid else 0},
            "output": {"rows": len(output_grid), "cols": len(output_grid[0]) if output_grid else 0}
        }
    
    return results


async def run_experiment(
    challenge_ids: List[str],
    output_dir: Path,
    challenges_path: Optional[Path] = None,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    rpm: Optional[int] = None,
    existing_experiment_dir: Optional[Path] = None,
    modality_types: Optional[List[str]] = None
):
    """Run the modality vision experiment for multiple challenges."""
    global _rate_limiter, _progress_tracker, _original_acompletion
    
    # Set up rate limiter
    if rpm is not None:
        _rate_limiter = RateLimiter(rpm)
        logger.info(f"Rate limiting enabled: {rpm} RPM")
        
        if _original_acompletion is None:
            _original_acompletion = litellm.acompletion
        
        async def patched_acompletion(*args, **kwargs):
            await _rate_limiter.wait_if_needed()
            return await _original_acompletion(*args, **kwargs)
        
        litellm.acompletion = patched_acompletion
        logger.info("Patched litellm.acompletion with rate limiting")
    else:
        logger.info("Rate limiting disabled (no RPM limit)")
    
    logger.info(f"Loading {len(challenge_ids)} challenges")
    logger.info(f"Using model: {model}, temperature: {temperature}")
    
    # Load challenges
    challenges = {}
    if challenges_path:
        from src.utils.data_loader import load_challenges_from_arc_prize_json
        solutions_path = challenges_path.parent / "arc-agi_evaluation_solutions.json"
        if not solutions_path.exists():
            solutions_path = None
            logger.warning(f"Solutions file not found")
        
        challenges = load_challenges_from_arc_prize_json(
            challenges_path, challenge_ids=set(challenge_ids), solutions_path=solutions_path
        )
    else:
        for challenge_id in challenge_ids:
            try:
                challenges[challenge_id] = load_challenge(challenge_id)
            except Exception as e:
                logger.error(f"Failed to load challenge {challenge_id}: {e}")
    
    logger.info(f"Loaded {len(challenges)} challenges")
    
    # Use existing experiment directory or create new one
    if existing_experiment_dir:
        experiment_dir = existing_experiment_dir
        # Load existing summary if it exists
        existing_summary = None
        summary_path = experiment_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                existing_summary = json.load(f)
            session_id = existing_summary.get("session_id", f"modality_vision-{datetime.now().strftime('%Y%m%d_%H%M')}")
        else:
            session_id = f"modality_vision-{datetime.now().strftime('%Y%m%d_%H%M')}"
        logger.info(f"Appending to existing experiment directory: {experiment_dir}")
    else:
        # Create output directory with timestamp
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = output_dir / f"modality_vision_{session_timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"modality_vision-{session_timestamp}"
        existing_summary = None
        logger.info(f"Output directory: {experiment_dir}")
    
    logger.info(f"Langfuse session_id: {session_id}")
    
    # Use provided modalities or default to all
    modalities_to_process = modality_types if modality_types else MODALITY_TYPES
    
    # Calculate total API calls needed
    # For each challenge: 2 grids (input + output) × number of modalities
    total_api_calls = len(challenges) * 2 * len(modalities_to_process)
    
    logger.info(f"Total API calls needed: {total_api_calls}")
    
    # Initialize progress tracker
    _progress_tracker = ProgressTracker(total_api_calls)
    print(f"\n[Progress] 0/{total_api_calls} API calls (0.0%) | Remaining: {total_api_calls} | Initializing")
    
    # Process all challenges
    all_results = {}
    
    # Load existing results if appending
    if existing_summary:
        all_results = existing_summary.get("results", {}).copy()
        # Load individual challenge results to merge
        for challenge_id in challenge_ids:
            challenge_dir = experiment_dir / challenge_id
            result_path = challenge_dir / "result.json"
            if result_path.exists():
                with open(result_path, "r") as f:
                    all_results[challenge_id] = json.load(f)
    else:
        all_results = {}
    
    with propagate_attributes(session_id=session_id):
        for challenge_id in challenge_ids:
            if challenge_id not in challenges:
                logger.warning(f"Challenge {challenge_id} not found, skipping")
                continue
            
            try:
                # Get existing result for this challenge if appending
                existing_result = all_results.get(challenge_id)
                
                result = await process_challenge(
                    challenge_id,
                    challenges[challenge_id],
                    experiment_dir,
                    model=model,
                    temperature=temperature,
                    session_id=session_id,
                    modality_types=modalities_to_process,
                    existing_result=existing_result
                )
                all_results[challenge_id] = result
                
                # Save individual challenge result
                challenge_dir = experiment_dir / challenge_id
                challenge_dir.mkdir(exist_ok=True)
                with open(challenge_dir / "result.json", "w") as f:
                    json.dump(result, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing challenge {challenge_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[challenge_id] = {"error": str(e)}
    
    # Save summary (merge with existing if appending)
    if existing_summary:
        # Merge modality types
        existing_modalities = set(existing_summary.get("modality_types", []))
        new_modalities = set(modalities_to_process)
        all_modalities = sorted(list(existing_modalities | new_modalities))
        
        summary = {
            "session_id": session_id,
            "model": model,  # Use current model (may differ if appending)
            "temperature": temperature,  # Use current temperature
            "rpm": rpm,  # Use current rpm
            "challenge_ids": sorted(list(set(existing_summary.get("challenge_ids", [])) | set(challenge_ids))),
            "modality_types": all_modalities,
            "results": all_results
        }
    else:
        summary = {
            "session_id": session_id,
            "model": model,
            "temperature": temperature,
            "rpm": rpm,
            "challenge_ids": challenge_ids,
            "modality_types": modalities_to_process,
            "results": all_results
        }
    
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print()  # New line after progress
    logger.info(f"Experiment complete. Results saved to: {experiment_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test model's ability to see things under different modalities"
    )
    parser.add_argument(
        "--challenge-ids",
        type=str,
        nargs="+",
        required=True,
        help="Challenge IDs to test (e.g., 13e47133 0934a4d8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data). Results will be saved to data/modality_vision_{timestamp}/"
    )
    parser.add_argument(
        "--challenges-file",
        type=Path,
        default=None,
        help=f"Path to challenges JSON file (default: {DEFAULT_CHALLENGES_FILE})"
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
        "--rpm",
        type=int,
        default=None,
        help="Maximum requests per minute (RPM) for rate limiting. Recommended: 60-120 RPM."
    )
    parser.add_argument(
        "--existing-experiment-dir",
        type=str,
        default=None,
        help="Path to existing experiment directory to append results to (instead of creating new one)"
    )
    parser.add_argument(
        "--modality-types",
        type=str,
        nargs="+",
        default=None,
        help="Specific modality types to process (default: all). Example: --modality-types image_768x768"
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
    
    # Determine challenges file
    if args.challenges_file is None:
        challenges_file = DEFAULT_CHALLENGES_FILE
    else:
        challenges_file = args.challenges_file
    
    output_base_dir = Path(args.output_dir)
    
    # Determine existing experiment directory if provided
    existing_experiment_dir = None
    if args.existing_experiment_dir:
        existing_experiment_dir = Path(args.existing_experiment_dir)
        if not existing_experiment_dir.exists():
            raise ValueError(f"Existing experiment directory does not exist: {existing_experiment_dir}")
    
    # Run experiment
    asyncio.run(run_experiment(
        args.challenge_ids,
        output_base_dir,
        challenges_file if challenges_file.exists() else None,
        model=args.model,
        temperature=args.temperature,
        rpm=args.rpm,
        existing_experiment_dir=existing_experiment_dir,
        modality_types=args.modality_types
    ))


if __name__ == "__main__":
    main()

