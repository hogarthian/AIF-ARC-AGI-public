"""
Modality Experiment Script: Run experiments with different input modalities.

This script supports:
1. Testing any modality type (json_only, row_col_json, row_col_json_image, image_json, etc.)
2. Ascending or descending order (default: ascending)
3. Single or multiple trials for diversity analysis
4. Saves complete results including hypothesis, grids, and scores

API Call Pattern:
- 1 API call for hypothesis generation
- For each training example: 1 call × num_train (held-out validation)
- For each test case: 1 call × num_test
- Same for "reduced" version (no training examples in context)
- Total API calls = 1 + num_train + num_test + num_train + num_test
  Example: 1 + 3 + 2 + 3 + 2 = 11 calls per trial

Output Structure:
- {challenge_id}/{timestamp}/
  - results_combined.json (summary with all trials)
  - results_trial01.json, results_trial02.json, ... (individual trial summaries)
  - {modality}_{order}_trial01/ (or {modality}_{order} if single trial)
    - hypothesis.json (LLM response + reasoning_content)
    - grids.json (all grids: E0, E1, T0, T1 as keys)
    - results.json (all scores: E0, E1, T0, T1 as keys)
    - grids_reduced.json, results_reduced.json (reduced context versions)

Usage:
uv run python run_modality_experiment.py --challenge-id <challenge_id> --modality-type <modality> --output-dir <output_dir> --challenges-file <challenges_file> --model <model> --temperature <temperature> --rpm <rpm> --num-trials <num_trials> --order <ascending|descending>

Examples:
# Single trial with json_only modality (ascending order)
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type json_only --rpm 60

# Multiple trials for diversity analysis
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --num-trials 4 --temperature 0.7 --rpm 60

# Reproducing the published experiments:
# 1. Cross-order experiment (20251105_1142) - uses test_modality_experiment.py (not included in public repo)
#    This tested all 7 modalities with both ascending and descending orders
#    Command: See test_modality_experiment.py documentation

# 2. JSON-only experiments (20251108_1435, 20251108_1444, 20251108_1455, 20251108_1504)
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type json_only --rpm 60
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json --rpm 60
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --rpm 60
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type image_json --rpm 60

# 3. Repeat sampling experiment (20251108_2058) - 4 trials with temperature 0.7
uv run python run_modality_experiment.py --challenge-id 13e47133 --modality-type row_col_json_image --num-trials 4 --temperature 0.7 --rpm 60

Note: The cross-order experiment (20251105_1142) was run using test_modality_experiment.py which tests
both ascending and descending orders. After concluding that order doesn't significantly affect quality,
we created this simplified script that only tests one order at a time (default: ascending).
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
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
from src.utils.modality_encoder import create_prompt_messages
from src.utils.follow_instructions import follow_instructions_to_generate_grid
from src.utils.scoring_engine import get_grid_similarity
from src.nodes.models import TransformationWithUncertainty
from src.nodes.hypothesis_fast_nodes import (
    GEMINI_MODEL,
    TEMPERATURE,
    HYPOTHESIS_FAST_SYSTEM_PROMPT
)

# Setup litellm
litellm.drop_params = True

# Fixed modality and order (can be overridden via command line)
DEFAULT_MODALITY_TYPE = "row_col_image"
DEFAULT_EXAMPLE_ORDER = None  # Ascending order (None = ascending, -1 = descending)


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


async def generate_hypothesis(
    challenge_data: Any,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    session_id: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Tuple[TransformationWithUncertainty, str | None]:
    """Generate hypothesis using Gemini with specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"Generating hypothesis: modality={modality_type}, order={order_name}, model={model}, temperature={temperature}")
    
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    
    DynamicModel = TransformationWithUncertainty.create_dynamic_model(num_train, num_test)
    
    modality_messages_list = await create_prompt_messages(
        challenge_data, modality_type, example_order=example_order
    )
    
    messages = [
        {"role": "system", "content": HYPOTHESIS_FAST_SYSTEM_PROMPT}
    ]
    messages.extend(modality_messages_list)
    
    metadata_dict = {
        "generation_name": "simplified_exp_hypothesis",
        "phase": "simplified_experiment",
        "model_type": "vision",
        "modality": modality_type,
        "challenge_id": getattr(challenge_data, 'id', 'unknown')
    }
    if session_id:
        metadata_dict["session_id"] = session_id
    
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        reasoning_effort="high",
        response_format=DynamicModel,
        metadata=metadata_dict
    )
    
    content = response.choices[0].message.content
    try:
        dynamic_instance = DynamicModel.model_validate_json(content)
    except Exception as e:
        logger.error(f"Failed to parse hypothesis JSON: {e}")
        logger.error(f"Response content (first 500 chars): {content[:500]}")
        raise
    
    try:
        # Log all attributes of dynamic_instance for debugging
        if hasattr(dynamic_instance, '__dict__'):
            all_attrs = {k: str(v)[:100] if isinstance(v, str) else type(v).__name__ for k, v in dynamic_instance.__dict__.items()}
            logger.debug(f"Dynamic instance attributes: {all_attrs}")
        elif hasattr(dynamic_instance, 'model_fields'):
            logger.debug(f"Dynamic instance model_fields: {list(dynamic_instance.model_fields.keys())}")
        
        belief = TransformationWithUncertainty.from_dynamic_model(dynamic_instance, num_train, num_test)
    except KeyError as e:
        logger.error(f"KeyError in from_dynamic_model: {e}")
        logger.error(f"Expected keys: general, E0-E{num_train-1}, T0-T{num_test-1}")
        if hasattr(dynamic_instance, '__dict__'):
            logger.error(f"Actual keys in dynamic_instance: {list(dynamic_instance.__dict__.keys())}")
        elif hasattr(dynamic_instance, 'model_fields'):
            logger.error(f"Model fields: {list(dynamic_instance.model_fields.keys())}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Failed to convert dynamic model to TransformationWithUncertainty: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        if hasattr(dynamic_instance, '__dict__'):
            logger.error(f"Dynamic instance keys: {list(dynamic_instance.__dict__.keys())}")
        elif hasattr(dynamic_instance, 'model_fields'):
            logger.error(f"Model fields: {list(dynamic_instance.model_fields.keys())}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    
    if _progress_tracker:
        await _progress_tracker.increment("Hypothesis generation")
        _progress_tracker.print_progress()
    
    return belief, reasoning_content


async def run_held_out_validation_reduced(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    output_dir: Path,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    session_id: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run held-out validation (reduced context) for specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running held-out validation (reduced): {modality_type} ({order_name})")
    
    training_examples = challenge_data.train
    transform_instructions = belief.transform_instructions
    working_hypothesis = belief.working_hypothesis
    
    held_out_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for held_out_idx in range(len(training_examples)):
        logger.info(f"    Holding out example {held_out_idx} (reduced)...")
        
        held_out_example = training_examples[held_out_idx]
        context_examples = training_examples[:held_out_idx] + training_examples[held_out_idx+1:]
        
        if not context_examples:
            logger.warning(f"Skipping held-out validation for example {held_out_idx} (only 1 training example, no context)")
            grids_dict[f"E{held_out_idx}"] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "skipped": "No context examples"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "skipped": True,
                "similarity": 0.0
            })
            continue
        
        held_out_instructions = {}
        if "general" in transform_instructions:
            held_out_instructions["general"] = transform_instructions["general"]
        
        held_out_key = f"E{held_out_idx}"
        grid_key = held_out_key
        if held_out_key in transform_instructions:
            held_out_instructions["T0"] = transform_instructions[held_out_key]
        else:
            logger.error(f"Held-out validation (reduced): Example instruction {held_out_key} not found")
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "error": "Missing instruction"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "error": "Missing instruction",
                "similarity": 0.0
            })
            continue
        
        try:
            step_name = f"Hold-out E{held_out_idx} reduced"
            
            grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
                instructions=held_out_instructions,
                training_examples=context_examples,
                test_input_grid=held_out_example.input,
                challenge_data=challenge_data,
                is_held_out=True,
                example_order=example_order,
                working_hypothesis=working_hypothesis,
                modality_type=modality_type,
                include_training_examples=False,
                session_id=session_id
            )
            
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            expected_grid = held_out_example.output
            similarity = get_grid_similarity(expected_grid, grid)
            
            result = {
                "held_out_idx": held_out_idx,
                "similarity": similarity,
                "uncertainty": uncertainty,
                "reasoning_content": reasoning
            }
            
            held_out_results.append(result)
            
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": expected_grid,
                "generated": grid
            }
            
        except Exception as e:
            logger.error(f"    Error in held-out validation (reduced) for example {held_out_idx}: {e}")
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "error": str(e)
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "error": str(e),
                "similarity": 0.0
            })
    
    return held_out_results, grids_dict


async def run_held_out_validation(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    output_dir: Path,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    session_id: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run held-out validation for specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running held-out validation: {modality_type} ({order_name})")
    
    training_examples = challenge_data.train
    transform_instructions = belief.transform_instructions
    working_hypothesis = belief.working_hypothesis
    
    held_out_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for held_out_idx in range(len(training_examples)):
        logger.info(f"    Holding out example {held_out_idx}...")
        
        held_out_example = training_examples[held_out_idx]
        context_examples = training_examples[:held_out_idx] + training_examples[held_out_idx+1:]
        
        if not context_examples:
            logger.warning(f"Skipping held-out validation for example {held_out_idx} (only 1 training example, no context)")
            grids_dict[f"E{held_out_idx}"] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "skipped": "No context examples"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "skipped": True,
                "similarity": 0.0
            })
            continue
        
        held_out_instructions = {}
        if "general" in transform_instructions:
            held_out_instructions["general"] = transform_instructions["general"]
        
        for key in transform_instructions.keys():
            if key.startswith("E") and key != f"E{held_out_idx}":
                held_out_instructions[key] = transform_instructions[key]
        
        held_out_key = f"E{held_out_idx}"
        grid_key = held_out_key
        if held_out_key in transform_instructions:
            held_out_instructions["T0"] = transform_instructions[held_out_key]
        else:
            logger.error(f"Held-out validation: Example instruction {held_out_key} not found")
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "error": "Missing instruction"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "error": "Missing instruction",
                "similarity": 0.0
            })
            continue
        
        try:
            step_name = f"Hold-out E{held_out_idx}"
            
            grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
                instructions=held_out_instructions,
                training_examples=context_examples,
                test_input_grid=held_out_example.input,
                challenge_data=challenge_data,
                is_held_out=True,
                example_order=example_order,
                working_hypothesis=working_hypothesis,
                modality_type=modality_type,
                session_id=session_id
            )
            
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            expected_grid = held_out_example.output
            similarity = get_grid_similarity(expected_grid, grid)
            
            result = {
                "held_out_idx": held_out_idx,
                "similarity": similarity,
                "uncertainty": uncertainty,
                "reasoning_content": reasoning
            }
            
            held_out_results.append(result)
            
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": expected_grid,
                "generated": grid
            }
            
        except Exception as e:
            logger.error(f"    Error in held-out validation for example {held_out_idx}: {e}")
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "error": str(e)
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "error": str(e),
                "similarity": 0.0
            })
    
    return held_out_results, grids_dict


async def run_test_cases_reduced(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    output_dir: Path,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    session_id: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run test cases (reduced context) for specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running test cases (reduced): {modality_type} ({order_name})")
    
    test_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for test_idx, test_case in enumerate(challenge_data.test):
        logger.info(f"    Processing test {test_idx} (reduced)...")
        
        test_instructions = {}
        if "general" in belief.transform_instructions:
            test_instructions["general"] = belief.transform_instructions["general"]
        
        test_key = f"T{test_idx}"
        grid_key = test_key
        if test_key in belief.transform_instructions:
            test_instructions[test_key] = belief.transform_instructions[test_key]
        else:
            logger.warning(f"Test instruction {test_key} not found, using general instructions only")
        
        try:
            step_name = f"Test T{test_idx} reduced"
            
            grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
                instructions=test_instructions,
                training_examples=challenge_data.train,
                test_input_grid=test_case.input,
                challenge_data=challenge_data,
                is_held_out=False,
                test_idx=test_idx,
                example_order=example_order,
                working_hypothesis=belief.working_hypothesis,
                modality_type=modality_type,
                include_training_examples=False,
                session_id=session_id
            )
            
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            similarity = 0.0
            expected_grid = None
            
            if hasattr(test_case, 'output') and test_case.output is not None:
                expected_grid = test_case.output
                similarity = get_grid_similarity(expected_grid, grid)
            
            result = {
                "test_idx": test_idx,
                "similarity": similarity,
                "uncertainty": uncertainty,
                "reasoning_content": reasoning,
                "has_ground_truth": hasattr(test_case, 'output') and test_case.output is not None
            }
            
            test_results.append(result)
            
            grid_data = {
                "input": test_case.input,
                "generated": grid
            }
            if expected_grid is not None:
                grid_data["expected"] = expected_grid
            grids_dict[grid_key] = grid_data
            
        except Exception as e:
            logger.error(f"    Error processing test {test_idx} (reduced): {e}")
            grids_dict[grid_key] = {
                "input": test_case.input,
                "error": str(e)
            }
            test_results.append({
                "test_idx": test_idx,
                "error": str(e),
                "similarity": 0.0
            })
    
    return test_results, grids_dict


async def run_test_cases(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    output_dir: Path,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    session_id: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run test cases for specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running test cases: {modality_type} ({order_name})")
    
    test_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for test_idx, test_case in enumerate(challenge_data.test):
        logger.info(f"    Processing test {test_idx}...")
        
        test_instructions = {}
        if "general" in belief.transform_instructions:
            test_instructions["general"] = belief.transform_instructions["general"]
        
        for key in belief.transform_instructions.keys():
            if key.startswith("E"):
                test_instructions[key] = belief.transform_instructions[key]
        
        test_key = f"T{test_idx}"
        grid_key = test_key
        if test_key in belief.transform_instructions:
            test_instructions[test_key] = belief.transform_instructions[test_key]
        else:
            logger.warning(f"Test instruction {test_key} not found, using general instructions only")
        
        try:
            step_name = f"Test T{test_idx}"
            
            grid, uncertainty, reasoning = await follow_instructions_to_generate_grid(
                instructions=test_instructions,
                training_examples=challenge_data.train,
                test_input_grid=test_case.input,
                challenge_data=challenge_data,
                is_held_out=False,
                test_idx=test_idx,
                example_order=example_order,
                working_hypothesis=belief.working_hypothesis,
                modality_type=modality_type,
                session_id=session_id
            )
            
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            similarity = 0.0
            expected_grid = None
            
            if hasattr(test_case, 'output') and test_case.output is not None:
                expected_grid = test_case.output
                similarity = get_grid_similarity(expected_grid, grid)
            
            result = {
                "test_idx": test_idx,
                "similarity": similarity,
                "uncertainty": uncertainty,
                "reasoning_content": reasoning,
                "has_ground_truth": hasattr(test_case, 'output') and test_case.output is not None
            }
            
            test_results.append(result)
            
            grid_data = {
                "input": test_case.input,
                "generated": grid
            }
            if expected_grid is not None:
                grid_data["expected"] = expected_grid
            grids_dict[grid_key] = grid_data
            
        except Exception as e:
            logger.error(f"    Error processing test {test_idx}: {e}")
            grids_dict[grid_key] = {
                "input": test_case.input,
                "error": str(e)
            }
            test_results.append({
                "test_idx": test_idx,
                "error": str(e),
                "similarity": 0.0
            })
    
    return test_results, grids_dict


async def save_results(
    output_dir: Path,
    all_results: Dict[str, Any],
    all_grids: Dict[str, Any],
    all_results_reduced: Dict[str, Any],
    all_grids_reduced: Dict[str, Any],
    hypothesis_data: Dict[str, Any],
    modality_type: str = DEFAULT_MODALITY_TYPE,
    trial_num: Optional[int] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
):
    """Save all results to files."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    if trial_num is not None:
        experiment_dir = output_dir / f"{modality_type}_{order_name}_trial{trial_num:02d}"
    else:
        experiment_dir = output_dir / f"{modality_type}_{order_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hypothesis
    with open(experiment_dir / "hypothesis.json", "w") as f:
        json.dump(hypothesis_data, f, indent=2)
    
    # Save grids
    with open(experiment_dir / "grids.json", "w") as f:
        json.dump(all_grids, f, indent=2)
    
    # Save results
    with open(experiment_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save reduced grids
    with open(experiment_dir / "grids_reduced.json", "w") as f:
        json.dump(all_grids_reduced, f, indent=2)
    
    # Save reduced results
    with open(experiment_dir / "results_reduced.json", "w") as f:
        json.dump(all_results_reduced, f, indent=2)
    
    logger.debug(f"Saved all results to {experiment_dir}")


async def run_experiment(
    challenge_data: Any,
    output_dir: Path,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    session_id: Optional[str] = None,
    trial_num: Optional[int] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
) -> Dict[str, Any]:
    """Run simplified experiment for specified modality and order."""
    
    order_name = "descending" if example_order == -1 else "ascending"
    trial_str = f" (trial {trial_num})" if trial_num is not None else ""
    logger.info(f"Running simplified experiment: {modality_type} ({order_name}){trial_str}")
    
    # Generate hypothesis
    belief, reasoning_content = await generate_hypothesis(
        challenge_data, modality_type=modality_type, model=model, temperature=temperature, 
        session_id=session_id, example_order=example_order
    )
    
    # Save hypothesis
    hypothesis_data = {
        "working_hypothesis": belief.working_hypothesis,
        "transform_instructions": belief.transform_instructions,
        "uncertainty": belief.uncertainty,
        "notebook": belief.notebook,
        "reasoning_content": reasoning_content,
        "modality_type": modality_type,
        "example_order": order_name
    }
    
    # Run all independent tasks in parallel
    logger.info(f"Running parallel tasks for {modality_type} ({order_name})")
    
    (
        (held_out_results, held_out_grids),
        (test_results, test_grids),
        (held_out_results_reduced, held_out_grids_reduced),
        (test_results_reduced, test_grids_reduced)
    ) = await asyncio.gather(
        run_held_out_validation(challenge_data, belief, output_dir, modality_type=modality_type, 
                                session_id=session_id, example_order=example_order),
        run_test_cases(challenge_data, belief, output_dir, modality_type=modality_type, 
                       session_id=session_id, example_order=example_order),
        run_held_out_validation_reduced(challenge_data, belief, output_dir, modality_type=modality_type, 
                                        session_id=session_id, example_order=example_order),
        run_test_cases_reduced(challenge_data, belief, output_dir, modality_type=modality_type, 
                              session_id=session_id, example_order=example_order)
    )
    
    logger.info(f"All parallel tasks completed for {modality_type} ({order_name})")
    
    # Prepare results dictionaries
    all_results_normal = {}
    sorted_held_out = sorted(held_out_results, key=lambda r: r.get("held_out_idx", 0))
    for result in sorted_held_out:
        grid_key = f"E{result.get('held_out_idx', 0)}"
        all_results_normal[grid_key] = result
    
    all_results = {**all_results_normal}
    sorted_test = sorted(test_results, key=lambda r: r.get("test_idx", 0))
    for result in sorted_test:
        grid_key = f"T{result.get('test_idx', 0)}"
        all_results[grid_key] = result
    
    all_grids = {**held_out_grids, **test_grids}
    
    # Prepare reduced results
    all_results_reduced_partial = {}
    sorted_held_out_reduced = sorted(held_out_results_reduced, key=lambda r: r.get("held_out_idx", 0))
    for result in sorted_held_out_reduced:
        grid_key = f"E{result.get('held_out_idx', 0)}"
        all_results_reduced_partial[grid_key] = result
    
    all_results_reduced = {**all_results_reduced_partial}
    sorted_test_reduced = sorted(test_results_reduced, key=lambda r: r.get("test_idx", 0))
    for result in sorted_test_reduced:
        grid_key = f"T{result.get('test_idx', 0)}"
        all_results_reduced[grid_key] = result
    
    all_grids_reduced = {**held_out_grids_reduced, **test_grids_reduced}
    
    # Save all results
    await save_results(
        output_dir, all_results, all_grids,
        all_results_reduced, all_grids_reduced, hypothesis_data, 
        modality_type=modality_type, trial_num=trial_num, example_order=example_order
    )
    
    return {
        "modality_type": modality_type,
        "example_order": order_name,
        "held_out_results": held_out_results,
        "test_results": test_results,
        "held_out_results_reduced": held_out_results_reduced,
        "test_results_reduced": test_results_reduced,
        "has_reasoning_content": reasoning_content is not None
    }


def calculate_total_api_calls(num_train: int, num_test: int) -> int:
    """Calculate total number of API calls needed."""
    return 1 + num_train + num_test + num_train + num_test  # hypothesis + held-out + test + held-out reduced + test reduced


async def run_full_experiment(
    challenge_id: str,
    output_base_dir: Path,
    challenges_path: Path | None = None,
    modality_type: str = DEFAULT_MODALITY_TYPE,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    rpm: Optional[int] = None,
    num_trials: int = 1,
    experiment_note: Optional[str] = None,
    example_order: Optional[int] = DEFAULT_EXAMPLE_ORDER
):
    """Run simplified experiment for a challenge, optionally multiple times."""
    
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
        
        import src.utils.follow_instructions as follow_instructions_module
        follow_instructions_module.litellm.acompletion = patched_acompletion
        
        logger.info("Patched litellm.acompletion with rate limiting")
    else:
        logger.info("Rate limiting disabled (no RPM limit)")
    
    logger.info(f"Loading challenge: {challenge_id}")
    logger.info(f"Using model: {model}, temperature: {temperature}")
    
    # Load challenge
    if challenges_path:
        from src.utils.data_loader import load_challenges_from_arc_prize_json
        solutions_path = challenges_path.parent / "arc-agi_evaluation_solutions.json"
        if not solutions_path.exists():
            solutions_path = None
            logger.warning(f"Solutions file not found, test cases will not have ground truth")
        
        challenges = load_challenges_from_arc_prize_json(
            challenges_path, challenge_ids={challenge_id}, solutions_path=solutions_path
        )
        if challenge_id not in challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        challenge_data = challenges[challenge_id]
    else:
        challenge_data = load_challenge(challenge_id)
    
    logger.info(f"Challenge loaded: {len(challenge_data.train)} train, {len(challenge_data.test)} test")
    
    # Create output directory with timestamp
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = output_base_dir / challenge_id / session_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment note if provided
    if experiment_note:
        with open(output_dir / "EXPERIMENT_NOTE.txt", "w") as f:
            f.write(experiment_note)
        logger.info(f"Experiment note saved: {experiment_note[:100]}...")
    
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    total_api_calls_per_trial = calculate_total_api_calls(num_train, num_test)
    total_api_calls = total_api_calls_per_trial * num_trials
    
    logger.info(f"Running {num_trials} trial(s)")
    logger.info(f"Total API calls needed: {total_api_calls} ({total_api_calls_per_trial} per trial)")
    
    all_summaries = []
    
    for trial in range(1, num_trials + 1):
        trial_session_id = f"{challenge_id}-simplified_experiment-{session_timestamp}-trial{trial:02d}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Trial {trial}/{num_trials}")
        logger.info(f"{'='*60}")
        logger.info(f"Langfuse session_id: {trial_session_id}")
        
        # Initialize progress tracker for this trial
        _progress_tracker = ProgressTracker(total_api_calls_per_trial)
        print(f"\n[Trial {trial}/{num_trials}] [Progress] 0/{total_api_calls_per_trial} API calls (0.0%) | Remaining: {total_api_calls_per_trial} | Initializing")
        
        # Run experiment with Langfuse session tracking
        with propagate_attributes(session_id=trial_session_id):
            try:
                result = await run_experiment(
                    challenge_data, output_dir,
                    modality_type=modality_type,
                    model=model, temperature=temperature, 
                    session_id=trial_session_id,
                    trial_num=trial,
                    example_order=example_order
                )
            except Exception as e:
                logger.error(f"Error in trial {trial}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                order_name = "descending" if example_order == -1 else "ascending"
                result = {
                    "modality_type": modality_type,
                    "example_order": order_name,
                    "trial_num": trial,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        # Save summary for this trial
        order_name = "descending" if example_order == -1 else "ascending"
        summary = {
            "session_id": trial_session_id,
            "trial_num": trial,
            "model": model,
            "temperature": temperature,
            "rpm": rpm,
            "num_train": num_train,
            "num_test": num_test,
            "modality_type": modality_type,
            "example_order": order_name,
            "result": result
        }
        
        trial_results_file = output_dir / f"results_trial{trial:02d}.json"
        with open(trial_results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        all_summaries.append(summary)
        
        print()  # New line after progress
        logger.info(f"Trial {trial} complete. Results saved to: {trial_results_file}")
    
    # Save combined summary
    order_name = "descending" if example_order == -1 else "ascending"
    combined_summary = {
        "experiment_note": experiment_note,
        "num_trials": num_trials,
        "modality_type": modality_type,
        "example_order": order_name,
        "model": model,
        "temperature": temperature,
        "rpm": rpm,
        "num_train": num_train,
        "num_test": num_test,
        "trials": all_summaries
    }
    
    with open(output_dir / "results_combined.json", "w") as f:
        json.dump(combined_summary, f, indent=2)
    
    logger.info(f"\nAll {num_trials} trial(s) complete. Combined results saved to: {output_dir}")
    
    return combined_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run simplified experiment (supports multiple modality types, ascending order only)"
    )
    parser.add_argument(
        "--challenge-id",
        type=str,
        required=True,
        help="Challenge ID to test"
    )
    parser.add_argument(
        "--modality-type",
        type=str,
        default=DEFAULT_MODALITY_TYPE,
        help=f"Modality type (default: {DEFAULT_MODALITY_TYPE}). Supported: json_only, row_col_json, row_col_json_image, image_json, row_col_image, etc."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="modality_experiment_results",
        help="Base output directory (default: modality_experiment_results)"
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
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials to run (default: 1). Each trial uses the same parameters but different random sampling."
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["ascending", "descending"],
        default="ascending",
        help="Order of training examples: 'ascending' (default) or 'descending'"
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
    
    # Convert order string to integer
    example_order = None if args.order == "ascending" else -1
    
    # Create experiment note for diversity baseline experiment
    experiment_note = None
    if args.num_trials > 1:
        experiment_note = f"""EXPERIMENT: Output Diversity Baseline - Multiple Sampling

Purpose:
This experiment measures output diversity when sampling multiple times with the same parameters
(modality, order, temperature) without changing the training example order. This serves as a
baseline to compare against order augmentation experiments.

Configuration:
- Modality: {args.modality_type}
- Order: {args.order} (fixed)
- Temperature: {args.temperature}
- Number of trials: {args.num_trials}
- Model: {args.model}

Hypothesis:
Our paper claims that varying training example orders can produce more diverse results than
simple multiple sampling. This experiment provides the baseline for that comparison.

Expected Analysis:
Compare the diversity of outputs across these {args.num_trials} trials with the diversity
observed when using different training example orders (ascending vs descending) in the
cross-order experiment.
"""
    
    # Run experiment
    asyncio.run(run_full_experiment(
        args.challenge_id,
        output_base_dir,
        challenges_file if challenges_file.exists() else None,
        modality_type=args.modality_type,
        model=args.model,
        temperature=args.temperature,
        rpm=args.rpm,
        num_trials=args.num_trials,
        experiment_note=experiment_note,
        example_order=example_order
    ))


if __name__ == "__main__":
    main()

