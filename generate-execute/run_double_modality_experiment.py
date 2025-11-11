"""
Modality experiment script: Test all modality types with ascending/descending orders.

For each modality type:
1. Generate hypothesis with ascending order
2. Generate hypothesis with descending order
3. Run held-out validation (leave-one-out on training examples)
4. Run follow_instructions for each test case with same modality
5. Score results and store individual scores for plotting
6. Save all outputs including reasoning_content and plots

API Call Pattern:
- Outer loop: 7 modalities × 2 orders = 14 combinations
- Per combination:
  - 1 API call for hypothesis generation
  - For each training example: 2 calls (ascending + descending order) × num_train
  - For each test case: 2 calls (ascending + descending order) × num_test
  - Same for "reduced" version (no training examples in context)
- Total API calls ≈ 14 × (1 + 2×num_train + 2×num_test + 2×num_train + 2×num_test)
  Example: 14 × (1 + 2×3 + 2×2 + 2×3 + 2×2) = 14 × 21 = 294 calls
- All calls are sequential (no parallelization) and may hit rate limits
- Use --rpm flag to enable rate limiting (recommended: 60-120 RPM)

Output structure:
- {challenge_id}/{timestamp}/
  - results.json (summary with all experiment results)
  - plots/
    - modality_comparison.png (x=modality, y=score, lines per example/test)
    - example_order_comparison.png (x=example_idx, y=score, asc vs desc)
  - {modality}_{order}/
    - hypothesis.json (LLM response + reasoning_content)
    - grids.json (all grids: E0, E1, T0, T1 as keys, each with input/expected/ascending/descending)
    - results.json (all scores: E0, E1, T0, T1 as keys, each with similarity scores and metadata)

Usage:
uv runpython test_modality_experiment.py --challenge-id <challenge_id> --output-dir <output_dir> --challenges-file <challenges_file> --model <model> --temperature <temperature> --rpm <rpm> --resume-from-dir <resume_from_dir>

Example:
# New experiment
uv run python test_modality_experiment.py --challenge-id 13e47133 --rpm 60

# Resume from previous run
uv run python test_modality_experiment.py --challenge-id 13e47133 --resume-from-dir modality_experiment_results/13e47133/20241101_1200 --rpm 60

candidate challenges: 
Medium:
13e47133: 3 Examples + 1 Test, 238 API calls
142ca369: 3 Examples + 1 Test, 238 API calls
Hard:
0934a4d8: 4 Examples + 1 Test, 294 API calls
"""

import argparse
import asyncio
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be set before importing pyplot
import matplotlib.pyplot as plt
import litellm

try:
    from langfuse import propagate_attributes
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create a no-op context manager if langfuse is not available
    from contextlib import contextmanager
    @contextmanager
    def propagate_attributes(**kwargs):
        yield

from src import logger, get_run_log_file
from src.utils.data_loader import load_challenge, DEFAULT_CHALLENGES_FILE, DEFAULT_SOLUTIONS_FILE
from src.utils.modality_encoder import create_prompt_messages
from src.utils.follow_instructions import follow_instructions_twice
# We'll patch litellm.acompletion after setting up rate limiter
from src.utils.scoring_engine import get_grid_similarity
from src.nodes.models import TransformationWithUncertainty
from src.nodes.hypothesis_fast_nodes import (
    GEMINI_MODEL,
    TEMPERATURE,
    HYPOTHESIS_FAST_SYSTEM_PROMPT
)
import sys

# Setup litellm
litellm.drop_params = True


class RateLimiter:
    """
    Rate limiter for API calls based on requests per minute (RPM).
    
    Uses a sliding window approach: tracks timestamps of recent API calls
    and waits if necessary to stay within the RPM limit.
    """
    
    def __init__(self, rpm: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rpm: Maximum requests per minute. If None, no rate limiting is applied.
        """
        self.rpm = rpm
        self.call_times: deque = deque()  # Timestamps of recent API calls
        self.lock = asyncio.Lock()  # Lock for thread-safe access
    
    async def wait_if_needed(self):
        """
        Wait if necessary to stay within RPM limit.
        
        This should be called before making an API call.
        """
        if self.rpm is None:
            return  # No rate limiting
        
        async with self.lock:
            now = time.time()
            minute_ago = now - 60.0
            
            # Remove timestamps older than 1 minute
            while self.call_times and self.call_times[0] < minute_ago:
                self.call_times.popleft()
            
            # If we're at the limit, wait until we can make another request
            if len(self.call_times) >= self.rpm:
                # Calculate how long to wait: time until oldest call is 1 minute old
                oldest_time = self.call_times[0]
                wait_time = 60.0 - (now - oldest_time) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.info(f"Rate limiter: Waiting {wait_time:.2f}s to stay within {self.rpm} RPM limit")
                    await asyncio.sleep(wait_time)
                    # Update now after waiting
                    now = time.time()
                    # Remove any timestamps that are now older than 1 minute
                    while self.call_times and self.call_times[0] < now - 60.0:
                        self.call_times.popleft()
            
            # Record this API call
            self.call_times.append(now)

# All supported modality types
MODALITY_TYPES = [
    "row_only",
    "col_only",
    "image_only",
    "row_col",
    "row_image",
    "col_image",
    "row_col_image",
]


# Global rate limiter instance (set by main function)
_rate_limiter: Optional[RateLimiter] = None

# Store original acompletion for patching
_original_acompletion = None

# Progress tracking
class ProgressTracker:
    """Track progress of API calls and experiment steps."""
    
    def __init__(self, total_api_calls: int):
        self.total_api_calls = total_api_calls
        self.completed_api_calls = 0
        self.current_step = "Initializing"
        self.lock = asyncio.Lock()
    
    async def increment(self, step_name: str = None):
        """Increment completed API calls counter."""
        async with self.lock:
            self.completed_api_calls += 1
            if step_name:
                self.current_step = step_name
    
    def get_progress(self) -> Tuple[int, int, str]:
        """Get current progress: (completed, total, current_step)."""
        return (self.completed_api_calls, self.total_api_calls, self.current_step)
    
    def print_progress(self):
        """Print current progress to console."""
        remaining = self.total_api_calls - self.completed_api_calls
        progress_pct = (self.completed_api_calls / self.total_api_calls * 100) if self.total_api_calls > 0 else 0.0
        print(f"\r[Progress] {self.completed_api_calls}/{self.total_api_calls} API calls ({progress_pct:.1f}%) | Remaining: {remaining} | {self.current_step}", end="", flush=True)

# Global progress tracker
_progress_tracker: Optional[ProgressTracker] = None

# File write lock for atomic operations (per modality+order combination)
_file_write_locks: Dict[str, asyncio.Lock] = {}


async def generate_hypothesis(
    challenge_data: Any,
    modality_type: str,
    example_order: int | None = None,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    session_id: Optional[str] = None
) -> Tuple[TransformationWithUncertainty, str | None]:
    """
    Generate hypothesis using Gemini with specified modality and order.
    
    Args:
        challenge_data: Challenge data
        modality_type: Modality type to use
        example_order: Example order (None for ascending, -1 for descending)
        model: Model to use (defaults to GEMINI_MODEL)
        temperature: Temperature setting (defaults to TEMPERATURE)
    
    Returns:
        (belief, reasoning_content)
    """
    order_name = "desc" if example_order == -1 else "asc"
    step_name = f"Hypothesis: {modality_type} ({order_name})"
    
    logger.info(f"Generating hypothesis: modality={modality_type}, order={order_name}, model={model}, temperature={temperature}")
    
    # Get number of examples for dynamic model creation
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    
    # Create dynamic model for this challenge
    DynamicModel = TransformationWithUncertainty.create_dynamic_model(num_train, num_test)
    
    # Get modality messages (without system prompt)
    modality_messages_list = await create_prompt_messages(
        challenge_data, modality_type, example_order=example_order
    )
    
    # Prepend system prompt
    messages = [
        {"role": "system", "content": HYPOTHESIS_FAST_SYSTEM_PROMPT}
    ]
    messages.extend(modality_messages_list)
    
    # Determine generation name
    order_suffix = "desc" if example_order == -1 else "asc"
    generation_name = f"modality_exp_{modality_type}_{order_suffix}"
    
    # Use litellm.acompletion (will be rate-limited if RPM is set)
    # Build metadata dict with session_id if provided
    metadata_dict = {
        "generation_name": generation_name,
        "phase": "modality_experiment",
        "model_type": "vision",
        "modality": modality_type,
        "challenge_id": getattr(challenge_data, 'id', 'unknown')
    }
    # Add session_id to metadata for LiteLLM/Langfuse integration
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
    
    # Parse structured output into dynamic model
    content = response.choices[0].message.content
    dynamic_instance = DynamicModel.model_validate_json(content)
    
    # Convert back to TransformationWithUncertainty for compatibility
    belief = TransformationWithUncertainty.from_dynamic_model(dynamic_instance, num_train, num_test)
    
    # Extract reasoning_content
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    
    # Update progress tracker after successful API call
    global _progress_tracker
    if _progress_tracker:
        await _progress_tracker.increment(step_name)
        _progress_tracker.print_progress()
    
    return belief, reasoning_content


async def run_held_out_validation_reduced(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    output_dir: Path,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run held-out validation (leave-one-out) for a modality + order combination with reduced context.
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per held-out example
        - grids_dict: Dict with keys E0, E1, ... containing grids
    """
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
        
        # Hold out example i, use others as context
        held_out_example = training_examples[held_out_idx]
        context_examples = training_examples[:held_out_idx] + training_examples[held_out_idx+1:]
        
        # Defensive check: skip if no context examples available
        if not context_examples:
            logger.warning(f"Skipping held-out validation for example {held_out_idx} (only 1 training example, no context)")
            # Store placeholder with NaN-equivalent (0.0) for plots
            grids_dict[f"E{held_out_idx}"] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "skipped": "No context examples"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "skipped": True,
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
            continue
        
        # Build instructions for held-out validation (reduced: only general + test instruction)
        held_out_instructions = {}
        # Copy general instructions only
        if "general" in transform_instructions:
            held_out_instructions["general"] = transform_instructions["general"]
        # Use held-out example's instruction as test instruction
        held_out_key = f"E{held_out_idx}"
        grid_key = held_out_key  # Use E0, E1, etc. as keys
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
            continue
        
        try:
            step_name = f"Hold-out E{held_out_idx} reduced: {modality_type} ({order_name[:3]})"
            
            # Apply transform_instructions twice with reduced context (no training examples)
            asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
                instructions=held_out_instructions,
                training_examples=context_examples,  # Passed but not used in modality
                test_input_grid=held_out_example.input,
                challenge_data=challenge_data,
                is_held_out=True,
                working_hypothesis=working_hypothesis,
                modality_type=modality_type,
                include_training_examples=False,  # Reduced mode: no training examples
                session_id=session_id
            )
            
            # Increment for both API calls (ascending + descending order) after successful completion
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            # Calculate scores for both orders
            expected_grid = held_out_example.output
            asc_similarity = get_grid_similarity(expected_grid, asc_grid)
            desc_similarity = get_grid_similarity(expected_grid, desc_grid)
            
            # Use best score (since ARC allows two trials)
            best_similarity = max(asc_similarity, desc_similarity)
            
            result = {
                "held_out_idx": held_out_idx,
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
                "best_similarity": best_similarity
            }
            
            held_out_results.append(result)
            
            # Store grids
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": expected_grid,
                "ascending": asc_grid,
                "descending": desc_grid
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
    
    return held_out_results, grids_dict


async def run_held_out_validation(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    output_dir: Path,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run held-out validation (leave-one-out) for a modality + order combination.
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per held-out example
        - grids_dict: Dict with keys E0, E1, ... containing grids
    """
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
        
        # Hold out example i, use others as context
        held_out_example = training_examples[held_out_idx]
        context_examples = training_examples[:held_out_idx] + training_examples[held_out_idx+1:]
        
        # Defensive check: skip if no context examples available
        if not context_examples:
            logger.warning(f"Skipping held-out validation for example {held_out_idx} (only 1 training example, no context)")
            # Store placeholder with NaN-equivalent (0.0) for plots
            grids_dict[f"E{held_out_idx}"] = {
                "input": held_out_example.input,
                "expected": held_out_example.output,
                "skipped": "No context examples"
            }
            held_out_results.append({
                "held_out_idx": held_out_idx,
                "skipped": True,
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
            continue
        
        # Build instructions for held-out validation:
        # - Include general + context example instructions (E0, E1, ... excluding E{held_out_idx})
        # - Use held-out example's instruction (E{held_out_idx}) as the test instruction (T0)
        held_out_instructions = {}
        # Copy general instructions
        if "general" in transform_instructions:
            held_out_instructions["general"] = transform_instructions["general"]
        # Copy context example instructions (all except held-out)
        for key in transform_instructions.keys():
            if key.startswith("E") and key != f"E{held_out_idx}":
                held_out_instructions[key] = transform_instructions[key]
        # Use held-out example's instruction as test instruction
        held_out_key = f"E{held_out_idx}"
        grid_key = held_out_key  # Use E0, E1, etc. as keys
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
            continue
        
        try:
            step_name = f"Hold-out E{held_out_idx}: {modality_type} ({order_name[:3]})"
            
            # Apply transform_instructions twice: ascending and descending order
            asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
                instructions=held_out_instructions,
                training_examples=context_examples,
                test_input_grid=held_out_example.input,
                challenge_data=challenge_data,
                is_held_out=True,
                working_hypothesis=working_hypothesis,
                modality_type=modality_type,
                session_id=session_id
            )
            
            # Increment for both API calls (ascending + descending order) after successful completion
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
            # Calculate scores for both orders
            expected_grid = held_out_example.output
            asc_similarity = get_grid_similarity(expected_grid, asc_grid)
            desc_similarity = get_grid_similarity(expected_grid, desc_grid)
            
            # Use best score (since ARC allows two trials)
            best_similarity = max(asc_similarity, desc_similarity)
            
            result = {
                "held_out_idx": held_out_idx,
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
                "best_similarity": best_similarity
            }
            
            held_out_results.append(result)
            
            # Store grids
            grids_dict[grid_key] = {
                "input": held_out_example.input,
                "expected": expected_grid,
                "ascending": asc_grid,
                "descending": desc_grid
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
    
    return held_out_results, grids_dict


async def run_test_cases_reduced(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    output_dir: Path,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run test cases for a modality + order combination with reduced context (no training examples).
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per test case
        - grids_dict: Dict with keys T0, T1, ... containing grids
    """
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running test cases (reduced): {modality_type} ({order_name})")
    
    test_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for test_idx, test_case in enumerate(challenge_data.test):
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
            step_name = f"Test T{test_idx} reduced: {modality_type} ({order_name[:3]})"
            
            # Apply transform_instructions twice with reduced context (no training examples)
            asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
                instructions=test_instructions,
                training_examples=challenge_data.train,  # Passed but not used in modality
                test_input_grid=test_case.input,
                challenge_data=challenge_data,
                is_held_out=False,
                test_idx=test_idx,
                working_hypothesis=belief.working_hypothesis,
                modality_type=modality_type,
                include_training_examples=False,  # Reduced mode: no training examples
                session_id=session_id
            )
            
            # Increment for both API calls (ascending + descending order) after successful completion
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
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
            
            test_results.append(result)
            
            # Store grids
            grid_data = {
                "input": test_case.input,
                "ascending": asc_grid,
                "descending": desc_grid
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
    
    return test_results, grids_dict


async def run_test_cases(
    challenge_data: Any,
    belief: TransformationWithUncertainty,
    modality_type: str,
    example_order: int | None,
    output_dir: Path,
    session_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run test cases for a modality + order combination.
    
    Returns:
        Tuple of (results_list, grids_dict) where:
        - results_list: List of results, one per test case
        - grids_dict: Dict with keys T0, T1, ... containing grids
    """
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"  Running test cases: {modality_type} ({order_name})")
    
    test_results = []
    grids_dict = {}
    
    global _progress_tracker
    
    for test_idx, test_case in enumerate(challenge_data.test):
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
            step_name = f"Test T{test_idx}: {modality_type} ({order_name[:3]})"
            
            # Apply transform_instructions twice: ascending and descending order
            asc_grid, asc_uncertainty, asc_reasoning, desc_grid, desc_uncertainty, desc_reasoning = await follow_instructions_twice(
                instructions=test_instructions,
                training_examples=challenge_data.train,
                test_input_grid=test_case.input,
                challenge_data=challenge_data,
                is_held_out=False,
                test_idx=test_idx,
                working_hypothesis=belief.working_hypothesis,
                modality_type=modality_type,
                session_id=session_id
            )
            
            # Increment for both API calls (ascending + descending order) after successful completion
            if _progress_tracker:
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
                await _progress_tracker.increment(step_name)
                _progress_tracker.print_progress()
            
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
            
            test_results.append(result)
            
            # Store grids
            grid_data = {
                "input": test_case.input,
                "ascending": asc_grid,
                "descending": desc_grid
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
                "ascending": {"similarity": 0.0},
                "descending": {"similarity": 0.0},
                "best_similarity": 0.0
            })
    
    return test_results, grids_dict


async def save_incremental_results(
    output_dir: Path,
    modality_type: str,
    order_name: str,
    all_results: Dict[str, Any],
    all_grids: Dict[str, Any],
    all_results_reduced: Dict[str, Any] = None,
    all_grids_reduced: Dict[str, Any] = None
):
    """Save results incrementally after each API call + post processing.
    
    Uses async lock to prevent race conditions when multiple tasks write to the same files.
    """
    lock_key = f"{modality_type}_{order_name}"
    
    # Get or create lock for this modality+order combination
    global _file_write_locks
    if lock_key not in _file_write_locks:
        _file_write_locks[lock_key] = asyncio.Lock()
    
    async with _file_write_locks[lock_key]:
        hypothesis_dir = output_dir / f"{modality_type}_{order_name}"
        hypothesis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data to merge (if files exist)
        existing_results = {}
        existing_grids = {}
        existing_results_reduced = {}
        existing_grids_reduced = {}
        
        results_file = hypothesis_dir / "results.json"
        grids_file = hypothesis_dir / "grids.json"
        results_reduced_file = hypothesis_dir / "results_reduced.json"
        grids_reduced_file = hypothesis_dir / "grids_reduced.json"
        
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    existing_results = json.load(f)
            except Exception:
                pass
        
        if grids_file.exists():
            try:
                with open(grids_file, "r") as f:
                    existing_grids = json.load(f)
            except Exception:
                pass
        
        if results_reduced_file.exists():
            try:
                with open(results_reduced_file, "r") as f:
                    existing_results_reduced = json.load(f)
            except Exception:
                pass
        
        if grids_reduced_file.exists():
            try:
                with open(grids_reduced_file, "r") as f:
                    existing_grids_reduced = json.load(f)
            except Exception:
                pass
        
        # Merge: new data overwrites existing (since we're building incrementally)
        merged_results = {**existing_results, **all_results}
        merged_grids = {**existing_grids, **all_grids}
        
        # Save grids.json
        with open(grids_file, "w") as f:
            json.dump(merged_grids, f, indent=2)
        
        # Save results.json
        with open(results_file, "w") as f:
            json.dump(merged_results, f, indent=2)
        
        # Save reduced versions if provided
        if all_results_reduced is not None:
            merged_results_reduced = {**existing_results_reduced, **all_results_reduced}
            with open(results_reduced_file, "w") as f:
                json.dump(merged_results_reduced, f, indent=2)
        
        if all_grids_reduced is not None:
            merged_grids_reduced = {**existing_grids_reduced, **all_grids_reduced}
            with open(grids_reduced_file, "w") as f:
                json.dump(merged_grids_reduced, f, indent=2)


async def run_experiment_for_modality_order(
    challenge_data: Any,
    modality_type: str,
    example_order: int | None,
    output_dir: Path,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run experiment for a single modality + order combination.
    
    After hypothesis generation, runs all independent tasks in parallel:
    - Held-out validation (normal)
    - Test cases (normal)
    - Held-out validation (reduced)
    - Test cases (reduced)
    
    Args:
        challenge_data: Challenge data
        modality_type: Modality type to use
        example_order: Example order (None for ascending, -1 for descending)
        output_dir: Output directory
        model: Model to use (defaults to GEMINI_MODEL)
        temperature: Temperature setting (defaults to TEMPERATURE)
    
    Returns:
        Dict with results including individual scores for plotting.
    """
    order_name = "descending" if example_order == -1 else "ascending"
    logger.info(f"Running experiment: {modality_type} ({order_name})")
    
    # Generate hypothesis
    belief, reasoning_content = await generate_hypothesis(
        challenge_data, modality_type, example_order, model=model, temperature=temperature,
        session_id=session_id
    )
    
    # Save hypothesis response immediately
    hypothesis_dir = output_dir / f"{modality_type}_{order_name}"
    hypothesis_dir.mkdir(parents=True, exist_ok=True)
    
    hypothesis_data = {
        "working_hypothesis": belief.working_hypothesis,
        "transform_instructions": belief.transform_instructions,
        "uncertainty": belief.uncertainty,
        "notebook": belief.notebook,
        "reasoning_content": reasoning_content,
        "modality_type": modality_type,
        "example_order": order_name
    }
    
    with open(hypothesis_dir / "hypothesis.json", "w") as f:
        json.dump(hypothesis_data, f, indent=2)
    logger.debug(f"Saved hypothesis: {hypothesis_dir / 'hypothesis.json'}")
    
    # Run all independent tasks in parallel
    logger.info(f"Running parallel tasks for {modality_type} ({order_name})")
    
    # Create tasks for all independent operations
    (
        (held_out_results, held_out_grids),
        (test_results, test_grids),
        (held_out_results_reduced, held_out_grids_reduced),
        (test_results_reduced, test_grids_reduced)
    ) = await asyncio.gather(
        run_held_out_validation(
            challenge_data, belief, modality_type, example_order, output_dir, session_id=session_id
        ),
        run_test_cases(
            challenge_data, belief, modality_type, example_order, output_dir, session_id=session_id
        ),
        run_held_out_validation_reduced(
            challenge_data, belief, modality_type, example_order, output_dir, session_id=session_id
        ),
        run_test_cases_reduced(
            challenge_data, belief, modality_type, example_order, output_dir, session_id=session_id
        )
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
    
    # Save all results (atomic operation with lock)
    await save_incremental_results(
        output_dir, modality_type, order_name,
        all_results, all_grids,
        all_results_reduced, all_grids_reduced
    )
    logger.debug(f"Saved final results (normal + reduced)")
    
    return {
        "modality_type": modality_type,
        "example_order": order_name,
        "held_out_results": held_out_results,
        "test_results": test_results,
        "held_out_results_reduced": held_out_results_reduced,
        "test_results_reduced": test_results_reduced,
        "has_reasoning_content": reasoning_content is not None
    }


def detect_completed_experiments(output_dir: Path) -> set:
    """
    Detect which modality+order combinations have been completed.
    
    Checks for existence of results.json files in each modality_order subdirectory.
    
    Returns:
        Set of (modality_type, order_name) tuples that are completed.
    """
    completed = set()
    
    for modality_type in MODALITY_TYPES:
        for example_order in [None, -1]:
            order_name = "descending" if example_order == -1 else "ascending"
            hypothesis_dir = output_dir / f"{modality_type}_{order_name}"
            results_file = hypothesis_dir / "results.json"
            
            # Check if both normal and reduced results exist
            results_reduced_file = hypothesis_dir / "results_reduced.json"
            if results_file.exists() and results_reduced_file.exists():
                completed.add((modality_type, order_name))
    
    return completed


def calculate_total_api_calls(num_train: int, num_test: int) -> int:
    """
    Calculate total number of API calls needed for the experiment.
    
    Per modality+order combination:
    - 1 call for hypothesis generation
    - num_train * 2 calls for held-out validation (asc + desc)
    - num_test * 2 calls for test cases (asc + desc)
    - Same for reduced version
    
    Total: 14 combinations * (1 + 2*num_train + 2*num_test + 2*num_train + 2*num_test)
         = 14 * (1 + 4*num_train + 4*num_test)
    """
    num_combinations = len(MODALITY_TYPES) * 2  # 7 modalities * 2 orders
    calls_per_combination = (
        1 +  # hypothesis
        2 * num_train +  # held-out validation (asc + desc)
        2 * num_test +  # test cases (asc + desc)
        2 * num_train +  # held-out validation reduced (asc + desc)
        2 * num_test  # test cases reduced (asc + desc)
    )
    return num_combinations * calls_per_combination


async def run_full_experiment(
    challenge_id: str,
    output_base_dir: Path,
    challenges_path: Path | None = None,
    model: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    rpm: Optional[int] = None,
    resume_from_dir: Optional[Path] = None
):
    """Run full experiment for a challenge.
    
    Args:
        challenge_id: Challenge ID to test
        output_base_dir: Base output directory
        challenges_path: Optional path to challenges JSON file
        model: Model to use (defaults to GEMINI_MODEL)
        temperature: Temperature setting (defaults to TEMPERATURE)
        rpm: Maximum requests per minute (None = no rate limiting)
        resume_from_dir: Optional path to previous output directory to resume from
    """
    global _rate_limiter, _progress_tracker, _original_acompletion
    
    # Set up rate limiter
    if rpm is not None:
        _rate_limiter = RateLimiter(rpm)
        logger.info(f"Rate limiting enabled: {rpm} RPM")
        
        # Patch litellm.acompletion globally to use rate limiting
        # Store original to avoid double-patching
        if _original_acompletion is None:
            _original_acompletion = litellm.acompletion
        
        async def patched_acompletion(*args, **kwargs):
            await _rate_limiter.wait_if_needed()
            return await _original_acompletion(*args, **kwargs)
        
        litellm.acompletion = patched_acompletion
        
        # Also patch in the follow_instructions module to ensure it uses the rate-limited version
        import src.utils.follow_instructions as follow_instructions_module
        follow_instructions_module.litellm.acompletion = patched_acompletion
        
        logger.info("Patched litellm.acompletion with rate limiting (global and follow_instructions module)")
    else:
        logger.info("Rate limiting disabled (no RPM limit)")
    
    logger.info(f"Loading challenge: {challenge_id}")
    logger.info(f"Using model: {model}, temperature: {temperature}")
    
    # Load challenge
    if challenges_path:
        from src.utils.data_loader import load_challenges_from_arc_prize_json
        # Determine solutions file path (same directory as challenges file, just different filename)
        solutions_path = challenges_path.parent / "arc-agi_evaluation_solutions.json"
        if not solutions_path.exists():
            solutions_path = None
            logger.warning(f"Solutions file not found at {solutions_path}, test cases will not have ground truth")
        
        challenges = load_challenges_from_arc_prize_json(
            challenges_path, challenge_ids={challenge_id}, solutions_path=solutions_path
        )
        if challenge_id not in challenges:
            raise ValueError(f"Challenge {challenge_id} not found")
        challenge_data = challenges[challenge_id]
    else:
        challenge_data = load_challenge(challenge_id)
    
    logger.info(f"Challenge loaded: {len(challenge_data.train)} train, {len(challenge_data.test)} test")
    
    # Determine output directory (resume or new)
    completed_experiments = set()
    if resume_from_dir and resume_from_dir.exists():
        # Resume from existing directory
        output_dir = resume_from_dir
        logger.info(f"Resuming from: {output_dir}")
        
        # Detect completed experiments
        completed_experiments = detect_completed_experiments(output_dir)
        logger.info(f"Found {len(completed_experiments)} completed experiments")
        
        # Load existing results.json if it exists
        existing_results_file = output_dir / "results.json"
        if existing_results_file.exists():
            try:
                with open(existing_results_file, "r") as f:
                    existing_summary = json.load(f)
                    session_id = existing_summary.get("session_id", f"{challenge_id}-modality_experiment-resumed")
            except Exception as e:
                logger.warning(f"Failed to load existing results.json: {e}")
                session_id = f"{challenge_id}-modality_experiment-resumed"
        else:
            session_id = f"{challenge_id}-modality_experiment-resumed"
    else:
        # Create new output directory with timestamp
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = output_base_dir / challenge_id / session_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        session_id = f"{challenge_id}-modality_experiment-{session_timestamp}"
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Langfuse session_id: {session_id}")
    
    # Calculate total API calls needed
    num_train = len(challenge_data.train)
    num_test = len(challenge_data.test)
    total_api_calls = calculate_total_api_calls(num_train, num_test)
    
    # Subtract completed experiments
    completed_api_calls = len(completed_experiments) * (
        1 + 2 * num_train + 2 * num_test + 2 * num_train + 2 * num_test
    )
    remaining_api_calls = total_api_calls - completed_api_calls
    
    logger.info(f"Total API calls needed: {total_api_calls}")
    logger.info(f"Completed API calls: {completed_api_calls}")
    logger.info(f"Remaining API calls: {remaining_api_calls}")
    
    # Initialize progress tracker
    _progress_tracker = ProgressTracker(remaining_api_calls)
    print(f"\n[Progress] 0/{remaining_api_calls} API calls (0.0%) | Remaining: {remaining_api_calls} | Initializing")
    
    # Run experiments for all modality + order combinations with Langfuse session tracking
    # All nested observations will automatically inherit session_id
    all_results = []
    
    # Load existing results if resuming
    if resume_from_dir and (output_dir / "results.json").exists():
        try:
            with open(output_dir / "results.json", "r") as f:
                existing_summary = json.load(f)
                all_results = existing_summary.get("results", [])
                logger.info(f"Loaded {len(all_results)} existing results")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
    
    with propagate_attributes(session_id=session_id):
        for modality_type in MODALITY_TYPES:
            for example_order in [None, -1]:  # ascending, descending
                order_name = "descending" if example_order == -1 else "ascending"
                
                # Skip if already completed
                if (modality_type, order_name) in completed_experiments:
                    logger.info(f"Skipping completed experiment: {modality_type} ({order_name})")
                    # Find existing result in all_results or create placeholder
                    existing_result = None
                    for r in all_results:
                        if r.get("modality_type") == modality_type and r.get("example_order") == order_name:
                            existing_result = r
                            break
                    if not existing_result:
                        # Load from individual results.json file
                        hypothesis_dir = output_dir / f"{modality_type}_{order_name}"
                        results_file = hypothesis_dir / "results.json"
                        results_reduced_file = hypothesis_dir / "results_reduced.json"
                        if results_file.exists() and results_reduced_file.exists():
                            try:
                                with open(results_file, "r") as f:
                                    individual_results = json.load(f)
                                with open(results_reduced_file, "r") as f:
                                    individual_results_reduced = json.load(f)
                                
                                # Reconstruct result dict
                                held_out_results = []
                                test_results = []
                                held_out_results_reduced = []
                                test_results_reduced = []
                                
                                for key, value in individual_results.items():
                                    if key.startswith("E"):
                                        held_out_results.append(value)
                                    elif key.startswith("T"):
                                        test_results.append(value)
                                
                                for key, value in individual_results_reduced.items():
                                    if key.startswith("E"):
                                        held_out_results_reduced.append(value)
                                    elif key.startswith("T"):
                                        test_results_reduced.append(value)
                                
                                existing_result = {
                                    "modality_type": modality_type,
                                    "example_order": order_name,
                                    "held_out_results": held_out_results,
                                    "test_results": test_results,
                                    "held_out_results_reduced": held_out_results_reduced,
                                    "test_results_reduced": test_results_reduced,
                                    "has_reasoning_content": True  # Assume yes if file exists
                                }
                                all_results.append(existing_result)
                            except Exception as e:
                                logger.warning(f"Failed to load existing result for {modality_type} ({order_name}): {e}")
                    continue
                
                try:
                    result = await run_experiment_for_modality_order(
                        challenge_data, modality_type, example_order, output_dir,
                        model=model, temperature=temperature, session_id=session_id
                    )
                    all_results.append(result)
                    
                    # Save summary incrementally after each experiment
                    summary = {
                        "session_id": session_id,
                        "model": model,
                        "temperature": temperature,
                        "rpm": rpm,
                        "num_train": num_train,
                        "num_test": num_test,
                        "results": all_results
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(summary, f, indent=2)
                    logger.debug(f"Saved incremental summary after {modality_type} ({order_name})")
                    
                except Exception as e:
                    logger.error(f"Error in {modality_type} ({order_name}): {e}")
                    all_results.append({
                        "modality_type": modality_type,
                        "example_order": order_name,
                        "error": str(e)
                    })
                    
                    # Save summary even on error
                    summary = {
                        "session_id": session_id,
                        "model": model,
                        "temperature": temperature,
                        "rpm": rpm,
                        "num_train": num_train,
                        "num_test": num_test,
                        "results": all_results
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(summary, f, indent=2)
    
    # Final save (redundant but ensures completeness)
    summary = {
        "session_id": session_id,
        "model": model,
        "temperature": temperature,
        "rpm": rpm,
        "num_train": num_train,
        "num_test": num_test,
        "results": all_results
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    create_plots(all_results, plots_dir, num_train, num_test)
    
    print()  # New line after progress
    logger.info(f"Experiment complete. Results saved to: {output_dir}")
    
    return summary


def create_plots(
    results: List[Dict[str, Any]], 
    plots_dir: Path,
    num_train_examples: int,
    num_test_cases: int
):
    """Create plots for modality comparison and example order analysis."""
    
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        logger.warning("No valid results to create plots")
        return
    
    # Plot 1: Modality comparison (x=modality, y=score, lines per example/test)
    create_modality_comparison_plot(valid_results, plots_dir, num_train_examples, num_test_cases)
    
    # Plot 2: Example order comparison (x=example_idx, y=score, asc vs desc)
    create_example_order_plot(valid_results, plots_dir, num_train_examples, num_test_cases)
    
    logger.info(f"Plots saved to: {plots_dir}")


def create_modality_comparison_plot(
    results: List[Dict[str, Any]],
    plots_dir: Path,
    num_train_examples: int,
    num_test_cases: int
):
    """Plot 1: x=modality type, y=score, each test/example as lines, average as thicker line. Includes normal (dots) and reduced (X) variants."""
    
    modalities = MODALITY_TYPES
    modality_positions = {mod: i for i, mod in enumerate(modalities)}
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect all scores by modality, order, and variant (normal/reduced)
    scores_by_modality_order_variant = {}
    for result in results:
        modality = result["modality_type"]
        order = result["example_order"]
        
        # Normal variant
        key_normal = (modality, order, "normal")
        held_out_scores_normal = []
        for ho_result in result.get("held_out_results", []):
            if "error" not in ho_result:
                held_out_scores_normal.append(ho_result.get("best_similarity", 0.0))
        test_scores_normal = []
        for test_result in result.get("test_results", []):
            if "error" not in test_result and test_result.get("has_ground_truth", False):
                test_scores_normal.append(test_result.get("best_similarity", 0.0))
        scores_by_modality_order_variant[key_normal] = {
            "held_out": held_out_scores_normal,
            "test": test_scores_normal
        }
        
        # Reduced variant
        key_reduced = (modality, order, "reduced")
        held_out_scores_reduced = []
        for ho_result in result.get("held_out_results_reduced", []):
            if "error" not in ho_result:
                held_out_scores_reduced.append(ho_result.get("best_similarity", 0.0))
        test_scores_reduced = []
        for test_result in result.get("test_results_reduced", []):
            if "error" not in test_result and test_result.get("has_ground_truth", False):
                test_scores_reduced.append(test_result.get("best_similarity", 0.0))
        scores_by_modality_order_variant[key_reduced] = {
            "held_out": held_out_scores_reduced,
            "test": test_scores_reduced
        }
    
    # Plot individual lines for held-out examples (normal - dots)
    for order in ["ascending", "descending"]:
        for held_out_idx in range(num_train_examples):
            scores = []
            for modality in modalities:
                key = (modality, order, "normal")
                if key in scores_by_modality_order_variant:
                    ho_scores = scores_by_modality_order_variant[key]["held_out"]
                    if held_out_idx < len(ho_scores):
                        scores.append(ho_scores[held_out_idx])
                    else:
                        scores.append(None)
                else:
                    scores.append(None)
            
            if any(s is not None for s in scores):
                color = 'lightblue' if order == "ascending" else 'lightcoral'
                ax.plot(
                    [modality_positions[m] for m in modalities],
                    scores,
                    color=color,
                    alpha=0.3,
                    linewidth=0.5,
                    marker='o',
                    markersize=3,
                    label=f"Held-out E{held_out_idx} ({order[:3]})" if held_out_idx < 3 else None
                )
    
    # Plot individual lines for held-out examples (reduced - X)
    for order in ["ascending", "descending"]:
        for held_out_idx in range(num_train_examples):
            scores = []
            for modality in modalities:
                key = (modality, order, "reduced")
                if key in scores_by_modality_order_variant:
                    ho_scores = scores_by_modality_order_variant[key]["held_out"]
                    if held_out_idx < len(ho_scores):
                        scores.append(ho_scores[held_out_idx])
                    else:
                        scores.append(None)
                else:
                    scores.append(None)
            
            if any(s is not None for s in scores):
                color = 'lightblue' if order == "ascending" else 'lightcoral'
                ax.plot(
                    [modality_positions[m] for m in modalities],
                    scores,
                    color=color,
                    alpha=0.3,
                    linewidth=0.5,
                    marker='x',
                    markersize=4,
                    label=f"Held-out E{held_out_idx} reduced ({order[:3]})" if held_out_idx < 3 else None
                )
    
    # Plot individual lines for test cases (normal - dots)
    for order in ["ascending", "descending"]:
        for test_idx in range(num_test_cases):
            scores = []
            for modality in modalities:
                key = (modality, order, "normal")
                if key in scores_by_modality_order_variant:
                    test_scores = scores_by_modality_order_variant[key]["test"]
                    if test_idx < len(test_scores):
                        scores.append(test_scores[test_idx])
                    else:
                        scores.append(None)
                else:
                    scores.append(None)
            
            if any(s is not None for s in scores):
                color = 'lightgreen' if order == "ascending" else 'lightpink'
                ax.plot(
                    [modality_positions[m] for m in modalities],
                    scores,
                    color=color,
                    alpha=0.4,
                    linewidth=0.6,
                    linestyle='--',
                    marker='o',
                    markersize=3,
                    label=f"Test T{test_idx} ({order[:3]})" if test_idx < 2 else None
                )
    
    # Plot individual lines for test cases (reduced - X)
    for order in ["ascending", "descending"]:
        for test_idx in range(num_test_cases):
            scores = []
            for modality in modalities:
                key = (modality, order, "reduced")
                if key in scores_by_modality_order_variant:
                    test_scores = scores_by_modality_order_variant[key]["test"]
                    if test_idx < len(test_scores):
                        scores.append(test_scores[test_idx])
                    else:
                        scores.append(None)
                else:
                    scores.append(None)
            
            if any(s is not None for s in scores):
                color = 'lightgreen' if order == "ascending" else 'lightpink'
                ax.plot(
                    [modality_positions[m] for m in modalities],
                    scores,
                    color=color,
                    alpha=0.4,
                    linewidth=0.6,
                    linestyle='--',
                    marker='x',
                    markersize=4,
                    label=f"Test T{test_idx} reduced ({order[:3]})" if test_idx < 2 else None
                )
    
    # Plot average lines (normal - thicker with dots)
    for order in ["ascending", "descending"]:
        avg_scores = []
        for modality in modalities:
            key = (modality, order, "normal")
            if key in scores_by_modality_order_variant:
                ho_scores = scores_by_modality_order_variant[key]["held_out"]
                test_scores = scores_by_modality_order_variant[key]["test"]
                all_scores = ho_scores + test_scores
                if all_scores:
                    avg_scores.append(sum(all_scores) / len(all_scores))
                else:
                    avg_scores.append(None)
            else:
                avg_scores.append(None)
        
        color = 'blue' if order == "ascending" else 'red'
        label = f"Average ({order})"
        ax.plot(
            [modality_positions[m] for m in modalities],
            avg_scores,
            color=color,
            linewidth=3,
            marker='o',
            markersize=8,
            label=label
        )
    
    # Plot average lines (reduced - thicker with X)
    for order in ["ascending", "descending"]:
        avg_scores = []
        for modality in modalities:
            key = (modality, order, "reduced")
            if key in scores_by_modality_order_variant:
                ho_scores = scores_by_modality_order_variant[key]["held_out"]
                test_scores = scores_by_modality_order_variant[key]["test"]
                all_scores = ho_scores + test_scores
                if all_scores:
                    avg_scores.append(sum(all_scores) / len(all_scores))
                else:
                    avg_scores.append(None)
            else:
                avg_scores.append(None)
        
        color = 'blue' if order == "ascending" else 'red'
        label = f"Average reduced ({order})"
        ax.plot(
            [modality_positions[m] for m in modalities],
            avg_scores,
            color=color,
            linewidth=3,
            marker='x',
            markersize=10,
            label=label
        )
    
    ax.set_xlabel("Modality Type", fontsize=12)
    ax.set_ylabel("Similarity Score", fontsize=12)
    ax.set_title("Modality Comparison: Normal (dots) vs Reduced (X) Context", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "modality_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved modality comparison plot: {plots_dir / 'modality_comparison.png'}")


def create_example_order_plot(
    results: List[Dict[str, Any]],
    plots_dir: Path,
    num_train_examples: int,
    num_test_cases: int
):
    """Plot 2: x=example numbers, y=scores, one line for asc and one for desc per training/test + modality. Includes normal (dots) and reduced (X) variants."""
    
    modalities = MODALITY_TYPES
    
    # Create subplots: one for held-out, one for tests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 2a: Held-out examples (normal - dots)
    for modality in modalities:
        for order in ["ascending", "descending"]:
            result = None
            for r in results:
                if r.get("modality_type") == modality and r.get("example_order") == order:
                    result = r
                    break
            
            if not result or "error" in result:
                continue
            
            scores = []
            example_indices = []
            for ho_result in result.get("held_out_results", []):
                if "error" not in ho_result:
                    idx = ho_result.get("held_out_idx")
                    if order == "ascending":
                        score = ho_result.get("ascending", {}).get("similarity", 0.0)
                    else:
                        score = ho_result.get("descending", {}).get("similarity", 0.0)
                    scores.append(score)
                    example_indices.append(idx)
            
            if scores:
                color = 'blue' if order == "ascending" else 'red'
                linestyle = '-' if order == "ascending" else '--'
                ax1.plot(
                    example_indices,
                    scores,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.6,
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    label=f"{modality} ({order[:3]})"
                )
    
    # Plot 2a: Held-out examples (reduced - X)
    for modality in modalities:
        for order in ["ascending", "descending"]:
            result = None
            for r in results:
                if r.get("modality_type") == modality and r.get("example_order") == order:
                    result = r
                    break
            
            if not result or "error" in result:
                continue
            
            scores = []
            example_indices = []
            for ho_result in result.get("held_out_results_reduced", []):
                if "error" not in ho_result:
                    idx = ho_result.get("held_out_idx")
                    if order == "ascending":
                        score = ho_result.get("ascending", {}).get("similarity", 0.0)
                    else:
                        score = ho_result.get("descending", {}).get("similarity", 0.0)
                    scores.append(score)
                    example_indices.append(idx)
            
            if scores:
                color = 'blue' if order == "ascending" else 'red'
                linestyle = '-' if order == "ascending" else '--'
                ax1.plot(
                    example_indices,
                    scores,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.6,
                    linewidth=1.5,
                    marker='x',
                    markersize=5,
                    label=f"{modality} reduced ({order[:3]})"
                )
    
    # Plot average for held-out (normal - thicker with dots)
    for order in ["ascending", "descending"]:
        avg_scores = []
        example_indices = list(range(num_train_examples))
        
        for idx in example_indices:
            scores_at_idx = []
            for modality in modalities:
                result = None
                for r in results:
                    if r.get("modality_type") == modality and r.get("example_order") == order:
                        result = r
                        break
                
                if result and "error" not in result:
                    for ho_result in result.get("held_out_results", []):
                        if ho_result.get("held_out_idx") == idx and "error" not in ho_result:
                            if order == "ascending":
                                score = ho_result.get("ascending", {}).get("similarity", 0.0)
                            else:
                                score = ho_result.get("descending", {}).get("similarity", 0.0)
                            scores_at_idx.append(score)
                            break
            
            if scores_at_idx:
                avg_scores.append(sum(scores_at_idx) / len(scores_at_idx))
            else:
                avg_scores.append(None)
        
        if any(s is not None for s in avg_scores):
            color = 'darkblue' if order == "ascending" else 'darkred'
            ax1.plot(
                example_indices,
                avg_scores,
                color=color,
                linewidth=3,
                marker='o',
                markersize=8,
                label=f"Average ({order})"
            )
    
    # Plot average for held-out (reduced - thicker with X)
    for order in ["ascending", "descending"]:
        avg_scores = []
        example_indices = list(range(num_train_examples))
        
        for idx in example_indices:
            scores_at_idx = []
            for modality in modalities:
                result = None
                for r in results:
                    if r.get("modality_type") == modality and r.get("example_order") == order:
                        result = r
                        break
                
                if result and "error" not in result:
                    for ho_result in result.get("held_out_results_reduced", []):
                        if ho_result.get("held_out_idx") == idx and "error" not in ho_result:
                            if order == "ascending":
                                score = ho_result.get("ascending", {}).get("similarity", 0.0)
                            else:
                                score = ho_result.get("descending", {}).get("similarity", 0.0)
                            scores_at_idx.append(score)
                            break
            
            if scores_at_idx:
                avg_scores.append(sum(scores_at_idx) / len(scores_at_idx))
            else:
                avg_scores.append(None)
        
        if any(s is not None for s in avg_scores):
            color = 'darkblue' if order == "ascending" else 'darkred'
            ax1.plot(
                example_indices,
                avg_scores,
                color=color,
                linewidth=3,
                marker='x',
                markersize=10,
                label=f"Average reduced ({order})"
            )
    
    ax1.set_xlabel("Held-Out Example Index", fontsize=12)
    ax1.set_ylabel("Similarity Score", fontsize=12)
    ax1.set_title("Held-Out Validation: Normal (dots) vs Reduced (X)", fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=7, ncol=2)
    
    # Plot 2b: Test cases (normal - dots)
    for modality in modalities:
        for order in ["ascending", "descending"]:
            result = None
            for r in results:
                if r.get("modality_type") == modality and r.get("example_order") == order:
                    result = r
                    break
            
            if not result or "error" in result:
                continue
            
            scores = []
            test_indices = []
            for test_result in result.get("test_results", []):
                if "error" not in test_result and test_result.get("has_ground_truth", False):
                    idx = test_result.get("test_idx")
                    if order == "ascending":
                        score = test_result.get("ascending", {}).get("similarity", 0.0)
                    else:
                        score = test_result.get("descending", {}).get("similarity", 0.0)
                    scores.append(score)
                    test_indices.append(idx)
            
            if scores:
                color = 'green' if order == "ascending" else 'orange'
                linestyle = '-' if order == "ascending" else '--'
                ax2.plot(
                    test_indices,
                    scores,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.6,
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    label=f"{modality} ({order[:3]})"
                )
    
    # Plot 2b: Test cases (reduced - X)
    for modality in modalities:
        for order in ["ascending", "descending"]:
            result = None
            for r in results:
                if r.get("modality_type") == modality and r.get("example_order") == order:
                    result = r
                    break
            
            if not result or "error" in result:
                continue
            
            scores = []
            test_indices = []
            for test_result in result.get("test_results_reduced", []):
                if "error" not in test_result and test_result.get("has_ground_truth", False):
                    idx = test_result.get("test_idx")
                    if order == "ascending":
                        score = test_result.get("ascending", {}).get("similarity", 0.0)
                    else:
                        score = test_result.get("descending", {}).get("similarity", 0.0)
                    scores.append(score)
                    test_indices.append(idx)
            
            if scores:
                color = 'green' if order == "ascending" else 'orange'
                linestyle = '-' if order == "ascending" else '--'
                ax2.plot(
                    test_indices,
                    scores,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.6,
                    linewidth=1.5,
                    marker='x',
                    markersize=5,
                    label=f"{modality} reduced ({order[:3]})"
                )
    
    # Plot average for tests (normal - thicker with dots)
    for order in ["ascending", "descending"]:
        avg_scores = []
        test_indices = list(range(num_test_cases))
        
        for idx in test_indices:
            scores_at_idx = []
            for modality in modalities:
                result = None
                for r in results:
                    if r.get("modality_type") == modality and r.get("example_order") == order:
                        result = r
                        break
                
                if result and "error" not in result:
                    for test_result in result.get("test_results", []):
                        if test_result.get("test_idx") == idx and "error" not in test_result and test_result.get("has_ground_truth", False):
                            if order == "ascending":
                                score = test_result.get("ascending", {}).get("similarity", 0.0)
                            else:
                                score = test_result.get("descending", {}).get("similarity", 0.0)
                            scores_at_idx.append(score)
                            break
            
            if scores_at_idx:
                avg_scores.append(sum(scores_at_idx) / len(scores_at_idx))
            else:
                avg_scores.append(None)
        
        if any(s is not None for s in avg_scores):
            color = 'darkgreen' if order == "ascending" else 'darkorange'
            ax2.plot(
                test_indices,
                avg_scores,
                color=color,
                linewidth=3,
                marker='o',
                markersize=8,
                label=f"Average ({order})"
            )
    
    # Plot average for tests (reduced - thicker with X)
    for order in ["ascending", "descending"]:
        avg_scores = []
        test_indices = list(range(num_test_cases))
        
        for idx in test_indices:
            scores_at_idx = []
            for modality in modalities:
                result = None
                for r in results:
                    if r.get("modality_type") == modality and r.get("example_order") == order:
                        result = r
                        break
                
                if result and "error" not in result:
                    for test_result in result.get("test_results_reduced", []):
                        if test_result.get("test_idx") == idx and "error" not in test_result and test_result.get("has_ground_truth", False):
                            if order == "ascending":
                                score = test_result.get("ascending", {}).get("similarity", 0.0)
                            else:
                                score = test_result.get("descending", {}).get("similarity", 0.0)
                            scores_at_idx.append(score)
                            break
            
            if scores_at_idx:
                avg_scores.append(sum(scores_at_idx) / len(scores_at_idx))
            else:
                avg_scores.append(None)
        
        if any(s is not None for s in avg_scores):
            color = 'darkgreen' if order == "ascending" else 'darkorange'
            ax2.plot(
                test_indices,
                avg_scores,
                color=color,
                linewidth=3,
                marker='x',
                markersize=10,
                label=f"Average reduced ({order})"
            )
    
    ax2.set_xlabel("Test Case Index", fontsize=12)
    ax2.set_ylabel("Similarity Score", fontsize=12)
    ax2.set_title("Test Cases: Normal (dots) vs Reduced (X)", fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "example_order_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved example order comparison plot: {plots_dir / 'example_order_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run modality experiment for a challenge"
    )
    parser.add_argument(
        "--challenge-id",
        type=str,
        required=True,
        help="Challenge ID to test"
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
        help="Maximum requests per minute (RPM) for rate limiting. If not specified, no rate limiting is applied. Recommended: 60-120 RPM depending on your API tier."
    )
    parser.add_argument(
        "--resume-from-dir",
        type=Path,
        default=None,
        help="Path to previous output directory to resume from (e.g., modality_experiment_results/challenge_id/20241101_1200). Checks for completed experiments and skips them."
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
    # Remove console handlers, keep file handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            root_logger.removeHandler(handler)
    
    # Determine challenges file
    if args.challenges_file is None:
        challenges_file = DEFAULT_CHALLENGES_FILE
    else:
        challenges_file = args.challenges_file
    
    output_base_dir = Path(args.output_dir)
    
    # Run experiment
    asyncio.run(run_full_experiment(
        args.challenge_id,
        output_base_dir,
        challenges_file if challenges_file.exists() else None,
        model=args.model,
        temperature=args.temperature,
        rpm=args.rpm,
        resume_from_dir=args.resume_from_dir
    ))


if __name__ == "__main__":
    main()

