"""
Utility for following transformation instructions (AGENT_FOLLOW_INSTRUCTIONS_PROMPT pattern).

Used by Scoring Agent to test instructions on training examples.
"""

import json
import litellm
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError as PydanticValidationError
from src.nodes.models import GridResponse
from src.utils.modality_encoder import create_prompt_messages
from src import logger

# Setup litellm
litellm.drop_params = True
# Note: litellm.callbacks are configured in src/__init__.py

# Model configuration
GEMINI_MODEL = "gemini/gemini-2.5-pro"
TEMPERATURE = 0.7


AGENT_FOLLOW_INSTRUCTIONS_PROMPT = """
You are an expert puzzle solver in a competition.

You will receive:
1. A working hypothesis describing the discovered rules and patterns
2. General transformation instructions that apply to all cases
3. Specific step-by-step instructions for each training example (demonstrating how the general rules apply)
4. Specific step-by-step instructions for the test case you need to solve
5. Visual grids showing all training examples and the test input

Your task: Apply the test case instructions precisely to transform the test input grid into its output grid.

The working hypothesis and general instructions provide the information you need to solve the puzzle. The example instructions show how these rules work in practice. Use all of this context to understand the pattern, then follow the test case instructions to solve the puzzle.
""".strip()


HELD_OUT_PROMPT = """
## Important: Held-Out Validation Context

You are participating in a held-out validation experiment. This means some training examples are not visible to you—they were intentionally excluded to test whether the instructions generalize.

**Key points:**
- Sometimes, discovering certain hidden clues requires seeing all training examples together
- However, ALL such hidden information has already been discovered and documented in the working_hypothesis and instructions provided above
- The working_hypothesis contains ALL the rules, patterns, and relationships discovered from analyzing all training examples
- The instructions have been crafted to include all necessary information, even if it wasn't obvious from just the visible examples
- You do NOT need to worry about missing information—everything you need is already in the working_hypothesis and instructions
- Simply use the provided working_hypothesis and instructions to solve the puzzle, trusting that all hidden clues have been documented, but if you find that you got stuck at some step and don't have enough information to solve it, you can try to look at the training examples that are visible to you to see if you can find new hidden clues, and REMEMBER TO DOCUMENT this in the uncertainty field and provide your suggestions to improve the instructions.

**Your task:** Apply the test case instructions using the working_hypothesis and general instructions as your guide. Trust that all necessary information is already provided.
""".strip()


def _build_instructions_text(
    working_hypothesis: Optional[str],
    transform_instructions: Dict[str, str],
    is_held_out: bool = False
) -> str:
    """
    Build instructions text from working_hypothesis and transform_instructions dict.
    
    Args:
        working_hypothesis: The working hypothesis describing discovered rules
        transform_instructions: Dict with keys 'general', 'E{N}', 'T{N}'
        is_held_out: If True, add HELD_OUT_PROMPT
        
    Returns:
        Formatted instructions text
    """
    parts = []
    
    if working_hypothesis:
        parts.append(f"## Working Hypothesis\n\n{working_hypothesis}\n")
    
    # General instructions
    if "general" in transform_instructions:
        parts.append(f"## General Instructions\n\n{transform_instructions['general']}\n")
    
    # Example instructions (E0, E1, ...)
    example_keys = sorted([k for k in transform_instructions.keys() if k.startswith("E")])
    if example_keys:
        parts.append("## Example Instructions\n\nThese show how the general rules apply to specific training examples:\n")
        for key in example_keys:
            parts.append(f"### Example {key}\n{transform_instructions[key]}\n")
    
    # Test instructions (T0, T1, ...)
    test_keys = sorted([k for k in transform_instructions.keys() if k.startswith("T")])
    if test_keys:
        parts.append("## Test Case Instructions\n\nApply these instructions to solve the test case:\n")
        for key in test_keys:
            parts.append(f"### Test {key}\n{transform_instructions[key]}\n")
    
    instructions_text = "\n".join(parts)
    
    if is_held_out:
        instructions_text += f"\n\n{HELD_OUT_PROMPT}\n"
    
    return instructions_text


async def follow_instructions_to_generate_grid(
    instructions: Dict[str, str],
    training_examples: List[Any],
    test_input_grid: List[List[int]],
    challenge_data: Any,
    is_held_out: bool = False,
    example_order: int | None = None,
    test_idx: int | None = None,
    working_hypothesis: Optional[str] = None,
    modality_type: str = "row_col_image",
    use_cache: bool = True,
    include_training_examples: bool = True,
    session_id: Optional[str] = None
) -> Tuple[List[List[int]], str, Optional[str]]:
    """
    Follow transformation instructions to generate output grid for a test input.
    
    This is the core function used by Scoring Agent for leave-one-out validation.
    
    Args:
        instructions: Transformation instructions - dict with 'general', 'E{N}', 'T{N}' keys
        training_examples: List of training examples (for context)
        test_input_grid: Input grid to transform
        challenge_data: Full challenge data (for modality encoding)
        is_held_out: If True, add HELD_OUT_PROMPT (use when doing held-out validation)
        example_order: Order for training examples. None/default for ascending, -1 for descending
        test_idx: Index of the test case being processed (for caching when multiple tests exist)
        working_hypothesis: Optional working hypothesis describing discovered rules
        modality_type: Modality type for encoding (default: "row_col_image")
        use_cache: Whether to use caching for modality messages
        include_training_examples: If False, exclude training examples from modality and filter out example instructions
        
    Returns:
        Tuple of (generated output grid, uncertainty string, reasoning_content)
        - grid: list of lists of integers
        - uncertainty: string describing uncertainties/assumptions (empty string if none)
        - reasoning_content: string containing LLM reasoning content (None if not available)
    """

    # Filter instructions if not including training examples
    if not include_training_examples:
        # Only keep general and test instructions (T0, T1, ...)
        filtered_instructions = {}
        if "general" in instructions:
            filtered_instructions["general"] = instructions["general"]
        for key in instructions.keys():
            if key.startswith("T"):
                filtered_instructions[key] = instructions[key]
        transform_instructions_dict = filtered_instructions
    else:
        transform_instructions_dict = instructions
    
    instructions_text = _build_instructions_text(working_hypothesis, transform_instructions_dict, is_held_out)
    
    # Build messages - IMPORTANT: Modality messages go first (they're cached at KV cache head)
    messages = []
    
    # Create challenge object for modality encoding
    class TempChallenge:
        def __init__(self, train_examples, test_input, original_test_idx=None):
            self.train = train_examples  # Empty list if include_training_examples=False
            self.test = [type('TestInput', (), {'input': test_input})()]  # Wrap grid as test input-like object
            self.id = getattr(challenge_data, 'id', 'unknown')
            # Store original test index for correct labeling when temp_challenge has only 1 test
            self.original_test_idx = original_test_idx
    
    # If not including training examples, use empty list
    # Otherwise, use provided training_examples
    if include_training_examples:
        modality_train_examples = training_examples
    else:
        modality_train_examples = []
    
    temp_challenge = TempChallenge(modality_train_examples, test_input_grid, original_test_idx=test_idx)
    
    # Get formatted messages from modality encoder
    # Returns list of messages (single message if 1 test, two messages if multiple tests)
    # If no training examples, create_prompt_messages will only include test input
    # Pass test_idx=0 since temp_challenge has only 1 test, but original_test_idx attribute
    # will be used by _create_test_inputs_message for correct labeling
    modality_messages_list = await create_prompt_messages(
        challenge_data=temp_challenge,
        modality_type=modality_type,
        example_order=example_order if include_training_examples else None,  # Only use order if including examples
        test_idx=0,  # Use 0 since temp_challenge has only 1 test (at index 0)
        cache=use_cache
    )
    
    # Append all modality messages first (cached at KV cache head)
    messages.extend(modality_messages_list)
    
    # System message
    messages.append({
        "role": "system",
        "content": AGENT_FOLLOW_INSTRUCTIONS_PROMPT
    })
    
    # Instructions and Task messages
    task_content = f"""{instructions_text}

## Your Task

Apply the test case instructions to transform the test input grid into its output grid. Use the working hypothesis, general instructions, and example instructions as context to understand the pattern.

**Important**: If you encounter any ambiguities, unclear parts of the instructions, or need to make assumptions to complete the task, document them in the "uncertainty" field. This helps improve the instructions. If you had no problems, leave uncertainty as an empty string.

Output your response in JSON format with:
- "grid" field: The output grid as a list of lists of integers
- "uncertainty" field: A string describing any aspects you were uncertain about or assumptions you made (empty string if everything was clear)
"""
    
    messages.append({
        "role": "user",
        "content": task_content
    })
    
    # Call LLM with structured output (with retry on validation failure)
    instructions_len = len(instructions_text) if isinstance(instructions_text, str) else sum(len(str(v)) for v in transform_instructions_dict.values())
    logger.info(f"Following instructions: {instructions_len} chars, {len(training_examples)} training examples, is_held_out={is_held_out}")
    
    # Build metadata dict with session_id if provided
    metadata_dict = {
        "generation_name": "follow_instructions",
        "phase": "scoring",
        "model_type": "structured_output",
        "challenge_id": getattr(challenge_data, 'id', 'unknown'),
        "is_held_out": is_held_out
    }
    # Add session_id to metadata for LiteLLM/Langfuse integration
    if session_id:
        metadata_dict["session_id"] = session_id
    
    order_label = "Descending" if example_order == -1 else "Ascending"
    max_retries = 3
    
    # Retry loop: retry on validation failure (LLM mistake)
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=GEMINI_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                response_format=GridResponse,
                metadata=metadata_dict
            )
            
            # Parse structured output
            content = response.choices[0].message.content
            
            # Validate grid (will raise PydanticValidationError if invalid)
            grid_response = GridResponse.model_validate_json(content)
            
            # Success: grid is valid
            # Extract reasoning_content if available
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
            
            uncertainty = grid_response.uncertainty or ""
            if uncertainty:
                logger.info(f"Follow instructions uncertainty ({order_label}): {uncertainty[:100]}...")
            
            logger.info(f"Generated grid ({order_label}): shape={len(grid_response.grid)}x{len(grid_response.grid[0]) if grid_response.grid else 0}")
            
            return grid_response.grid, uncertainty, reasoning_content
            
        except PydanticValidationError as e:
            # Grid validation failed (e.g., inconsistent row lengths) - LLM made a mistake
            error_messages = []
            for error in e.errors():
                error_messages.append(f"{error['loc']}: {error['msg']}")
            
            # Try to extract grid from content to provide more context
            try:
                raw_data = json.loads(content)
                raw_grid = raw_data.get("grid", [])
                if raw_grid:
                    row_lengths = [len(row) for row in raw_grid if isinstance(row, list)]
                    error_messages.append(f"Row lengths in invalid grid: {set(row_lengths) if row_lengths else 'N/A'}")
            except:
                pass
            
            error_msg = f"Grid validation failed: {'; '.join(error_messages)}"
            
            if attempt < max_retries - 1:
                # Retry: LLM made a mistake, try again
                logger.warning(
                    f"Follow instructions validation error ({order_label}), attempt {attempt + 1}/{max_retries}: {error_msg}. Retrying..."
                )
                continue
            else:
                # All retries exhausted: return placeholder grid
                logger.error(
                    f"Follow instructions validation error ({order_label}), all {max_retries} attempts failed: {error_msg}. "
                    f"Returning placeholder grid [[0,0],[0,0]]"
                )
                placeholder_uncertainty = (
                    f"Failed to generate valid grid after {max_retries} attempts. "
                    f"Last validation error: {error_msg}. "
                    f"Returned placeholder grid [[0,0],[0,0]]."
                )
                return [[0, 0], [0, 0]], placeholder_uncertainty, None
        
        except Exception as e:
            # Other errors (API errors, network issues, etc.) - also retry
            logger.error(f"Follow instructions error ({order_label}), attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying due to error...")
                continue
            else:
                # All retries exhausted: return placeholder grid
                logger.error(
                    f"Follow instructions error ({order_label}), all {max_retries} attempts failed: {e}. "
                    f"Returning placeholder grid [[0,0],[0,0]]"
                )
                placeholder_uncertainty = (
                    f"Failed to generate grid after {max_retries} attempts due to error: {e}. "
                    f"Returned placeholder grid [[0,0],[0,0]]."
                )
                return [[0, 0], [0, 0]], placeholder_uncertainty, None
    
    # Should never reach here, but just in case
    return [[0, 0], [0, 0]], "Unexpected error: retry loop completed without returning", None


async def follow_instructions_twice(
    instructions: Dict[str, str],
    training_examples: List[Any],
    test_input_grid: List[List[int]],
    challenge_data: Any,
    is_held_out: bool = False,
    test_idx: int | None = None,
    working_hypothesis: Optional[str] = None,
    modality_type: str = "row_col_image",
    use_cache: bool = True,
    include_training_examples: bool = True,
    session_id: Optional[str] = None
) -> Tuple[List[List[int]], str, Optional[str], List[List[int]], str, Optional[str]]:
    """
    Call follow_instructions_to_generate_grid twice: once with ascending order, once with descending order.
    
    This provides two solutions as ARC competition allows submitting two solutions per challenge.
    
    Args:
        instructions: Transformation instructions - dict with 'general', 'E{N}', 'T{N}' keys
        training_examples: List of training examples (for context)
        test_input_grid: Input grid to transform
        challenge_data: Full challenge data (for modality encoding)
        is_held_out: If True, add HELD_OUT_PROMPT (use when doing held-out validation)
        test_idx: Index of the test case being processed (for caching when multiple tests exist)
        working_hypothesis: Optional working hypothesis describing discovered rules
        modality_type: Modality type for encoding (default: "row_col_image")
        use_cache: Whether to use caching for modality messages
        include_training_examples: If False, exclude training examples from modality and filter out example instructions
        
    Returns:
        Tuple of (ascending_grid, ascending_uncertainty, ascending_reasoning, descending_grid, descending_uncertainty, descending_reasoning)
    """
    # Note: When include_training_examples=False, example_order doesn't matter since there are no examples
    # But we still call twice for consistency (both will have same result)
    # Call with ascending order (default, example_order=None)
    ascending_grid, ascending_uncertainty, ascending_reasoning = await follow_instructions_to_generate_grid(
        instructions=instructions,
        training_examples=training_examples,
        test_input_grid=test_input_grid,
        challenge_data=challenge_data,
        is_held_out=is_held_out,
        example_order=None,  # Ascending order (ignored if include_training_examples=False)
        test_idx=test_idx,
        working_hypothesis=working_hypothesis,
        modality_type=modality_type,
        use_cache=use_cache,
        include_training_examples=include_training_examples,
        session_id=session_id
    )
    
    # Call with descending order (example_order=-1)
    descending_grid, descending_uncertainty, descending_reasoning = await follow_instructions_to_generate_grid(
        instructions=instructions,
        training_examples=training_examples,
        test_input_grid=test_input_grid,
        challenge_data=challenge_data,
        is_held_out=is_held_out,
        example_order=-1,  # Descending order (ignored if include_training_examples=False)
        test_idx=test_idx,
        working_hypothesis=working_hypothesis,
        modality_type=modality_type,
        use_cache=use_cache,
        include_training_examples=include_training_examples,
        session_id=session_id
    )
    
    return ascending_grid, ascending_uncertainty, ascending_reasoning, descending_grid, descending_uncertainty, descending_reasoning

