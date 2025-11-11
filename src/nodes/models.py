"""
Shared Pydantic models for agent structured outputs.
"""

from pydantic import BaseModel, Field, create_model, model_validator
from typing import List, Literal, Optional, Dict, Any, Type


def _create_transform_instructions_fields(
    num_train: int,
    num_test: int,
    general_description: str = "WHAT to do as a reusable step-by-step algorithm, like pseudocode without example-specific constants. Use numbered steps, no placeholders. For every algorithmic step, provide pseudocode-level detail: how to iterate (scan order, processing order), how to calculate (formulas, methods), how to break ties (specify for ANY comparison), termination conditions (stopping criteria), and edge case handling (boundaries, empty sets, special cases). Specify all boundary conditions explicitly. Examples: '1: Detect objects (4-connected components of non-background cells, scan row-major). For each object, calculate area = cell count. Sort objects by area descending; tie-break by top-left position (row first, then column). 2: For each object in sorted order: determine target corner by color_to_corner_map[object.color]. Calculate new position: for top-right, new_row=1, new_col=grid_width-bbox_width+1. If target occupied, try next clockwise position...' OR '1: Build color mapping by analyzing training pairs. For each unique color C in input: target_color = (C + shift) mod palette_size. Apply to each cell row-major: output[r][c] = mapping[input[r][c]]...'",
    example_description_template: str = "HOW to execute the general algorithm for training example {i} (E{i}), adapting to this example's unique characteristics. Identify the PRIMARY distinguishing feature and make it the focus. Show step-by-step computation (don't just state results). Provide exact coordinates/counts. Flag special cases prominently at the start using '⚠️ EXCEPTION:' or '**CRITICAL:**'. Use numbered steps, no placeholders, make all values explicit. Examples: (partition puzzle) '⚠️ EXCEPTION: Off-center partition. Region 1 (A1:H9, 72 cells): Step 1: Scan row-major. Step 2: Found marker at A1 (orange, 7). Step 3: Found at B2 (teal, 8). Step 4: List=[7,8] row-major. Step 5: Two markers→spiral. Step 6: Draw layer 0 (30 cells)=7, layer 1 (26 cells)=8...' OR (movement puzzle) 'Input has 3 objects: Step 1: Detect 4-connected. Step 2: Obj1 (red, area=12, bbox C5:E8). Step 3: Obj2 (blue, area=8, bbox G2:H5). Step 4: Apply mapping: red→top-right (row=1, col=26), blue→bottom-left (row=28, col=1)...' OR (color puzzle) 'Input palette {{0,1,2,5}}. Step 1: Mapping 1→2, 2→5, 5→1, 0→0. Step 2: A1:0→0, A2:1→2, A3:2→5. Verify: 15 blues→15 reds...'",
    test_description_template: str = "HOW to execute the general algorithm for test case {i} (T{i}), with MORE detail than example instructions (test cases have no visual examples to reference). Provide exact coordinates for all spatial elements, explicit positions/counts, and any test-specific complications. Include verification steps or checksums. State any overrides of general/example instructions explicitly. Use numbered steps, no placeholders, make all values explicit. Examples: (partition puzzle) '1: Separator: teal (8) at columns L,U (inclusive), rows 8,11,15,26 (inclusive). Partition: 3 vertical columns. VC1 (A1:K20, 220 cells): Top region (A1:K8, 88 cells) has markers at B3(1), D5(2), F7(6). Sorted [1,2,6] row-major. Apply frames: layer 0 (perimeter, 30 cells)=1, layer 1 (22 cells)=2, layer 2 (14 cells)=6... Verify: 220-3=217 cells transformed.' OR (movement puzzle) 'Input: 4 objects. Obj1: red, area=15, bbox D3:G7, top-left D3. Obj2: blue, area=10, bbox A15:C19, top-left A15. Apply: red→TR (row=1, col=27), blue→BL (row=26, col=1). Check: no overlaps. Verify: 15 red cells moved, 10 blue cells moved, others unchanged.'"
) -> Dict[str, tuple]:
    """
    Helper function to create field definitions for transform_instructions.
    
    Returns a dictionary of field definitions that can be merged into a larger fields dict.
    """
    fields = {
        "general": (
            str,
            Field(description=general_description)
        ),
    }
    
    # Add E0, E1, ..., E{num_train-1} fields
    for i in range(num_train):
        fields[f"E{i}"] = (
            str,
            Field(description=example_description_template.format(i=i))
        )
    
    # Add T0, T1, ..., T{num_test-1} fields
    for i in range(num_test):
        fields[f"T{i}"] = (
            str,
            Field(description=test_description_template.format(i=i))
        )
    
    return fields


def _create_transform_instructions_updates_fields(
    num_train: int,
    num_test: int,
    general_description: str = "WHAT to do as a reusable step-by-step algorithm, like pseudocode without example-specific constants. Use numbered steps, no placeholders. For every algorithmic step, provide pseudocode-level detail: how to iterate, how to calculate, how to break ties, termination conditions, and edge case handling. Specify all boundary conditions explicitly.",
    example_description_template: str = "HOW to execute the general algorithm for training example {i} (E{i}), adapting to this example's unique characteristics. Identify the PRIMARY distinguishing feature and make it the focus. Show step-by-step computation. Provide exact inclusive boundaries. List all seed locations with coordinates. Flag special cases prominently at the start.",
    test_description_template: str = "HOW to execute the general algorithm for test case {i} (T{i}), with MORE detail than example instructions. Provide exact coordinates for all regions, explicit seed locations, and any test-specific complications. Include verification steps. State any overrides explicitly."
) -> Dict[str, tuple]:
    """
    Helper function to create optional field definitions for transform_instructions_updates.
    
    All fields are Optional[str] since these are incremental updates - only include what needs updating.
    """
    fields = {
        "general": (
            Optional[str],
            Field(description=general_description, default=None)
        ),
    }
    
    # Add E0, E1, ..., E{num_train-1} fields (all optional)
    for i in range(num_train):
        fields[f"E{i}"] = (
            Optional[str],
            Field(description=example_description_template.format(i=i), default=None)
        )
    
    # Add T0, T1, ..., T{num_test-1} fields (all optional)
    for i in range(num_test):
        fields[f"T{i}"] = (
            Optional[str],
            Field(description=test_description_template.format(i=i), default=None)
        )
    
    return fields


def _extract_transform_instructions_from_dynamic(
    dynamic_instance: BaseModel,
    num_train: int,
    num_test: int
) -> Dict[str, str]:
    """
    Helper function to extract transform_instructions dict from a dynamic model instance.
    
    Returns a dictionary with keys: 'general', 'E0', 'E1', ..., 'T0', 'T1', ...
    """
    transform_instructions = {}
    
    # Get general field
    if hasattr(dynamic_instance, "general"):
        transform_instructions["general"] = getattr(dynamic_instance, "general", "")
    
    # Get E0, E1, ..., E{num_train-1}
    for i in range(num_train):
        field_name = f"E{i}"
        if hasattr(dynamic_instance, field_name):
            transform_instructions[field_name] = getattr(dynamic_instance, field_name, "")
    
    # Get T0, T1, ..., T{num_test-1}
    for i in range(num_test):
        field_name = f"T{i}"
        if hasattr(dynamic_instance, field_name):
            transform_instructions[field_name] = getattr(dynamic_instance, field_name, "")
    
    return transform_instructions



class TransformationWithUncertainty(BaseModel):
    """Base structured output for transformation analysis with uncertainty tracking.
    
    Note: For structured output with Gemini, use create_dynamic_model() class method
    to create a model with explicit fields instead of Dict[str, str].
    """
    working_hypothesis: str = Field(
        description="WHAT + WHY. A complete list of rules you discovered. Each rule must be a specific, testable statement (the 'what'), immediately followed by the reasoning/evidence (the 'why') grounded in training examples. Use explicit evidence references like 'because E0,E2 show ...' or 'across all training examples ...'. Avoid abstractions without evidence. The result should be self-sufficient: a reader can understand the puzzle and its rationale without seeing the examples. e.g. 'In all training examples, yellow objects in the input moved to the right side in the output, indicating rightward gravity for yellow objects (evidence: E0,E1,E2).'"
    )
    transform_instructions: Dict[str, str] = Field(
        description="A dictionary with keys: 'general' = WHAT to do as a reusable step-by-step algorithm with pseudocode-level detail (how to iterate, calculate, break ties, termination conditions, edge cases); 'E{N}' = HOW to execute for each training example, adapting to that example's unique characteristics, showing step-by-step computation, providing exact coordinates/counts, flagging special cases prominently; 'T{N}' = HOW to execute for each test case with MORE detail than examples, providing exact coordinates/positions/counts, verification steps, and stated overrides. Include an 'E{n}' and 'T{n}' entry for every example/test shown. Use numbered steps, no placeholders, make all values explicit. Define common concepts (adjacency, ordering, distance, tie-breaking) when used. Examples vary by puzzle type: (partition) {'general': '1: Find separator (continuous lines). 2: For each region, detect markers (4-way adjacency), sort row-major...', 'E0': '⚠️ EXCEPTION: Off-center partition. Region 1 (A1:H9, 72 cells): Step 1: Scan row-major. Step 2: Found at A1(7)...', 'T0': 'Separator at L,U,rows 8,11,15,26. Region (A1:K8, 88 cells): markers B3(1), D5(2)...'} OR (movement) {'general': '1: Detect objects (4-connected). Sort by area desc; tie-break top-left. 2: For each: target=color_map[color], position=corner_formula...', 'E0': '3 objects: Obj1 (red, area=12, bbox C5:E8). Apply: red→TR (row=1, col=26)...', 'T0': '4 objects. Obj1 (red, area=15, D3:G7). Move to (1,27). Verify: 15 cells moved...'}"
    )
    uncertainty: str = Field(
        description="A string to list any aspects or questions you are uncertain about based on available information. Leave empty string if you have high confidence.",
        default=""
    )
    notebook: str = Field(
        description="A summary of your reasoning process, including the steps you took, the evidence you collected, and the conclusions you drew. Describe the abstract reasoning you applied, what differences you found, and how they led to your understanding. Add other relevant information to help other agents to re-examine your work.",
        default=""
    )
    
    @classmethod
    def create_dynamic_model(cls, num_train: int, num_test: int) -> Type[BaseModel]:
        """Create a dynamic model with explicit fields for Gemini structured output."""
        # Build field definitions
        fields = {
            "working_hypothesis": (
                str,
                Field(
                    description="WHAT + WHY. A complete list of rules you discovered. Each rule must be a specific, testable statement (the 'what'), immediately followed by the reasoning/evidence (the 'why') grounded in training examples. Use explicit evidence references like 'because E0,E2 show ...' or 'across all training examples ...'. Avoid abstractions without evidence. The result should be self-sufficient: a reader can understand the puzzle and its rationale without seeing the examples. Define any common concepts you use (adjacency type, ordering method, distance metric, tie-breaking rules). Examples: 'In all training examples, objects moved to corners based on their color, indicating color-to-position mapping (evidence: E0 red→top-right, E1 blue→bottom-left, E2 green→top-left). Object detection uses 4-connected components. If multiple objects have same area, process by top-left corner position row-major order.' OR 'All examples show color cycling pattern where each input color maps to the next in palette sequence (evidence: E0,E1,E2 all show 1→2, 2→5, 5→1). Background (color 0) remains unchanged.'"
                )
            ),
            "uncertainty": (
                str,
                Field(
                    description="A string to list any aspects or questions you are uncertain about based on available information. Leave empty string if you have high confidence.",
                    default=""
                )
            ),
            "notebook": (
                str,
                Field(
                    description="A summary of your reasoning process, including the steps you took, the evidence you collected, and the conclusions you drew. Describe the abstract reasoning you applied, what differences you found, and how they led to your understanding. Add other relevant information to help other agents to re-examine your work.",
                    default=""
                )
            ),
        }
        
        # Add transform_instructions fields
        fields.update(_create_transform_instructions_fields(num_train, num_test))
        
        # Create the model dynamically
        model = create_model(
            "TransformationWithUncertainty",
            **fields
        )
        
        return model
    
    @classmethod
    def from_dynamic_model(
        cls,
        dynamic_instance: BaseModel,
        num_train: int,
        num_test: int
    ) -> "TransformationWithUncertainty":
        """Convert a dynamic model instance back to TransformationWithUncertainty."""
        transform_instructions = _extract_transform_instructions_from_dynamic(
            dynamic_instance, num_train, num_test
        )
        return TransformationWithUncertainty(
            working_hypothesis=getattr(dynamic_instance, "working_hypothesis", ""),
            transform_instructions=transform_instructions,
            uncertainty=getattr(dynamic_instance, "uncertainty", ""),
            notebook=getattr(dynamic_instance, "notebook", "")
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert transform_instructions dict to flat dict with explicit keys."""
        result = {
            "working_hypothesis": self.working_hypothesis,
            "uncertainty": self.uncertainty,
            "notebook": self.notebook,
        }
        result.update(self.transform_instructions)
        return result


class GridResponse(BaseModel):
    """Structured output for grid generation (following instructions)."""
    grid: List[List[int]] = Field(
        description="The output grid which is the transform instructions applied to the test input grid. All rows must have the same length (rectangular grid)."
    )
    uncertainty: str = Field(
        description="A string describing any aspects you were uncertain about when following the instructions, including any assumptions you made to close gaps and complete the task. Empty string if you had no problems following the instructions.",
        default=""
    )
    
    @model_validator(mode='after')
    def validate_grid_consistency(self):
        """Validate that all rows in the grid have the same length (rectangular grid)."""
        grid = self.grid
        
        # Check basic structure
        if not grid:
            raise ValueError("Grid cannot be empty")
        
        if not isinstance(grid, list):
            raise ValueError(f"Grid must be a list, got {type(grid)}")
        
        if len(grid) == 0:
            raise ValueError("Grid must have at least one row")
        
        # Check that all rows are lists
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                raise ValueError(f"Row {i} must be a list, got {type(row)}")
        
        # Check that all rows have the same length
        if len(grid) > 0:
            first_row_length = len(grid[0])
            for i, row in enumerate(grid[1:], start=1):
                if len(row) != first_row_length:
                    raise ValueError(
                        f"Grid has inconsistent row lengths: row 0 has {first_row_length} columns, "
                        f"but row {i} has {len(row)} columns. All rows must have the same length."
                    )
        
        return self
