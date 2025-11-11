"""
Hypothesis Fast Path nodes - Constants only for public repository.

This file contains only the constants needed by experiment scripts.
The full implementation is not included in the public repository.
"""

# Model configuration
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_MODALITY = "row_col_image"  # Text + images for vision model
TEMPERATURE = 0.3

# System prompt for hypothesis generation
HYPOTHESIS_FAST_SYSTEM_PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the given training examples.

[Full prompt text omitted for brevity - see paper for details]"""

