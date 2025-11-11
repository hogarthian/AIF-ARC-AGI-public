"""
Modality encoding utilities for ARC-AGI puzzles.

Provides functions for:
- Visual encoding: Configurable resolution (default 16x16 px/cell) PNG with spreadsheet coordinate annotations (A1, B2, etc.)
- Text encoding: Row-wise and column-wise views with simplified diff format
- Prompt construction: System prompts and message builders

Supported modality types:
- row_only: Row-wise text only
- col_only: Column-wise text only
- image_only: Image only
- json_only: JSON format only
- row_col: Row-wise + column-wise text
- row_image: Row-wise text + images
- col_image: Column-wise text + images
- row_col_image: Row-wise + column-wise text + images
- row_col_json: Row-wise + column-wise text + JSON
- row_col_json_image: Row-wise + column-wise text + JSON + images
- image_json: Images + JSON

Image resolution:
- Default: 16x16 pixels per cell
- Configurable via resolution parameter (must be >= 14)
- Patterns scale proportionally to fit the specified resolution
"""

import base64
import io
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw


# ARC official color scheme (for image generation)
ARC_COLOR_MAP = {
    0: (0, 0, 0),        # Black for 0 (empty cells)
    1: (0, 116, 217),    # Blue for 1 #0074D9
    2: (255, 65, 54),    # Red for 2 #FF4136
    3: (46, 204, 64),    # Green for 3 #2ECC40
    4: (255, 220, 0),    # Yellow for 4 #FFDC00
    5: (170, 170, 170),  # Gray for 5 #AAAAAA
    6: (240, 18, 190),   # Magenta for 6 #F012BE
    7: (255, 133, 27),   # Orange for 7 #FF851B
    8: (127, 219, 255),  # Teal for 8 #7FDBFF
    9: (135, 12, 37),    # Brown for 9 #870C25
}

# Spreadsheet column labels (A-Z, AA-AD for max 30 columns)
SPREADSHEET_COL_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD",
]


def create_pixel_number_patterns():
    """Create pixel-perfect number patterns for 0-9 (5x7 pixels each)"""
    return {
        '0': [[1,1,1,1,1], [1,0,0,0,1], [1,0,0,1,1], [1,0,1,0,1], [1,1,0,0,1], [1,0,0,0,1], [1,1,1,1,1]],  # Added diagonal slash to distinguish from O
        '1': [[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [1,1,1,1,1]],
        '2': [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]],
        '3': [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,1]],
        '4': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1]],
        '5': [[1,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,1]],
        '6': [[1,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1]],
        '7': [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0]],
        '8': [[1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1]],
        '9': [[1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,1]]
    }


def create_pixel_letter_patterns():
    """Create pixel-perfect uppercase letter patterns A-Z (5x7 pixels each)"""
    return {
        'A': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
        'B': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0]],
        'C': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,1], [0,1,1,1,0]],
        'D': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0]],
        'E': [[1,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]],
        'F': [[1,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0]],
        'G': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,0], [1,0,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'H': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
        'I': [[1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [1,1,1,1,1]],
        'J': [[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'K': [[1,0,0,0,1], [1,0,0,1,0], [1,0,1,0,0], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0], [1,0,0,0,1]],
        'L': [[1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]],
        'M': [[1,0,0,0,1], [1,1,0,1,1], [1,0,1,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
        'N': [[1,0,0,0,1], [1,1,0,0,1], [1,0,1,0,1], [1,0,0,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
        'O': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'P': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0]],
        'Q': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,1,0,1], [1,0,0,1,0], [0,1,1,0,1]],
        'R': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0], [1,0,1,0,0], [1,0,0,1,0], [1,0,0,0,1]],
        'S': [[0,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [0,1,1,1,0], [0,0,0,0,1], [0,0,0,0,1], [1,1,1,1,0]],
        'T': [[1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]],
        'U': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'V': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0]],
        'W': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,1,0,1], [1,1,0,1,1], [1,0,0,0,1]],
        'X': [[1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1], [1,0,0,0,1]],
        'Y': [[1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]],
        'Z': [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0], [1,1,1,1,1]]
    }


def draw_pixel_number(draw, number: int, x: int, y: int, color=(255, 255, 255), scale: float = 1.0, spacing: int = 1, skip_leading_zero: bool = False):
    """Draw a number using pixel patterns (handles 0-99)
    
    Args:
        draw: ImageDraw object
        number: Number to draw (0-99)
        x, y: Top-left position
        color: RGB color tuple
        scale: Scaling factor for pattern size (default 1.0)
        spacing: Spacing between digits in pixels (default 1)
        skip_leading_zero: If True and number < 10, draw only the single digit (no leading zero)
    """
    if number < 0 or number > 99:
        return
    
    patterns = create_pixel_number_patterns()
    pattern_width = int(5 * scale)
    pattern_height = int(7 * scale)
    
    # Determine which digits to draw
    if skip_leading_zero and number < 10:
        # Single digit: draw only the digit itself
        digits = [number]
    else:
        # Two digits: use leading zero padding
        display_str = f"{number:02d}"
        digits = [int(display_str[0]), int(display_str[1])]
    
    for digit_idx, digit_num in enumerate(digits):
        if 0 <= digit_num <= 9:
            pattern = patterns[str(digit_num)]
            digit_x = x + digit_idx * (pattern_width + spacing)
            
            # Scale the pattern
            for row_idx in range(7):
                for col_idx in range(5):
                    if pattern[row_idx][col_idx]:
                        # Calculate scaled pixel positions
                        scaled_col_start = int(col_idx * scale)
                        scaled_col_end = max(scaled_col_start + 1, int((col_idx + 1) * scale))
                        scaled_row_start = int(row_idx * scale)
                        scaled_row_end = max(scaled_row_start + 1, int((row_idx + 1) * scale))
                        
                        # Draw scaled pixel (may be multiple pixels if scale > 1)
                        for px in range(scaled_col_start, scaled_col_end):
                            for py in range(scaled_row_start, scaled_row_end):
                                pixel_x = digit_x + px
                                pixel_y = y + py
                                draw.rectangle([pixel_x, pixel_y, pixel_x, pixel_y], fill=color)


def draw_pixel_letter(draw, letter: str, x: int, y: int, color=(255, 255, 255), scale: float = 1.0):
    """Draw a single uppercase letter using pixel patterns (A-Z)
    
    Args:
        draw: ImageDraw object
        letter: Single uppercase letter (A-Z)
        x, y: Top-left position
        color: RGB color tuple
        scale: Scaling factor for pattern size (default 1.0)
    """
    if len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return
    
    patterns = create_pixel_letter_patterns()
    pattern = patterns[letter]
    
    # Scale the pattern
    for row_idx in range(7):
        for col_idx in range(5):
            if pattern[row_idx][col_idx]:
                # Calculate scaled pixel positions
                scaled_col_start = int(col_idx * scale)
                scaled_col_end = max(scaled_col_start + 1, int((col_idx + 1) * scale))
                scaled_row_start = int(row_idx * scale)
                scaled_row_end = max(scaled_row_start + 1, int((row_idx + 1) * scale))
                
                # Draw scaled pixel (may be multiple pixels if scale > 1)
                for px in range(scaled_col_start, scaled_col_end):
                    for py in range(scaled_row_start, scaled_row_end):
                        pixel_x = x + px
                        pixel_y = y + py
                        draw.rectangle([pixel_x, pixel_y, pixel_x, pixel_y], fill=color)


def draw_pixel_label(draw, label: str, x: int, y: int, color=(255, 255, 255), scale: float = 1.0, spacing: int = 1):
    """Draw a spreadsheet label (e.g., "A1", "AA2", "B30") using pixel patterns
    
    Args:
        draw: ImageDraw object
        label: Spreadsheet label string
        x, y: Top-left position
        color: RGB color tuple
        scale: Scaling factor for pattern size (default 1.0)
        spacing: Spacing between letters/digits in pixels (default 1)
    """
    if not label:
        return
    
    current_x = x
    letter_width = int(5 * scale)
    
    # Handle two-letter column labels (AA, AB, AC, AD)
    if len(label) >= 2 and label[:2] in ["AA", "AB", "AC", "AD"]:
        # Draw first letter (A)
        draw_pixel_letter(draw, "A", current_x, y, color, scale)
        current_x += letter_width + spacing
        # Draw second letter (A, B, C, or D)
        second_letter = label[1]
        if second_letter in "ABCD":
            draw_pixel_letter(draw, second_letter, current_x, y, color, scale)
            current_x += letter_width + spacing
        # Draw number part if present
        if len(label) > 2:
            try:
                number = int(label[2:])
                draw_pixel_number(draw, number, current_x, y, color, scale)
            except ValueError:
                pass
    else:
        # Single letter column label
        if label[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            draw_pixel_letter(draw, label[0], current_x, y, color, scale)
            current_x += letter_width + spacing
            # Draw number part if present
            if len(label) > 1:
                try:
                    number = int(label[1:])
                    draw_pixel_number(draw, number, current_x, y, color, scale)
                except ValueError:
                    pass


def create_16x16_image_with_coordinates(grid: List[List[int]], resolution: int = 16) -> Image.Image:
    """
    Create image with spreadsheet coordinate labels (A1, B2, etc.).
    Spreadsheet notation shown in non-empty cells:
    - Column letters (A-Z, AA-AD) on top half
    - Row numbers (1-30) on bottom half
    This vertical layout ensures labels like "AA30" fit within the cell.
    
    Args:
        grid: 2D grid of integers (ARC color values)
        resolution: Pixel size per cell (default 16). Must be >= 14.
    
    Returns:
        PIL Image with grid cells and coordinate labels
    """
    if not grid:
        return Image.new('RGB', (resolution, resolution), (255, 255, 255))
    
    assert resolution >= 14, f"Resolution must be >= 14 (got {resolution}). Pattern labels require at least 14 pixels per cell."
    
    rows, cols = len(grid), len(grid[0])
    assert cols <= 30, f"Grid too wide: {cols} columns (max 30)"
    assert rows <= 30, f"Grid too tall: {rows} rows (max 30)"
    
    patch_size = resolution
    
    # For resolution < 16, keep patterns at full size (no scaling)
    # For resolution >= 16, scale patterns proportionally
    if resolution < 16:
        scale = 1.0  # No scaling - keep patterns at full 5x7 size
        spacing = 1  # Keep horizontal spacing between characters
    else:
        scale = resolution / 16.0  # Scale factor for patterns
        spacing = max(1, int(1 * scale))  # Scaled spacing
    
    img_width = cols * patch_size
    img_height = rows * patch_size
    
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Calculate scaled pattern dimensions
    pattern_width = int(5 * scale)
    pattern_height = int(7 * scale)
    
    # Draw each cell
    for i in range(rows):
        for j in range(cols):
            value = grid[i][j]
            cell_color = ARC_COLOR_MAP.get(value, (200, 200, 200))
            
            x1 = j * patch_size
            y1 = i * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            
            # Draw cell background
            draw.rectangle([x1, y1, x2, y2], fill=cell_color)
            
            # Add spreadsheet coordinate label if non-empty
            if value != 0:
                # Get column label and row number separately
                col_label = SPREADSHEET_COL_LABELS[j]
                row_num = i + 1  # 1-indexed for spreadsheet notation
                
                # Calculate widths for centering (scaled)
                # Column label width: single letter = pattern_width, two letters = pattern_width + spacing + pattern_width
                col_label_width = pattern_width if len(col_label) == 1 else (pattern_width + spacing + pattern_width)
                
                # Row number width calculation
                # For resolution >= 15: single digits don't use leading zero, so width = pattern_width
                # For resolution < 15: single digits use leading zero, so width = pattern_width + spacing + pattern_width
                if resolution >= 15 and row_num < 10:
                    row_num_width = pattern_width  # Single digit without leading zero
                else:
                    row_num_width = pattern_width if row_num < 10 else (pattern_width + spacing + pattern_width)
                
                # Calculate vertical positioning based on resolution
                if resolution == 14:
                    # Resolution 14: Move letters to touch top, no vertical spacing
                    col_offset_y = 0  # Top of cell
                    row_offset_y = pattern_height  # Immediately after column letters (7 pixels)
                elif resolution == 15:
                    # Resolution 15: Move letters to top, leave 1 row space between top and bottom
                    col_offset_y = 0  # Top of cell
                    row_offset_y = pattern_height + 1  # After column letters + 1 pixel gap (8 pixels)
                else:
                    # Resolution >= 16: Center vertically in each half
                    half_size = patch_size // 2
                    col_offset_y = max(1, (half_size - pattern_height) // 2)
                    row_offset_y = half_size + max(1, (half_size - pattern_height) // 2)
                
                # Calculate horizontal positioning based on resolution
                if resolution == 14:
                    # Resolution 14: Column letters start from left edge, numbers touch right edge
                    col_offset_x = 0  # Left edge
                    row_offset_x = patch_size - row_num_width  # Right edge (numbers touch right)
                else:
                    # Resolution >= 15: Center horizontally
                    col_offset_x = max(1, (patch_size - col_label_width) // 2)
                    row_offset_x = max(1, (patch_size - row_num_width) // 2)
                
                # Top half: column letters
                if len(col_label) == 1:
                    # Single letter
                    draw_pixel_letter(draw, col_label, x1 + col_offset_x, y1 + col_offset_y, scale=scale)
                else:
                    # Two letters (AA, AB, AC, AD)
                    draw_pixel_letter(draw, "A", x1 + col_offset_x, y1 + col_offset_y, scale=scale)
                    draw_pixel_letter(draw, col_label[1], x1 + col_offset_x + pattern_width + spacing, y1 + col_offset_y, scale=scale)
                
                # Bottom half: row number
                # For resolution >= 15, skip leading zero for single digits
                skip_leading_zero = (resolution >= 15 and row_num < 10)
                draw_pixel_number(draw, row_num, x1 + row_offset_x, y1 + row_offset_y, scale=scale, spacing=spacing, skip_leading_zero=skip_leading_zero)
    
    return img


def create_image_without_coordinates(grid: List[List[int]], resolution: int = 16) -> Image.Image:
    """
    Create image WITHOUT spreadsheet coordinate labels - just the colored grid cells.
    
    Args:
        grid: 2D grid of integers (ARC color values)
        resolution: Pixel size per cell (default 16). Must be >= 1.
    
    Returns:
        PIL Image with grid cells only (no coordinate labels)
    """
    if not grid:
        return Image.new('RGB', (resolution, resolution), (255, 255, 255))
    
    assert resolution >= 1, f"Resolution must be >= 1 (got {resolution})"
    
    rows, cols = len(grid), len(grid[0])
    
    patch_size = resolution
    img_width = cols * patch_size
    img_height = rows * patch_size
    
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw each cell - just the color, no labels
    for i in range(rows):
        for j in range(cols):
            value = grid[i][j]
            cell_color = ARC_COLOR_MAP.get(value, (200, 200, 200))
            
            x1 = j * patch_size
            y1 = i * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            
            # Draw cell background only
            draw.rectangle([x1, y1, x2, y2], fill=cell_color)
    
    return img


def format_grid_row_wise(grid: List[List[int]], prefix: str) -> str:
    """
    Format grid as row-wise text: E0i1: 000100300\n\nE0i2: 030030\n\n...
    No brackets, commas, or spaces between numbers.
    Rows are separated by double newlines.
    """
    if not grid:
        return ""
    row_lines = []
    for i, row in enumerate(grid):
        # Convert row to string without brackets/commas/spaces
        row_str = "".join(str(val) for val in row)
        row_lines.append(f"{prefix}{i+1}: {row_str}")
    return "\n\n".join(row_lines)


def format_grid_column_wise(grid: List[List[int]], prefix: str) -> str:
    """
    Format grid as column-wise text: E0oA: 000300\n\nE0oB: 030000\n\n...
    Uses spreadsheet column labels (A-Z, AA-AD).
    No brackets, commas, or spaces between numbers.
    Columns are separated by double newlines.
    """
    if not grid:
        return ""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    assert cols <= 30, f"Grid too wide: {cols} columns (max 30)"
    col_lines = []
    for j in range(cols):
        col = [grid[i][j] for i in range(rows)]
        # Convert column to string without brackets/commas/spaces
        col_str = "".join(str(val) for val in col)
        col_label = SPREADSHEET_COL_LABELS[j]
        col_lines.append(f"{prefix}{col_label}: {col_str}")
    return "\n\n".join(col_lines)


def get_spreadsheet_notation_str(row: int, col: int) -> str:
    """Get spreadsheet notation for cell at (row, col) - 0-indexed"""
    assert col < 30, f"Column {col} exceeds max 30"
    return f"{SPREADSHEET_COL_LABELS[col]}{row+1}"


def get_spreadsheet_range_str(rows: int, cols: int) -> str:
    """Get spreadsheet range notation for a grid (e.g., 'A1:E3' for 5 cols x 3 rows)"""
    if rows == 0 or cols == 0:
        return ""
    assert cols <= 30, f"Column {cols} exceeds max 30"
    top_left = "A1"
    bottom_right_col = SPREADSHEET_COL_LABELS[cols - 1]
    bottom_right_row = rows
    return f"{top_left}:{bottom_right_col}{bottom_right_row}"


def get_spreadsheet_notation_with_runs(locations: List[Tuple[int, int]]) -> str:
    """
    Get spreadsheet notation for a list of locations, with run compression.
    Format: "A1 B2 C3" or "A1 ... A5" for runs > 3 consecutive cells.
    Checks for consecutive cells in both rows (same row, consecutive columns) 
    and columns (same column, consecutive rows).
    """
    if not locations:
        return ""
    
    # Convert to set for O(1) lookup
    locs_set = set(locations)
    sorted_locs = sorted(locations, key=lambda x: (x[0], x[1]))
    processed = set()
    
    result_parts = []
    
    for r, c in sorted_locs:
        if (r, c) in processed:
            continue
        
        # Check for consecutive cells in the same row (horizontal run)
        # Cells must be consecutive: same row, consecutive columns starting from current column
        count_in_row = 0
        for col_offset in range(30):  # Max 30 columns in ARC
            check_col = c + col_offset
            if (r, check_col) in locs_set and (r, check_col) not in processed:
                count_in_row += 1
            else:
                break
        
        # Check for consecutive cells in the same column (vertical run)
        # Cells must be consecutive: same column, consecutive rows starting from current row
        count_in_col = 0
        for row_offset in range(30):  # Max 30 rows in ARC
            check_row = r + row_offset
            if (check_row, c) in locs_set and (check_row, c) not in processed:
                count_in_col += 1
            else:
                break
        
        # Use whichever run is longer, prefer row runs if equal
        if count_in_row >= count_in_col and count_in_row > 3:
            # Use row range notation
            start = get_spreadsheet_notation_str(r, c)
            c_end = c + count_in_row - 1
            end = get_spreadsheet_notation_str(r, c_end)
            result_parts.append(f"{start} ... {end}")
            # Mark all cells in row run as processed
            for col_offset in range(count_in_row):
                processed.add((r, c + col_offset))
        elif count_in_col > count_in_row and count_in_col > 3:
            # Use column range notation
            start = get_spreadsheet_notation_str(r, c)
            r_end = r + count_in_col - 1
            end = get_spreadsheet_notation_str(r_end, c)
            result_parts.append(f"{start} ... {end}")
            # Mark all cells in column run as processed
            for row_offset in range(count_in_col):
                processed.add((r + row_offset, c))
        else:
            # Individual cell (no run > 3)
            result_parts.append(get_spreadsheet_notation_str(r, c))
            processed.add((r, c))
    
    return " ".join(result_parts)


def format_grid_diff_simplified(input_grid: List[List[int]], output_grid: List[List[int]], example_prefix: str) -> Tuple[str, str]:
    """
    Format simplified diff for same-size grids.
    Returns: (diff_text, spreadsheet_notation_list)
    
    Diff format: E01: 00000|F1: 3->2|000300\n\n...
    - Unchanged cells shown as values: 00000
    - Changed cells shown as |F1: 3->2| where F1 is spreadsheet notation
    - Rows separated by double newlines
    
    Spreadsheet notation list: "Blue (1) to Red (2): A1 B2 C3"
    """
    if not input_grid or not output_grid:
        return "", ""
    
    rows = len(input_grid)
    cols = len(input_grid[0]) if rows > 0 else 0
    
    row_lines = []
    differences_by_color_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    
    for y in range(rows):
        row_parts = []
        for x in range(cols):
            input_val = input_grid[y][x]
            output_val = output_grid[y][x]
            
            if input_val == output_val:
                # No change: just show the value
                row_parts.append(str(input_val))
            else:
                # Changed: show |COL_ROW: input->output|
                spreadsheet_label = get_spreadsheet_notation_str(y, x)
                row_parts.append(f"|{spreadsheet_label}: {input_val}->{output_val}|")
                differences_by_color_pairs[(input_val, output_val)].append((y, x))
        
        row_str = "".join(row_parts)
        row_lines.append(f"{example_prefix}{y+1}: {row_str}")
    
    diff_text = "\n\n".join(row_lines)
    
    # Build spreadsheet notation list by color pairs
    notation_lines = []
    for (color_input, color_output), locs in sorted(differences_by_color_pairs.items(), key=lambda x: x[0]):
        color_names = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
            5: "gray", 6: "magenta", 7: "orange", 8: "teal", 9: "maroon"
        }
        input_name = color_names.get(color_input, str(color_input))
        output_name = color_names.get(color_output, str(color_output))
        notation_str = get_spreadsheet_notation_with_runs(locs)
        notation_lines.append(f"{input_name.capitalize()} ({color_input}) to {output_name.capitalize()} ({color_output}): {notation_str}")
    
    notation_list = "\n".join(notation_lines)
    
    return diff_text, notation_list


def get_text_guide_row_col_format() -> str:
    """Get text guide for row-wise and column-wise format."""
    return '''Text format (row-wise and column-wise):
- Row-wise format: E{N}i{R}: values (e.g., E0i1: 000100300)
  - E{N} means Example N (e.g., E0, E1, E2, etc.)
  - 'i' indicates input, 'o' indicates output
  - {R} is the row number (1-indexed)
  - Values are concatenated without spaces, brackets, or commas
  
- Column-wise format: E{N}o{C}: values (e.g., E0oA: 000300)
  - E{N} means Example N
  - 'i' indicates input, 'o' indicates output
  - {C} is the column label using spreadsheet notation (A-Z, AA-AD)
  - Values are concatenated without spaces, brackets, or commas
  
- Rows/columns are separated by double newlines (\n\n)
- Test inputs use prefix T{N}i (e.g., T0i1, T0i2, etc. for rows, T0iA, T0iB, etc. for columns)
'''


def get_text_guide_diff_format() -> str:
    """Get text guide for simplified diff format (same-size grids)."""
    return '''
- Row-wise diff format: E{N}{R}: values|SPREADSHEET: input->output|values
  - Example: E01: 00000|F1: 3->2|000300
  - Unchanged cells shown as plain values: 00000
  - Changed cells shown as |SPREADSHEET: input->output| where SPREADSHEET is cell notation (e.g., F1, A2)
  
- Spreadsheet notation list: Color name (value) to Color name (value): locations
  - Example: Blue (1) to Red (2): A1 B2 C3
  - Groups changes by color pairs
  - Uses spreadsheet notation (A1, B2, etc.) for locations
  - Consecutive cells (>3) shown as ranges: A1 ... A5 (for row runs) or A1 ... D1 (for column runs)
  - Checks for consecutive cells in both rows (same row, consecutive columns) and columns (same column, consecutive rows)
  - Uses whichever run is longer, preferring row runs if equal
  
- Rows are separated by double newlines (\n\n)
- Spreadsheet notation uses A-Z for columns 0-25, AA-AD for columns 26-29
- Row numbers are 1-indexed (row 0 is row 1 in spreadsheet notation)
'''


async def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 for API transmission"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _has_row_wise(modality_type: str) -> bool:
    """Check if modality includes row-wise text."""
    return modality_type in ["row_only", "row_col", "row_image", "row_col_image"]


def _has_col_wise(modality_type: str) -> bool:
    """Check if modality includes column-wise text."""
    return modality_type in ["col_only", "row_col", "col_image", "row_col_image"]


def _has_images(modality_type: str) -> bool:
    """Check if modality includes images."""
    return modality_type in ["image_only", "row_image", "col_image", "row_col_image", "row_col_json_image", "image_json"]


def _has_json(modality_type: str) -> bool:
    """Check if modality includes JSON format."""
    return modality_type in ["json_only", "row_col_json", "row_col_json_image", "image_json"]


async def _create_training_examples_message(
    challenge_data: Any,
    modality_type: str,
    example_order: List[int] | int | None = None,
    resolution: int = 16
) -> Dict[str, Any]:
    """
    Helper function to create training examples message content.
    
    Returns:
        Message dict with training examples only.
    """
    content_parts = []
    
    # Determine what views are available
    has_row_wise = _has_row_wise(modality_type)
    has_col_wise = _has_col_wise(modality_type)
    has_images = _has_images(modality_type)
    has_json = _has_json(modality_type)
    
    # Prepare examples (with optional reordering or filtering)
    examples = challenge_data.train
    if example_order is not None:
        if isinstance(example_order, int):
            if example_order == -1:
                # Special case: reverse order of all examples
                num_examples = len(examples)
                example_order = list(range(num_examples - 1, -1, -1))  # Descending: [n-1, n-2, ..., 0]
                examples = [examples[i] for i in example_order]
            else:
                # Single example index
                examples = [examples[example_order]]
        else:
            # List of indices
            examples = [examples[i] for i in example_order]
    
    # Process training examples and determine format consistency
    has_same_size = False
    for idx, example in enumerate(examples):
        original_idx = challenge_data.train.index(example)
        example_prefix = f"E{original_idx}"
        input_rows = len(example.input)
        input_cols = len(example.input[0]) if input_rows > 0 else 0
        output_rows = len(example.output)
        output_cols = len(example.output[0]) if output_rows > 0 else 0
        same_size = (input_rows == output_rows) and (input_cols == output_cols)
        
        if same_size:
            has_same_size = True
        
        # === GRID SIZE ANNOTATION ===
        input_range = get_spreadsheet_range_str(input_rows, input_cols)
        output_range = get_spreadsheet_range_str(output_rows, output_cols)
        grid_size_info = f"{example_prefix}: Input grid size: Columns={input_cols}, Rows={input_rows} (spreadsheet range: {input_range}), Output grid size: Columns={output_cols}, Rows={output_rows} (spreadsheet range: {output_range})"
        content_parts.append({"type": "text", "text": grid_size_info})
        
        # === ROW-WISE INPUT ===
        if has_row_wise:
            row_wise_input = format_grid_row_wise(example.input, f"{example_prefix}i")
            content_parts.append({"type": "text", "text": f"Row-wise input:\n{row_wise_input}"})
        
        # === ROW-WISE OUTPUT ===
        if has_row_wise:
            row_wise_output = format_grid_row_wise(example.output, f"{example_prefix}o")
            content_parts.append({"type": "text", "text": f"Row-wise output:\n{row_wise_output}"})
        
        # === ROW-WISE DIFF (only for same-size) ===
        if has_row_wise and same_size:
            diff_text, notation_list = format_grid_diff_simplified(example.input, example.output, example_prefix)
            content_parts.append({"type": "text", "text": f"Row-wise diff:\n{diff_text}"})
            if notation_list:
                content_parts.append({"type": "text", "text": f"Changes by color:\n{notation_list}"})
        
        # === COLUMN-WISE INPUT ===
        if has_col_wise:
            col_wise_input = format_grid_column_wise(example.input, f"{example_prefix}i")
            content_parts.append({"type": "text", "text": f"Column-wise input:\n{col_wise_input}"})
        
        # === COLUMN-WISE OUTPUT ===
        if has_col_wise:
            col_wise_output = format_grid_column_wise(example.output, f"{example_prefix}o")
            content_parts.append({"type": "text", "text": f"Column-wise output:\n{col_wise_output}"})
        
        # === JSON INPUT ===
        if has_json:
            input_json = json.dumps(example.input)
            content_parts.append({"type": "text", "text": f"{example_prefix} input (JSON list of lists format):\n{input_json}"})
        
        # === JSON OUTPUT ===
        if has_json:
            output_json = json.dumps(example.output)
            content_parts.append({"type": "text", "text": f"{example_prefix} output (JSON list of lists format):\n{output_json}"})
        
        # === IMAGES ===
        if has_images:
            content_parts.append({"type": "text", "text": f"{example_prefix} input image:"})
            img = create_16x16_image_with_coordinates(example.input, resolution=resolution)
            base64_img = await encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
            
            content_parts.append({"type": "text", "text": f"{example_prefix} output image:"})
            img = create_16x16_image_with_coordinates(example.output, resolution=resolution)
            base64_img = await encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
        
        # Add blank line between examples
        if idx < len(examples) - 1:
            content_parts.append({"type": "text", "text": "\n---\n"})
    
    content_parts.append({"type": "text", "text": "\n--End of Training Examples--\n"})
    
    # Build format instruction block
    format_instructions = []
    
    # Coordinate system
    format_instructions.append("Coordinate system: Uses spreadsheet notation (A1, B2, etc.) where columns are A-Z (0-25) and AA-AD (26-29), rows are 1-indexed. Origin (A1) is top-left.")
    
    # Color map
    format_instructions.append("ARC-AGI Color map: 0=black; 1=blue; 2=red; 3=green; 4=yellow; 5=gray; 6=magenta; 7=orange; 8=teal; 9=maroon")
    
    # Text format guide(s)
    if has_row_wise or has_col_wise:
        format_instructions.append(get_text_guide_row_col_format())
        if has_same_size and has_row_wise:
            format_instructions.append(get_text_guide_diff_format())
    
    # JSON format guide
    if has_json:
        json_guide = """JSON format: Raw JSON array of arrays (list of lists), matching the native ARC-AGI data structure.
Example: [[1,2,3,4,5],[1,0,0,4,5],[1,8,9,4,5]]
This format is compact and preserves exact structure but requires parsing nested brackets and commas."""
        format_instructions.append(json_guide)
    
    # Image guide
    if has_images:
        if has_row_wise or has_col_wise:
            img_guide = """Image view: Images show the same grid as the text representation with the same orientation.
Non-empty cells include spreadsheet coordinate annotations (e.g., A1, B2, AA1).
Use images for 2-D spatial reasoning; use text for exact values and verification."""
        else:
            img_guide = """Image view: Images depict the grid with spreadsheet coordinate annotations inside non-empty cells (e.g., A1, B2, AA1).
Use images for 2-D spatial reasoning and pattern detection."""
        format_instructions.append(img_guide)
    
    # Combine all format instructions into a single block
    format_block = "\n\n".join(format_instructions)
    
    # Insert format block at the beginning of content_parts
    content_parts.insert(0, {"type": "text", "text": format_block})
    content_parts.insert(1, {"type": "text", "text": "\n--- Below are training examples ---\n"})
    
    message = {
        "role": "user",
        "content": content_parts,
    }
    
    # Add cache_control only if caching is enabled
    # Note: cache parameter is passed through but not used here since we always want to cache training examples
    # when cache=True. The actual cache control is handled by the caller if needed.
    
    return message


async def _create_test_inputs_message(
    challenge_data: Any,
    modality_type: str,
    test_idx: int | None = None,
    resolution: int = 16
) -> Dict[str, Any]:
    """
    Helper function to create test inputs message content.
    
    Args:
        challenge_data: Challenge data with test examples
        modality_type: Modality type
        test_idx: If provided, only include this test index. Otherwise include all tests.
    
    Returns:
        Message dict with test inputs only.
    """
    content_parts = []
    
    # Determine what views are available
    has_row_wise = _has_row_wise(modality_type)
    has_col_wise = _has_col_wise(modality_type)
    has_images = _has_images(modality_type)
    has_json = _has_json(modality_type)
    
    # Filter test cases if test_idx is provided
    test_cases = challenge_data.test
    if test_idx is not None:
        test_cases = [test_cases[test_idx]]
        test_indices = [test_idx]
    else:
        test_indices = list(range(len(test_cases)))
    
    # Check if challenge_data has original_test_idx attribute (for TempChallenge with single test)
    # This handles the case where we create a temp challenge with 1 test but need correct labeling
    if hasattr(challenge_data, 'original_test_idx') and challenge_data.original_test_idx is not None:
        # Override test_indices to use the original test index for labeling
        test_indices = [challenge_data.original_test_idx]
    
    # Add test inputs
    content_parts.append({"type": "text", "text": "\n--- Below are test examples ---\n"})
    for idx, test_case in enumerate(test_cases):
        original_test_idx = test_indices[idx]
        test_prefix = f"T{original_test_idx}i"
        
        # === GRID SIZE ANNOTATION ===
        test_input_rows = len(test_case.input)
        test_input_cols = len(test_case.input[0]) if test_input_rows > 0 else 0
        test_range = get_spreadsheet_range_str(test_input_rows, test_input_cols)
        test_grid_size_info = f"Test Input {original_test_idx}: Grid size: Columns={test_input_cols}, Rows={test_input_rows} (spreadsheet range: {test_range})"
        content_parts.append({"type": "text", "text": test_grid_size_info})
        
        # === JSON REPRESENTATION ===
        if has_json:
            test_input_json = json.dumps(test_case.input)
            content_parts.append({
                "type": "text",
                "text": f"Test Input {original_test_idx} (JSON list of lists format - use this to help you generate the output grid):\n{test_input_json}"
            })
        
        if has_row_wise or has_col_wise:
            text_parts = []
            if has_row_wise:
                test_row_wise = format_grid_row_wise(test_case.input, f"{test_prefix}")
                text_parts.append(f"Row-wise:\n{test_row_wise}")
            if has_col_wise:
                test_col_wise = format_grid_column_wise(test_case.input, f"{test_prefix}")
                text_parts.append(f"Column-wise:\n{test_col_wise}")
            content_parts.append({"type": "text", "text": f"Test Input {original_test_idx}:\n" + "\n\n".join(text_parts)})
        
        if has_images:
            content_parts.append({"type": "text", "text": f"{test_prefix} input image:"})
            img = create_16x16_image_with_coordinates(test_case.input, resolution=resolution)
            base64_img = await encode_image_to_base64(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
        
        if idx < len(test_cases) - 1:
            content_parts.append({"type": "text", "text": "\n---\n"})
    
    return {
        "role": "user",
        "content": content_parts,
    }


async def create_prompt_messages(
    challenge_data: Any,
    modality_type: str,
    example_order: List[int] | int | None = None,
    test_idx: int | None = None,
    cache: bool = True,
    resolution: int = 16
) -> List[Dict[str, Any]]:
    """
    Create modality messages (user messages with examples/test inputs) without system prompt.
    
    Format for each example (same-size grids):
    - Row-wise input: E0i1: ..., E0i2: ...
    - Row-wise output: E0o1: ..., E0o2: ...
    - Row-wise diff: E01: ...|F1: 3->2|... plus spreadsheet notation list
    - Column-wise input: E0iA: ..., E0iB: ...
    - Column-wise output: E0oA: ..., E0oB: ...
    - Images: input and output
    
    Format for different-size grids: Same but without diff part.
    
    Caching strategy:
    - If test list len == 1: Return single message with everything (like before)
    - If test list len > 1: Return list of two messages:
      1. Training examples message (cached if cache=True)
      2. Test inputs message (if test_idx provided, only that test; otherwise all tests)
    
    Args:
        challenge_data: Challenge data with train/test examples
        modality_type: Modality type. Supported:
            - row_only: Row-wise text only
            - col_only: Column-wise text only
            - image_only: Image only
            - json_only: JSON format only
            - row_col: Row-wise + column-wise text
            - row_image: Row-wise text + images
            - col_image: Column-wise text + images
            - row_col_image: Row-wise + column-wise text + images
            - row_col_json: Row-wise + column-wise text + JSON
            - row_col_json_image: Row-wise + column-wise text + JSON + images
            - image_json: Images + JSON
        example_order: 
            - None: Use all examples in default order
            - -1: Use all examples in reverse (descending) order
            - List[int]: Reorder examples by these indices (e.g., [0, 2, 1])
            - int: Use only this single example index (e.g., 3)
        test_idx: If provided and test list len > 1, only include this test index in test message.
            Ignored if test list len == 1.
        cache: If True, enable caching for training examples message (default True)
        resolution: Pixel size per cell for images (default 16). Must be >= 14.
    
    Returns:
        List of message dicts, each with "role", "content", and optionally "cache_control" keys. Does NOT include system prompt.
        If test list len == 1, returns single-element list with combined message.
        If test list len > 1, returns two messages: [training_examples_message, test_inputs_message].
    """
    # Check number of test cases
    num_tests = len(challenge_data.test)
    
    # If only one test, return combined message (like before)
    if num_tests == 1:
        # Create combined message with training examples and test inputs
        training_msg = await _create_training_examples_message(
            challenge_data=challenge_data,
            modality_type=modality_type,
            example_order=example_order,
            resolution=resolution
        )
        
        # Append test inputs to training message content
        test_msg = await _create_test_inputs_message(
            challenge_data=challenge_data,
            modality_type=modality_type,
            test_idx=None,  # Include all tests (which is just one)
            resolution=resolution
        )
        
        # Combine: training message content + test message content
        combined_content = training_msg["content"] + test_msg["content"]
        
        result_msg = {
            "role": "user",
            "content": combined_content,
        }
        
        # Add cache_control if caching is enabled
        if cache:
            result_msg["cache_control"] = {"type": "ephemeral"}  # Cache for 1 hour
        
        return [result_msg]
    
    # Multiple tests: return two separate messages
    # Message 1: Training examples (cached if cache=True)
    training_msg = await _create_training_examples_message(
        challenge_data=challenge_data,
        modality_type=modality_type,
        example_order=example_order,
        resolution=resolution
    )
    
    # Add cache_control to training message if caching is enabled
    if cache:
        training_msg["cache_control"] = {"type": "ephemeral"}  # Cache for 1 hour
    
    # Message 2: Test inputs (if test_idx provided, only that test; otherwise all tests)
    test_msg = await _create_test_inputs_message(
        challenge_data=challenge_data,
        modality_type=modality_type,
        test_idx=test_idx,
        resolution=resolution
    )
    
    return [training_msg, test_msg]
