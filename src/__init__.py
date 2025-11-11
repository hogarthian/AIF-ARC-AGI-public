"""ARC-AGI Public Repository - Minimal utilities for experiment scripts."""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Set up file logging for run history
_run_log_file = None
_logs_dir = Path("logs")
_logs_dir.mkdir(exist_ok=True)
_run_log_file = _logs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_file_handler = logging.FileHandler(_run_log_file)
_file_handler.setLevel(logging.INFO)
_file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_file_handler.setFormatter(_file_formatter)
_root_logger = logging.getLogger()
_root_logger.addHandler(_file_handler)
logger.info(f"Run log file: {_run_log_file}")

# Export run log file path for reference
def get_run_log_file():
    """Get the current run log file path."""
    return _run_log_file

