# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:19:39 2024

@author: huber
"""

from pathlib import Path
import logging
import sys

from utils.file_utils import create_directory, delete_directory


# Define directories
ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / 'logs'
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
HISTOGRAM_ORYGINAL_PLOTS_DIR = PLOTS_DIR / 'oryginal_meas' / 'histograms'
SCATTER_ORYGINAL_PLOTS_DIR = PLOTS_DIR / 'oryginal_meas' / 'scatters'
ANALYSIS_DATA_DIR = RESULTS_DIR / "analysis_data"


# Create directories if they don't exist
for directory in [LOGS_DIR, DATA_DIR, RESULTS_DIR, PLOTS_DIR, ANALYSIS_DATA_DIR, HISTOGRAM_ORYGINAL_PLOTS_DIR, SCATTER_ORYGINAL_PLOTS_DIR]:
    create_directory(directory)
    
LOGGING_LVL = 'INFO'  # Can be changed to 'DEBUG', 'ERROR', 'INFO', 'WARNING' , etc.
LOG_FILE = LOGS_DIR / 'app.log'

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# Map string levels to logging constants
log_level = getattr(logging, LOGGING_LVL.upper(), logging.INFO)

# Set up logging
logging.basicConfig(
    level=log_level,  # Set the log level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Define log format
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w')  # Log to file, overwrite each time
    ]
)

# Get the logger instance
logger = logging.getLogger(__name__)

#%%
COOPERATION_1_TIME_INTERVALS = {
#   (0, 8000): "1_z1_instr",#nie
    (8000, 28000): "1_z1_1_k",
    (28000, 48000): "1_z1_2_m",
    (48000, 68000): "1_z1_3_k",
    (68000, 88000): "1_z1_4_m",
    (88000, 108000): "1_z1_5_k",
    (108000, 128000): "1_z1_6_m",
#   (128000, 136000): "1_z1_odp_idle",#nie
#   (226000, 226000): "1_z1_odp",#nie
#   (246000, 286000): "1_pause",#nie
#   (286000, 294000): "1_z2_instr",#nie
    (294000, 314000): "1_z2_1_m",
    (314000, 334000): "1_z2_2_k",
    (334000, 354000): "1_z2_3_m",
    (354000, 374000): "1_z2_4_k",
    (374000, 394000): "1_z2_5_m",
    (394000, 414000): "1_z2_6_k",
#   (414000, 422000): "1_z2_odp_idle",#nie
#   (422000, 512000): "1_z2_odp",#nie
#   (512000, 547000): "1_baseline2_idle",#nie
    (547000, 787000): "1_baseline2",
}

COOPERATION_2_TIME_INTERVALS = {
#   (0, 8000): "2_z1_instr",
    (8000, 28000): "2_z1_1_k",
    (28000, 48000): "2_z1_2_m",
    (48000, 68000): "2_z1_3_k",
    (68000, 88000): "2_z1_4_m",
    (88000, 108000): "2_z1_5_k",
    (108000, 128000): "2_z1_6_m",
#   (128000, 136000): "2_z1_odp_idle",
#   (226000, 226000): "2_z1_odp",
#   (246000, 286000): "2_pause",
#   (286000, 294000): "2_z2_instr",
    (294000, 314000): "2_z2_1_m",
    (314000, 334000): "2_z2_2_k",
    (334000, 354000): "2_z2_3_m",
    (354000, 374000): "2_z2_4_k",
    (374000, 394000): "2_z2_5_m",
    (394000, 414000): "2_z2_6_k",
#   (414000, 422000): "2_z2_odp_idle",
#   (422000, 512000): "2_z2_odp",
#   (512000, 547000): "2_baseline2_idle",
    (547000, 787000): "2_baseline2",
}

BASELINE_1_TIME_INTERVALS = {
#   (0, 20000): "1_baseline1_idle",#nie
    (20000, 260000): "1_baseline1",
}

BASELINE_2_TIME_INTERVALS = {
#   (0, 20000): "2_baseline1_idle",
    (20000, 260000): "2_baseline1",
}

RELAXATION_1_TIME_INTERVALS = {
    (0, float('inf')): "1_relaxation"
}

RELAXATION_2_TIME_INTERVALS = {
    (0, float('inf')): "2_relaxation"
}

# Define meas_types with measurement type and regex pattern
MEAS_TYPES = [
    ('Relaxation', r'[12]r.*'),
    ('Baseline', r'[12]w.*\d+_1'),
    ('Cooperation', r'[12]w.*\d+_(?!1)')
]

intervals = {
    'BASELINE_1': BASELINE_1_TIME_INTERVALS,
    'BASELINE_2': BASELINE_2_TIME_INTERVALS,
    'COOPERATION_1': COOPERATION_1_TIME_INTERVALS,
    'COOPERATION_2': COOPERATION_2_TIME_INTERVALS,
    'RELAXATION_1': RELAXATION_1_TIME_INTERVALS,
    'RELAXATION_2': RELAXATION_2_TIME_INTERVALS
}

def get_time_intervals(meas_number: int, meas_type: str) -> dict:
    """
    Returns the appropriate time intervals based on the measurement number and type.
    
    Args:
        meas_number (int): The measurement number, e.g., 1 or 2.
        meas_type (str): The measurement type, e.g., 'Baseline' or 'Cooperation'.
    
    Returns:
        dict: The corresponding dictionary of time intervals.
    """
    # Create the key name based on meas_number and meas_type
    key = f"{meas_type.upper()}_{meas_number}"
    
    # Return the appropriate time intervals
    return intervals.get(key, None)

#%%
