# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:19:39 2024

@author: huber
"""

from pathlib import Path
import logging
import sys

from src.utils.file_utils import create_directory


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
cooperation_1_time_intervals = {
    (0, 8000): "1_z1_instr",#nie
    (8000, 28000): "1_z1_1_k",
    (28000, 48000): "1_z1_2_m",
    (48000, 68000): "1_z1_3_k",
    (68000, 88000): "1_z1_4_m",
    (88000, 108000): "1_z1_5_k",
    (108000, 128000): "1_z1_6_m",
    (128000, 136000): "1_z1_odp_idle",#nie
    (226000, 226000): "1_z1_odp",#nie
    (246000, 286000): "1_pause",#nie
    (286000, 294000): "1_z2_instr",#nie
    (294000, 314000): "1_z2_1_m",
    (314000, 334000): "1_z2_2_k",
    (334000, 354000): "1_z2_3_m",
    (354000, 374000): "1_z2_4_k",
    (374000, 394000): "1_z2_5_m",
    (394000, 414000): "1_z2_6_k",
    (414000, 422000): "1_z2_odp_idle",#nie
    (422000, 512000): "1_z2_odp",#nie
    (512000, 547000): "1_baseline2_idle",#nie
    (547000, 787000): "1_baseline2",
}

cooperation_2_time_intervals = {
    (0, 8000): "2_z1_instr",
    (8000, 28000): "2_z1_1_k",
    (28000, 48000): "2_z1_2_m",
    (48000, 68000): "2_z1_3_k",
    (68000, 88000): "2_z1_4_m",
    (88000, 108000): "2_z1_5_k",
    (108000, 128000): "2_z1_6_m",
    (128000, 136000): "2_z1_odp_idle",
    (226000, 226000): "2_z1_odp",
    (246000, 286000): "2_pause",
    (286000, 294000): "2_z2_instr",
    (294000, 314000): "2_z2_1_m",
    (314000, 334000): "2_z2_2_k",
    (334000, 354000): "2_z2_3_m",
    (354000, 374000): "2_z2_4_k",
    (374000, 394000): "2_z2_5_m",
    (394000, 414000): "2_z2_6_k",
    (414000, 422000): "2_z2_odp_idle",
    (422000, 512000): "2_z2_odp",
    (512000, 547000): "2_baseline2_idle",
    (547000, 787000): "2_baseline2",
}

baseline_1_time_intervals = {
    (0, 20000): "1_baseline1_idle",#nie
    (20000, 260000): "1_baseline1",
}

baseline_2_time_intervals = {
    (0, 20000): "2_baseline1_idle",
    (20000, 260000): "2_baseline1",
}

# Define meas_types with measurement type and regex pattern
meas_types = [
    ('Baseline', '1w', r'.\d+_1'),
    ('Cooperation', '1w', r'.\d+_(?!1)'),
    ('Baseline', '2w', r'.\d+_1'),
    ('Cooperation', '2w', r'.\d+_(?!1)')
]

#%%
