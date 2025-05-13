# my_ai_orbital_project/data_generation/__init__.py

"""
Data Generation Package

This package is responsible for:
- Defining orbital scenarios (scenarios.py).
- Saving raw and normalized trajectory data (data_saver.py).
- Normalizing raw data and engineering features (normalizer.py).
- Orchestrating the entire data generation pipeline (run_generation.py).
"""

# Import key functions, classes, or variables from the modules in this package
# to make them easily accessible at the package level.
# For example: from data_generation import LEO_ORBIT_ELEMENT_RANGES

# From scenarios.py:
# You might want to expose the ranges dictionary or a function that generates scenarios.
try:
    from .scenarios import LEO_ORBIT_ELEMENT_RANGES
except ImportError:
    print("Warning: Could not import LEO_ORBIT_ELEMENT_RANGES from data_generation.scenarios")
    LEO_ORBIT_ELEMENT_RANGES = {} # Fallback

# From data_saver.py:
# Expose the saving functions and the output directory paths.
try:
    from .data_saver import (
        save_raw_trajectory,
        save_norm_trajectory,
        RAW_TRAJECTORIES_OUTPUT_DIR,
        NORMALIZED_TRAJECTORIES_OUTPUT_DIR
    )
except ImportError:
    print("Warning: Could not import from data_generation.data_saver")
    # Define fallbacks or leave them undefined if critical
    def save_raw_trajectory(*args, **kwargs):
        print("Fallback save_raw_trajectory called. Please check data_saver.py.")
    def save_norm_trajectory(*args, **kwargs):
        print("Fallback save_norm_trajectory called. Please check data_saver.py.")
    RAW_TRAJECTORIES_OUTPUT_DIR = "data_training/raw_trajectories_fallback_init"
    NORMALIZED_TRAJECTORIES_OUTPUT_DIR = "data_training/normalized_trajectories_fallback_init"


# From normalizer.py (once you create it):
# You might have a main processing function, e.g., process_and_normalize_all_raw_data
# try:
#     from .normalizer import normalize_trajectory_data # Example function name
# except ImportError:
#     print("Warning: Could not import from data_generation.normalizer")
#     def normalize_trajectory_data(*args, **kwargs):
#         print("Fallback normalize_trajectory_data called. Please check normalizer.py.")


# It's good practice to define __all__ to specify the public API of the package
# when 'from data_generation import *' is used.
__all__ = [
    'LEO_ORBIT_ELEMENT_RANGES',
    'save_raw_trajectory',
    'save_norm_trajectory',
    'RAW_TRAJECTORIES_OUTPUT_DIR',
    'NORMALIZED_TRAJECTORIES_OUTPUT_DIR',
    # Add other imported names here, e.g., 'normalize_trajectory_data'
]

print("Data Generation package initialized.")
