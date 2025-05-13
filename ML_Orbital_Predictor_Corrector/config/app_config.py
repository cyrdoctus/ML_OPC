# ML_Orbital_Predictor_Corrector/config/app_config.py
import os

print("Loading application configuration from config/app_config.py...")

# --- Path for Storing/Loading Trained AI Model Artifacts ---
MODEL_ARTIFACTS_DIR = "/home/nekolny/PycharmProjects/ECE579_FInal_Project/ML_Orbital_Predictor_Corrector/data"

# --- Paths for Generated Trajectory Data (on LAN storage via GVFS) ---
# Path for RAW generated data
RAW_GENERATED_DATA_DIR = "/run/user/1000/gvfs/smb-share:server=truenas.local,share=arrakis/ece579/ece579_data_raw"

# Path for NORMALIZED generated data ready for training
NORMALIZED_GENERATED_DATA_DIR = "/run/user/1000/gvfs/smb-share:server=truenas.local,share=arrakis/ece579/ece579_data_norm"

# --- Paths for Output of the AI Driven Simulation ---
SIMULATION_OUTPUT_DIR = "/home/nekolny/PycharmProjects/ECE579_FInal_Project/ML_Orbital_Predictor_Corrector/data_sim"

# --- Checks for Paths ---

def check_path(path_variable_name, path_value, create_if_not_exists=False, is_critical=True):
    """Helper function to check if a path exists and optionally create it."""
    exists = os.path.exists(path_value)
    if exists and not os.path.isdir(path_value): # It exists but is not a directory
        message = f"CRITICAL ERROR: Configured path {path_variable_name} ('{path_value}') exists but is not a directory."
        print(message)
        if is_critical:
            raise NotADirectoryError(message)
        return False
    elif not exists:
        if create_if_not_exists:
            try:
                os.makedirs(path_value, exist_ok=True)
                print(f"INFO: Created directory for {path_variable_name}: {path_value}")
                return True
            except Exception as e:
                message = f"ERROR: Could not create directory for {path_variable_name} ('{path_value}'): {e}"
                print(message)
                if is_critical:
                    raise OSError(message) from e
                return False
        else:
            message = f"WARNING: Configured path {path_variable_name} ('{path_value}') does not exist."
            print(message)
            if is_critical:
                critical_error_message = (
                    f"CRITICAL CONFIG ERROR: The required path for '{path_variable_name}'\n"
                    f"('{path_value}')\n"
                    f"does not exist and is marked as critical. Application cannot proceed."
                )
                print(critical_error_message)
                raise FileNotFoundError(critical_error_message)
            return False
    else: # Path exists and is a directory
        print(f"INFO: Confirmed path for {path_variable_name}: {path_value}")
        return True

# Perform checks:
check_path("MODEL_ARTIFACTS_DIR", MODEL_ARTIFACTS_DIR, create_if_not_exists=True, is_critical=True)
check_path("RAW_GENERATED_DATA_DIR", RAW_GENERATED_DATA_DIR, create_if_not_exists=False, is_critical=False)
check_path("NORMALIZED_GENERATED_DATA_DIR", NORMALIZED_GENERATED_DATA_DIR, create_if_not_exists=False, is_critical=False)
check_path("SIMULATION_OUTPUT_DIR", SIMULATION_OUTPUT_DIR, create_if_not_exists=True, is_critical=True)

print("Application configuration loaded successfully.")
