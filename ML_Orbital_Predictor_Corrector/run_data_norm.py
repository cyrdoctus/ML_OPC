# ML_Orbital_Predictor_Corrector/run_data_norm.py

from data_generation.normalizer import process_and_normalize_all_raw_data

def main():
    """Orchestrate the full normalization pipeline."""
    print("--- Orbital Data Normalization Pipeline ---")
    process_and_normalize_all_raw_data()
    print("--- Normalization Complete ---")

if __name__ == "__main__":
    main()