#!/usr/bin/env python3
# run_ai_train.py
import argparse
import os
from config.app_config import MODEL_ARTIFACTS_DIR, NORMALIZED_GENERATED_DATA_DIR
from training.predictor import train_lstm_predictor
from training.corrector import train_fnn_corrector

def main():
    parser = argparse.ArgumentParser(
        description="Train one of the orbital-AI models (LSTM predictor or FNN corrector)."
    )
    parser.add_argument(
        "--model", choices=["ai1", "ai2"],
        help="Which AI to train: 'ai1' for LSTM predictor, 'ai2' for FNN corrector."
    )
    args = parser.parse_args()

    choice = args.model or input("Which model to train? [ai1 / ai2]: ").strip()

    if choice == "ai1":
        ckpt_dir = os.path.join(MODEL_ARTIFACTS_DIR, "ai1_predictor")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_lstm_predictor(
            data_dir=NORMALIZED_GENERATED_DATA_DIR,
            checkpoint_dir=ckpt_dir
        )
    elif choice == "ai2":
        ckpt_dir = os.path.join(MODEL_ARTIFACTS_DIR, "ai2_corrector")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_fnn_corrector(
            data_dir=NORMALIZED_GENERATED_DATA_DIR,
            checkpoint_dir=ckpt_dir
        )
    else:
        print(f"Unrecognized choice '{choice}'. Exiting.")

if __name__ == "__main__":
    main()
