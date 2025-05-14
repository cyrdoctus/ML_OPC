"""
Hyperparameters for AI Models: Predictor (LSTM) and Corrector (FNN).
"""

import torch

# --- Common Training Parameters ---
VALIDATION_SPLIT_RATIO = 0.2  # Fraction of data to use for validation
RANDOM_SEED = 42              # For reproducibility

# --- Predictor: LSTM-based Orbit Predictor ---
PREDICTOR_MODEL_CONFIG = {
    "model_type": "LSTM",
    "sequence_length_in": 30,      # Number of past steps (e.g. 30 × 10 min = 5 h)
    "sequence_length_out": 6,      # Predict one hour ahead at 10 min steps → 6
    "features": ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'],
    "targets": ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'],

    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "optimizer": "Adam",

    # LSTM-specific
    "lstm_units": [128, 128],            # two-layer LSTM, 256 units each
    "recurrent_dropout": 0.1,
    "dense_units_after_recurrent": [64], # one dense layer after LSTM
    "activation_after_recurrent": "relu",

    "dropout_rate": 0.2,                  # dropout on the dense layers
    "l2_reg": 1e-4                        # weight decay
}

# --- Corrector: FNN-based Orbit Corrector ---
CORRECTOR_MODEL_CONFIG = {
    "model_type": "FNN",
    "input_features": [
        'current_pos_x', 'current_pos_y', 'current_pos_z',
        'current_vel_x', 'current_vel_y', 'current_vel_z',
        'target_pos_x',  'target_pos_y',  'target_pos_z',
        'target_vel_x',  'target_vel_y',  'target_vel_z',
    ],
    "output_targets": ['delta_v_x', 'delta_v_y', 'delta_v_z'],

    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "optimizer": "Adam",

    # FNN-specific
    "fnn_hidden_layers": [128, 128, 64],
    "fnn_activation": "relu",
    "fnn_output_activation": "linear",

    "dropout_rate": 0.15,
    "l2_reg": 5e-4
}

# --- Device and dtype selection ---
# Use GPU (fp32) if available, otherwise CPU (fp64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    DTYPE = torch.float32
else:
    DTYPE = torch.float64

# Allow easy lookup by name
MODEL_CONFIGS = {
    'predictor': PREDICTOR_MODEL_CONFIG,
    'corrector': CORRECTOR_MODEL_CONFIG
}

print(f"Using device={DEVICE}, default dtype={DTYPE}")
