# my_ai_orbital_project/config/model_hyperparams.py
"""
Hyperparameters for AI Models (AI1 Predictor and AI2 Corrector).
"""

# --- Common Training Parameters ---
VALIDATION_SPLIT_RATIO = 0.2  # Fraction of data to use for validation
RANDOM_SEED = 42              # For reproducibility

# --- AI1: Orbit Predictor (e.g., LSTM-based) ---
AI1_MODEL_CONFIG = {
    "model_type": "LSTM",             # 'LSTM', 'GRU', 'Transformer'
    "sequence_length_in": 30,         # Number of past time steps to use as input (e.g., 30 steps of 60s = 30 mins)
    "sequence_length_out": 5,         # Number of future time steps to predict (e.g., 5 steps of 60s = 5 mins)
                                      # Ensure PREDICTION_HORIZON_S from sim_params.py relates to this.
    "features": ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'acc_imu_x', 'acc_imu_y', 'acc_imu_z'], # Example input features
    "targets": ['pos_x_future', 'pos_y_future', 'pos_z_future', 'vel_x_future', 'vel_y_future', 'vel_z_future'], # Example output targets

    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adam",                # 'adam', 'rmsprop', 'sgd'

    # LSTM/GRU specific
    "lstm_units": [128, 128],            # List of units for each LSTM/GRU layer (e.g., 2 layers)
    "recurrent_dropout": 0.1,
    "dense_units_after_recurrent": [32], # Optional dense layers after recurrent block
    "activation_after_recurrent": "relu",

    # Transformer specific (if model_type is 'Transformer')
    # "transformer_num_heads": 4,
    # "transformer_num_encoder_layers": 3,
    # "transformer_d_model": 128,       # Embedding dimension
    # "transformer_dff": 256,           # Dimension of feed-forward layer

    "dropout_rate": 0.2,                # General dropout for dense layers
    "l2_reg": 0.0001                     # L2 regularization factor
}

# --- AI2: Orbit Corrector (e.g., FNN for Delta-V regression) ---
AI2_MODEL_CONFIG = {
    "model_type": "FNN",              # 'FNN', 'RL_PPO', 'GP' (Gaussian Process would have different params)
    "input_features": [
        'current_pos_x', 'current_pos_y', 'current_pos_z',
        'current_vel_x', 'current_vel_y', 'current_vel_z',
        'target_ref_pos_x', 'target_ref_pos_y', 'target_ref_pos_z', # Example for targetting a point on ref orbit
        'target_ref_vel_x', 'target_ref_vel_y', 'target_ref_vel_z',
        # OR 'deviation_x', 'deviation_y', ...
    ],
    "output_targets": ['delta_v_x', 'delta_v_y', 'delta_v_z'], # Delta-V components

    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "optimizer": "adam",

    # FNN specific
    "fnn_hidden_layers": [256, 128, 64], # Number of neurons in each hidden layer
    "fnn_activation": "relu",           # Activation for hidden layers
    "fnn_output_activation": "linear",  # For regression of delta-v values

    "dropout_rate": 0.15,
    "l2_reg": 0.0005,

    # Reinforcement Learning specific (if model_type is 'RL_PPO', 'RL_SAC', etc.)
    # "rl_gamma": 0.99,                   # Discount factor
    # "rl_lambda_gae": 0.95,              # Generalized Advantage Estimation lambda
    # "rl_clip_epsilon_ppo": 0.2,         # PPO clipping parameter
    # "rl_replay_buffer_size": 1000000,
    # "rl_steps_per_epoch": 4000,
    # "rl_actor_lr": 3e-4,
    # "rl_critic_lr": 1e-3,
    # "rl_target_entropy": "auto",        # For SAC
    # "rl_alpha": 0.2                     # For SAC
}

print("Model hyperparameters from model_hyperparams.py loaded.")