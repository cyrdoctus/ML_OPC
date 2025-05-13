# ML_Orbital_Corrector_Predictor/ml_models/ai1_predictor_def.py

import torch
import torch.nn as nn
import collections

# --- Model Configuration Constants ---
# These should ideally be configurable, perhaps loaded from model_hyperparams.py or passed as arguments

# INPUT_SEQUENCE_LENGTH: Number of past time steps fed into the model.
DEFAULT_INPUT_SEQUENCE_LENGTH = 20

# PREDICTION_HORIZON: Number of future time steps the model will predict.
DEFAULT_PREDICTION_HORIZON = 10

# NUM_INPUT_FEATURES: Number of features per time step in the input sequence.
DEFAULT_NUM_INPUT_FEATURES = 6  # norm_pos (3) + vel (3)

# NUM_OUTPUT_FEATURES: Number of features per time step in the output sequence.
DEFAULT_NUM_OUTPUT_FEATURES = 6  # norm_pos (3) + vel (3)


# --- Encoder-Decoder Model (PyTorch) ---

class Encoder(nn.Module):
    """
    Encoder part of the Encoder-Decoder LSTM model.
    """

    def __init__(self, input_features: int, lstm_units_encoder: list[int], dropout_rate: float):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        current_features = input_features
        for i, units in enumerate(lstm_units_encoder):
            is_last_layer = (i == len(lstm_units_encoder) - 1)
            self.layers.append(nn.LSTM(current_features, units, batch_first=True, num_layers=1))
            current_features = units  # Output features of this LSTM become input for the next
            if not is_last_layer:  # No BN/Dropout after the final LSTM outputting context
                self.layers.append(nn.BatchNorm1d(units))  # Expects (N, C) or (N, C, L) - need to reshape LSTM output
                self.layers.append(nn.Dropout(dropout_rate))
        self.lstm_units_encoder = lstm_units_encoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.
        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, seq_len, input_features).
        Returns:
            tuple:
                - encoder_outputs (torch.Tensor): Output from the last LSTM layer if return_sequences=True for it.
                                                 Shape (batch_size, seq_len, last_lstm_units).
                                                 If last LSTM return_sequences=False (not typical for direct state passing),
                                                 this would be the last hidden state.
                - hidden (torch.Tensor): Hidden state of the last LSTM layer, shape (num_layers*num_directions, batch_size, hidden_size).
                - cell (torch.Tensor): Cell state of the last LSTM layer, shape (num_layers*num_directions, batch_size, hidden_size).
        """
        hidden, cell = None, None  # To store the final states
        encoder_output = x  # Initialize with input

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                encoder_output, (hidden, cell) = layer(encoder_output)  # LSTM returns output, (h_n, c_n)
            elif isinstance(layer, nn.BatchNorm1d):
                # BatchNorm1d expects (N, C) or (N, C, L). LSTM output is (N, L, C) if batch_first=True.
                # So, we need to permute, apply BN, then permute back.
                encoder_output = encoder_output.permute(0, 2, 1)  # (N, C, L)
                encoder_output = layer(encoder_output)
                encoder_output = encoder_output.permute(0, 2, 1)  # (N, L, C)
            elif isinstance(layer, nn.Dropout):
                encoder_output = layer(encoder_output)

        # The 'encoder_output' from the last LSTM (if it returned sequences)
        # and the final 'hidden' and 'cell' states are what we need.
        # Typically, for an encoder, we only pass the final hidden and cell states.
        return encoder_output, hidden, cell


class Decoder(nn.Module):
    """
    Decoder part of the Encoder-Decoder LSTM model.
    """

    def __init__(self, output_features: int, lstm_units_decoder: list[int],
                 dense_units: list[int], dropout_rate: float):
        super(Decoder, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.output_features = output_features

        # Assuming the first LSTM in decoder takes concatenated context or similar feature size
        # For simplicity, let's assume the first decoder LSTM input feature size matches output_features
        # or the hidden size of the last encoder LSTM if context is directly fed.
        # Here, we'll assume input to decoder LSTM cell is `output_features` (for teacher forcing)
        current_lstm_features = output_features  # Input to first decoder LSTM (target sequence feature)

        for units in lstm_units_decoder:
            self.lstm_layers.append(nn.LSTM(current_lstm_features, units, batch_first=True, num_layers=1))
            self.lstm_layers.append(nn.BatchNorm1d(units))  # Expects (N, C) or (N, C, L)
            self.lstm_layers.append(nn.Dropout(dropout_rate))
            current_lstm_features = units  # Output of this LSTM is input to next

        current_dense_features = current_lstm_features  # Output of last LSTM
        for units in dense_units:
            self.dense_layers.append(nn.Linear(current_dense_features, units))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.BatchNorm1d(units))
            self.dense_layers.append(nn.Dropout(dropout_rate))
            current_dense_features = units

        self.output_layer = nn.Linear(current_dense_features, output_features)

    def forward(self, x: torch.Tensor, hidden_encoder: torch.Tensor, cell_encoder: torch.Tensor,
                prediction_horizon: int) -> torch.Tensor:
        """
        Forward pass for the decoder.
        Args:
            x (torch.Tensor): Input sequence for teacher forcing (e.g., ground truth shifted).
                              Shape (batch_size, prediction_horizon, num_output_features).
                              During inference, this would be the previously predicted output.
            hidden_encoder (torch.Tensor): Initial hidden state from encoder.
            cell_encoder (torch.Tensor): Initial cell state from encoder.
            prediction_horizon (int): Number of steps to predict.

        Returns:
            torch.Tensor: Output sequence, shape (batch_size, prediction_horizon, num_output_features).
        """
        # For a simple decoder that processes the whole sequence with teacher forcing:
        decoder_output = x
        current_hidden, current_cell = hidden_encoder, cell_encoder

        for i in range(0, len(self.lstm_layers), 3):  # LSTM, BN, Dropout
            lstm_layer = self.lstm_layers[i]
            bn_layer = self.lstm_layers[i + 1]
            dropout_layer = self.lstm_layers[i + 2]

            decoder_output, (current_hidden, current_cell) = lstm_layer(decoder_output, (current_hidden, current_cell))

            # Permute for BatchNorm1d
            original_shape = decoder_output.shape
            if len(original_shape) == 3:  # (N, L, C)
                decoder_output_permuted = decoder_output.permute(0, 2, 1)  # (N, C, L)
                bn_output = bn_layer(decoder_output_permuted)
                decoder_output = bn_output.permute(0, 2, 1)  # (N, L, C)
            elif len(original_shape) == 2:  # (N, C) - if LSTM output single step
                decoder_output = bn_layer(decoder_output)

            decoder_output = dropout_layer(decoder_output)

        # Apply dense layers (TimeDistributed equivalent)
        # The LSTM output is (batch_size, seq_len, features). Linear layers expect (..., in_features).
        # We can apply Linear to each time step.
        outputs_seq = []
        for t in range(decoder_output.size(1)):  # Iterate over time steps
            dense_out_t = decoder_output[:, t, :]
            for i in range(0, len(self.dense_layers), 4):  # Linear, ReLU, BN, Dropout
                lin_layer = self.dense_layers[i]
                relu_layer = self.dense_layers[i + 1]
                bn_dense_layer = self.dense_layers[i + 2]
                dropout_dense_layer = self.dense_layers[i + 3]

                dense_out_t = lin_layer(dense_out_t)
                dense_out_t = relu_layer(dense_out_t)
                dense_out_t = bn_dense_layer(dense_out_t)  # BN1d expects (N,C)
                dense_out_t = dropout_dense_layer(dense_out_t)

            final_out_t = self.output_layer(dense_out_t)
            outputs_seq.append(final_out_t.unsqueeze(1))  # Add time dimension back

        return torch.cat(outputs_seq, dim=1)


class AI1PredictorEncoderDecoder(nn.Module):
    """
    AI1 Orbit Predictor Model (Sequence-to-Sequence LSTM) using PyTorch.
    """

    def __init__(self,
                 input_features: int = DEFAULT_NUM_INPUT_FEATURES,
                 output_features: int = DEFAULT_NUM_OUTPUT_FEATURES,
                 prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
                 lstm_units_encoder: list[int] = [128, 64],
                 lstm_units_decoder: list[int] = [64, 128],
                 dense_units: list[int] = [64],
                 dropout_rate: float = 0.2):
        super(AI1PredictorEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_features, lstm_units_encoder, dropout_rate)
        # The first LSTM in decoder should take `output_features` if using teacher forcing with target sequence.
        # Or, it could take `lstm_units_encoder[-1]` if directly feeding encoder's output sequence.
        # For this example, assuming teacher forcing with target sequence as input to decoder.
        self.decoder = Decoder(output_features, lstm_units_decoder, dense_units, dropout_rate)
        self.prediction_horizon = prediction_horizon
        self.output_features = output_features

    def forward(self, encoder_input_seq: torch.Tensor, decoder_input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Encoder-Decoder model.

        Args:
            encoder_input_seq (torch.Tensor): Past sequence.
                                             Shape (batch_size, input_sequence_length, num_input_features).
            decoder_input_seq (torch.Tensor): Target sequence for teacher forcing.
                                             Shape (batch_size, prediction_horizon, num_output_features).
                                             During inference, this would be generated step-by-step.
        Returns:
            torch.Tensor: Predicted future sequence.
                          Shape (batch_size, prediction_horizon, num_output_features).
        """
        _, hidden_encoder, cell_encoder = self.encoder(encoder_input_seq)

        # Decoder input for teacher forcing is the ground truth sequence
        # For inference, one would typically loop `prediction_horizon` times,
        # feeding the previous output as the next input, starting with a <SOS> token.
        # This simplified forward pass assumes teacher forcing is handled by `decoder_input_seq`.
        output = self.decoder(decoder_input_seq, hidden_encoder, cell_encoder, self.prediction_horizon)
        return output


# --- Simpler LSTM Model (PyTorch) ---

class AI1PredictorSimpleLSTM(nn.Module):
    """
    Simpler AI1 orbit predictor model using stacked LSTMs.
    Predicts the entire future sequence from the final hidden state of the LSTM.
    """

    def __init__(self,
                 input_features: int = DEFAULT_NUM_INPUT_FEATURES,
                 output_features: int = DEFAULT_NUM_OUTPUT_FEATURES,
                 prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
                 lstm_units: list[int] = [128, 64, 32],
                 dense_units: list[int] = [64],
                 dropout_rate: float = 0.2):
        super(AI1PredictorSimpleLSTM, self).__init__()
        self.prediction_horizon = prediction_horizon
        self.output_features = output_features

        self.lstm_layers_seq = nn.Sequential()
        current_features = input_features
        for i, units in enumerate(lstm_units):
            self.lstm_layers_seq.add_module(f"lstm_{i + 1}", nn.LSTM(current_features, units, batch_first=True))
            # BatchNorm1d needs to be applied carefully after LSTM if return_sequences=True
            # For this simple model, we take the last hidden state, so BN on that.
            current_features = units

        self.dense_layers_seq = nn.Sequential()
        # Input to dense layers is the output of the last LSTM layer (last hidden state)
        current_dense_features = lstm_units[-1]
        for i, units in enumerate(dense_units):
            self.dense_layers_seq.add_module(f"dense_{i + 1}", nn.Linear(current_dense_features, units))
            self.dense_layers_seq.add_module(f"relu_{i + 1}", nn.ReLU())
            self.dense_layers_seq.add_module(f"bn_dense_{i + 1}", nn.BatchNorm1d(units))
            self.dense_layers_seq.add_module(f"dropout_dense_{i + 1}", nn.Dropout(dropout_rate))
            current_dense_features = units

        self.output_transform = nn.Linear(current_dense_features, prediction_horizon * output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the simple LSTM model.
        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, input_sequence_length, num_input_features).
        Returns:
            torch.Tensor: Predicted future sequence,
                          shape (batch_size, prediction_horizon, num_output_features).
        """
        # Pass through LSTMs
        # We need the output of the last time step from the last LSTM layer
        lstm_out = x
        for i in range(len(self.lstm_layers_seq)):  # Iterate through LSTM modules in sequential
            lstm_out, _ = self.lstm_layers_seq[i](lstm_out)  # (output, (h_n, c_n))

        # Take the output of the last LSTM layer corresponding to the last time step
        # If all LSTMs have return_sequences=True, lstm_out is (batch, seq_len, features)
        # We need the features from the last time step: lstm_out[:, -1, :]
        final_lstm_output_state = lstm_out[:, -1, :]  # (batch_size, last_lstm_units)

        # Pass through dense layers
        dense_output = self.dense_layers_seq(final_lstm_output_state)

        # Transform to final output shape
        output_flat = self.output_transform(dense_output)

        # Reshape to (batch_size, prediction_horizon, num_output_features)
        output_reshaped = output_flat.view(-1, self.prediction_horizon, self.output_features)

        return output_reshaped


# --- Helper function to create models (similar to Keras version) ---
def create_ai1_predictor_pytorch_model(
        model_type: str = "encoder_decoder",  # "encoder_decoder" or "simple_lstm"
        input_sequence_length: int = DEFAULT_INPUT_SEQUENCE_LENGTH,
        prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
        num_input_features: int = DEFAULT_NUM_INPUT_FEATURES,
        num_output_features: int = DEFAULT_NUM_OUTPUT_FEATURES,
        lstm_units_encoder: list[int] = [128, 64],
        lstm_units_decoder: list[int] = [64, 128],
        lstm_units_simple: list[int] = [128, 64, 32],
        dense_units: list[int] = [64],
        dropout_rate: float = 0.2
        # Learning rate is handled by the optimizer in the training script for PyTorch
) -> nn.Module:
    """
    Factory function to create the AI1 predictor model using PyTorch.

    Args:
        model_type (str): Type of model to create ("encoder_decoder" or "simple_lstm").
        input_sequence_length, prediction_horizon, num_input_features, num_output_features: Model shape params.
        lstm_units_encoder, lstm_units_decoder, lstm_units_simple: LSTM layer units.
        dense_units: Dense layer units.
        dropout_rate: Dropout rate.

    Returns:
        torch.nn.Module: The PyTorch model instance.
    """
    if model_type == "encoder_decoder":
        print(f"Creating PyTorch AI1 Predictor Model (Encoder-Decoder):")
        model = AI1PredictorEncoderDecoder(
            input_features=num_input_features,
            output_features=num_output_features,
            prediction_horizon=prediction_horizon,
            lstm_units_encoder=lstm_units_encoder,
            lstm_units_decoder=lstm_units_decoder,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )
    elif model_type == "simple_lstm":
        print(f"Creating PyTorch AI1 Predictor Model (Simple LSTM):")
        model = AI1PredictorSimpleLSTM(
            input_features=num_input_features,
            output_features=num_output_features,
            prediction_horizon=prediction_horizon,
            lstm_units=lstm_units_simple,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'encoder_decoder' or 'simple_lstm'.")

    # In PyTorch, model.summary() is not a built-in method.
    # One can print the model structure: print(model)
    # Or use libraries like torchinfo: from torchinfo import summary; summary(model, input_size=...)
    print(model)
    return model


if __name__ == '__main__':
    print("--- Testing AI1 Predictor Model Creation (PyTorch) ---")
    batch_size = 4  # Example batch size

    # Test the Encoder-Decoder model
    print("\nTesting Encoder-Decoder Model (PyTorch):")
    pytorch_model_encdec = create_ai1_predictor_pytorch_model(
        model_type="encoder_decoder",
        input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        num_input_features=DEFAULT_NUM_INPUT_FEATURES,
        num_output_features=DEFAULT_NUM_OUTPUT_FEATURES
    )
    # Dummy input data for testing the forward pass
    dummy_encoder_input = torch.randn(batch_size, DEFAULT_INPUT_SEQUENCE_LENGTH, DEFAULT_NUM_INPUT_FEATURES)
    dummy_decoder_input = torch.randn(batch_size, DEFAULT_PREDICTION_HORIZON,
                                      DEFAULT_NUM_OUTPUT_FEATURES)  # For teacher forcing

    try:
        output_encdec = pytorch_model_encdec(dummy_encoder_input, dummy_decoder_input)
        print(f"Encoder-Decoder Output Shape: {output_encdec.shape}")  # Expected: (batch_size, horizon, out_features)
    except Exception as e:
        print(f"Error during Encoder-Decoder forward pass: {e}")
        import traceback

        traceback.print_exc()

    # Test the Simple LSTM model
    print("\nTesting Simple LSTM Model (PyTorch):")
    pytorch_model_simple = create_ai1_predictor_pytorch_model(
        model_type="simple_lstm",
        input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        num_input_features=DEFAULT_NUM_INPUT_FEATURES,
        num_output_features=DEFAULT_NUM_OUTPUT_FEATURES
    )
    dummy_simple_input = torch.randn(batch_size, DEFAULT_INPUT_SEQUENCE_LENGTH, DEFAULT_NUM_INPUT_FEATURES)
    try:
        output_simple = pytorch_model_simple(dummy_simple_input)
        print(f"Simple LSTM Output Shape: {output_simple.shape}")  # Expected: (batch_size, horizon, out_features)
    except Exception as e:
        print(f"Error during Simple LSTM forward pass: {e}")
        import traceback

        traceback.print_exc()

    print("\nPyTorch Model creation tests complete.")
    # Note: Optimizers and loss functions are defined and used in the training script.
