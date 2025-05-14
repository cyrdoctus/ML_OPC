"""
LSTM-based Orbit Predictor

Loads normalized .npz data, builds an LSTM model, and runs training.
"""

import os
import glob
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

# Hyperparameters and paths
from config.model_hyperparams import (
    PREDICTOR_MODEL_CONFIG as cfg,
    VALIDATION_SPLIT_RATIO,
    RANDOM_SEED,
    DEVICE,
    DTYPE
)
from config.app_config import NORMALIZED_GENERATED_DATA_DIR

# Reproducibility
torch.manual_seed(RANDOM_SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(RANDOM_SEED)
torch.set_default_dtype(DTYPE)


class OrbitDataset(Dataset):
    """
    Loads each .npz in NORMALIZED_GENERATED_DATA_DIR and
    slices it into sliding windows of (seq_in → seq_out).
    Assumes each file contains:
      - 'positions_normalized' shape (T,3)
      - 'velocities_normalized' shape (T,3)
    """

    def __init__(self, data_dir: str, seq_in: int, seq_out: int):
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        self.index_map = []  # list of (file_path, start_idx)

        for fp in self.file_paths:
            data = np.load(fp)
            pos = data['positions_normalized']
            vel = data['velocities_normalized']
            states = np.hstack((pos, vel))  # (T,6)
            length = states.shape[0]
            max_start = length - seq_in - seq_out + 1
            for i in range(max_start):
                self.index_map.append((fp, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fp, start = self.index_map[idx]
        data = np.load(fp)
        pos = data['positions_normalized']
        vel = data['velocities_normalized']
        states = np.hstack((pos, vel))  # (T,6)

        seq_in = states[start : start + self.seq_in]
        seq_out = states[start + self.seq_in : start + self.seq_in + self.seq_out]

        # to torch.Tensor
        return (
            torch.from_numpy(seq_in).to(device=DEVICE),
            torch.from_numpy(seq_out).to(device=DEVICE)
        )


class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_feats = len(config['features'])
        out_feats = len(config['targets'])
        seq_out = config['sequence_length_out']
        units = config['lstm_units']

        # build two-layer LSTM manually
        self.lstm1 = nn.LSTM(input_size=in_feats,
                             hidden_size=units[0],
                             batch_first=True,
                             dropout=config['recurrent_dropout'])
        self.lstm2 = nn.LSTM(input_size=units[0],
                             hidden_size=units[1],
                             batch_first=True,
                             dropout=config['recurrent_dropout'])

        # optional dense layers
        self.dropout = nn.Dropout(config['dropout_rate'])
        dense_layers = []
        last = units[1]
        for h in config['dense_units_after_recurrent']:
            dense_layers.append(nn.Linear(last, h))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(config['dropout_rate']))
            last = h
        self.dense_net = nn.Sequential(*dense_layers)

        # final output: flatten into (seq_out × out_feats)
        self.output = nn.Linear(last, seq_out * out_feats)

    def forward(self, x):
        # x: (B, seq_in, in_feats)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # take only last time step
        h = out[:, -1, :]
        h = self.dropout(h)
        h = self.dense_net(h)
        h = self.output(h)
        B = h.size(0)
        # reshape → (B, seq_out, out_feats)
        return h.view(B, cfg['sequence_length_out'], len(cfg['targets']))


def train_predictor():
    # Prepare dataset and loaders
    dataset = OrbitDataset(
        NORMALIZED_GENERATED_DATA_DIR,
        cfg['sequence_length_in'],
        cfg['sequence_length_out']
    )
    total = len(dataset)
    val_sz = int(total * VALIDATION_SPLIT_RATIO)
    train_sz = total - val_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)

    # Model, loss, optimizer
    model = LSTMPredictor(cfg).to(DEVICE)
    criterion = nn.MSELoss()
    Optim = getattr(optim, cfg['optimizer'])
    optimizer = Optim(model.parameters(),
                      lr=cfg['learning_rate'],
                      weight_decay=cfg['l2_reg'])

    # Training loop
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train = total_loss / train_sz

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        avg_val = val_loss / val_sz

        print(f"Epoch {epoch}/{cfg['epochs']}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

    # Save final model
    save_dir = os.path.join("data", "predictor")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lstm_predictor.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved predictor to {save_path}")


if __name__ == "__main__":
    train_predictor()
