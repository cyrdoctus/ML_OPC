# training/predictor.py

import os
import glob
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

from config.model_hyperparams import (
    PREDICTOR_MODEL_CONFIG as cfg,
    VALIDATION_SPLIT_RATIO,
    RANDOM_SEED,
    DTYPE
)
from config.app_config import NORMALIZED_GENERATED_DATA_DIR

# ─── FORCE CPU ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
torch.set_default_dtype(DTYPE)
torch.manual_seed(RANDOM_SEED)

# ==== CPU threading / oneDNN tweaks ====
os.environ.setdefault('OMP_NUM_THREADS', '24')
os.environ.setdefault('MKL_NUM_THREADS', '24')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '24')
torch.set_num_threads(32)
torch.set_num_interop_threads(16)
torch.backends.mkldnn.enabled = True


class OrbitDataset(Dataset):
    """
    Loads each .npz in data_dir and slices it into (seq_in → seq_out) windows.
    Assumes each file contains:
      - 'positions_norm' shape (T,3)
      - 'velocities_norm' shape (T,3)
    """
    def __init__(self, data_dir: str, seq_in: int, seq_out: int):
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        self.index_map = []  # list of (file_path, start_idx)

        for fp in self.file_paths:
            data = np.load(fp)
            pos = data['positions_norm']
            vel = data['velocities_norm']
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
        pos = data['positions_norm']
        vel = data['velocities_norm']
        states = np.hstack((pos, vel))  # (T,6)

        xin = states[start : start + self.seq_in]
        xout = states[start + self.seq_in : start + self.seq_in + self.seq_out]

        return (
            torch.from_numpy(xin).to(DEVICE),
            torch.from_numpy(xout).to(DEVICE)
        )


class LSTMPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_feats = len(config['features'])
        out_feats = len(config['targets'])
        seq_out  = config['sequence_length_out']
        u1, u2   = config['lstm_units']

        # Because dropout>0 on a single-layer LSTM warns you,
        # we wrap the dropout only if num_layers>1.
        self.lstm1 = nn.LSTM(input_size=in_feats,
                             hidden_size=u1,
                             batch_first=True,
                             num_layers=1,
                             dropout=0.0)
        self.lstm2 = nn.LSTM(input_size=u1,
                             hidden_size=u2,
                             batch_first=True,
                             num_layers=1,
                             dropout=0.0)

        self.dropout = nn.Dropout(config['dropout_rate'])
        # fully-connected after RNN
        layers = []
        last = u2
        for h in config['dense_units_after_recurrent']:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(config['dropout_rate'])]
            last = h
        self.dense_net = nn.Sequential(*layers)
        self.output    = nn.Linear(last, seq_out * out_feats)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        h = out[:, -1, :]            # pick last time step
        h = self.dropout(h)
        h = self.dense_net(h)
        h = self.output(h)
        B = h.size(0)
        return h.view(B, cfg['sequence_length_out'], len(cfg['targets']))


def train_predictor(
    data_dir: str = NORMALIZED_GENERATED_DATA_DIR,
    checkpoint_dir: str = os.path.join("data", "predictor")
):
    """
    Train the LSTM predictor on all .npz files in data_dir
    and save the final .pth in checkpoint_dir.
    """
    print(f"👉 Starting training of LSTM predictor (AI1) on CPU...")

    # 1) prepare data
    dataset = OrbitDataset(data_dir, cfg['sequence_length_in'], cfg['sequence_length_out'])
    total   = len(dataset)
    val_sz  = int(total * VALIDATION_SPLIT_RATIO)
    train_sz= total - val_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
    train_loader    = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,num_workers=24)
    val_loader      = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,num_workers=24)

    # 2) model / optimizer / loss
    model     = LSTMPredictor(cfg).to(DEVICE)
    criterion = nn.MSELoss()
    Optim     = getattr(optim, cfg['optimizer'])
    optimizer = Optim(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['l2_reg'])

    # 3) training loop
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE).float()  # cast from float64→float32
            y = y.to(DEVICE).float()

            optimizer.zero_grad()
            pred  = model(x)
            loss  = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        avg_train = train_loss / train_sz

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x), y).item() * x.size(0)
        avg_val = val_loss / val_sz

        print(f"Epoch {epoch}/{cfg['epochs']}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

    # 4) save final checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "lstm_predictor_cpu.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved CPU‐only predictor to {save_path}")
