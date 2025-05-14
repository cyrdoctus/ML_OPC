# training/predictor.py

import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from config.app_config import NORMALIZED_GENERATED_DATA_DIR    # from config/app_config.py :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
from config.model_hyperparams import AI1_MODEL_CONFIG, VALIDATION_SPLIT_RATIO, RANDOM_SEED  # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}

class OrbitSequenceDataset(Dataset):
    def __init__(self, data_dir, horizon_steps=6):
        """
        Builds sliding-window (input_seq, target_seq) pairs from all .npz files.
        horizon_steps: number of 10-min steps to forecast (6 → 1 hr).
        """
        self.examples = []
        files = glob.glob(os.path.join(data_dir, "norm_trajectory_*.npz"))
        for f in files:
            arr = np.load(f)
            # assume normalized positions & velocities stored as e.g. 'pos_x', etc. or as arrays:
            # here we assume arrays 'positions' (T×3) & 'velocities' (T×3)
            pos = arr['positions']      # shape (T,3)
            vel = arr['velocities']     # shape (T,3)
            states = np.concatenate([pos, vel], axis=1)  # (T,6)
            T = states.shape[0]
            for i in range(T - horizon_steps):
                x = states[i : i + 1]            # current state, shape (1,6)
                y = states[i + horizon_steps]    # state one hour ahead, shape (6,)
                self.examples.append((x.astype(np.float32), y.astype(np.float32)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class PredictorFNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[64,64], output_dim=6):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 1, 6) → flatten to (batch,6)
        x = x.view(x.size(0), -1)
        return self.net(x)

def train_predictor():
    cfg = AI1_MODEL_CONFIG
    # For one-hour ahead at 10-min steps:
    horizon = 3600 // 600  # =6
    ds = OrbitSequenceDataset(NORMALIZED_GENERATED_DATA_DIR, horizon_steps=horizon)

    # split into train/val
    val_sz = int(len(ds) * VALIDATION_SPLIT_RATIO)
    train_sz = len(ds) - val_sz
    torch.manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(ds, [train_sz, val_sz])
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'])

    # model
    model = PredictorFNN(
        input_dim=6,
        hidden_dims=cfg.get('fnn_hidden_layers', [128,128]),
        output_dim=6
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt_cls = getattr(torch.optim, cfg['optimizer'].capitalize())
    optimizer = opt_cls(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= train_sz

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= val_sz

        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train_loss={train_loss:.4e}  val_loss={val_loss:.4e}")

    # save final model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/predictor.pt")
    print("Saved predictor model → models/predictor.pt")
