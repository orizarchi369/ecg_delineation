# scripts/train_unet_improved.py

import os, csv, argparse, math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description='Train 1D U-Net for ECG segmentation on Lead II (improved)')
parser.add_argument('--data_dir', type=str, default='/content/ecg_delineation/data/processed_cleaned')
parser.add_argument('--split_dir', type=str, default='/content/ecg_delineation/data/splits')
parser.add_argument('--output_dir', type=str, default='/content/drive/My Drive/ecg_project/models')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)

# LR scheduler and Early stopping
parser.add_argument('--plateau_patience', type=int, default=5, help='epochs with no val loss improvement before LR reduces')
parser.add_argument('--plateau_factor', type=float, default=0.5, help='LR reduce factor on plateau')
parser.add_argument('--early_stop_patience', type=int, default=20, help='epochs with no val F1 improvement to stop')
parser.add_argument('--min_delta', type=float, default=1e-4, help='minimum improvement to reset patience (F1)')

# Live plotting
parser.add_argument('--live_plot', action='store_true', help='update and save curves every plot_interval epochs')
parser.add_argument('--plot_interval', type=int, default=5)

# Model
parser.add_argument('--out_channels', type=int, default=4)
parser.add_argument('--use_batchnorm', action='store_true', help='add BatchNorm1d after convs')
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

# ----------------------------
# Repro
# ----------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ----------------------------
# Paths
# ----------------------------
os.makedirs(args.output_dir, exist_ok=True)
history_csv = os.path.join(args.output_dir, 'training_history.csv')

# ----------------------------
# Dataset
# ----------------------------
class ECGDataset(Dataset):
    def __init__(self, record_ids, data_dir):
        self.data_dir = data_dir
        self.record_ids = set(record_ids)
        self.files = [f for f in os.listdir(data_dir)
                      if '_' in f and f.split('_')[0] in self.record_ids and 'ii' in f and f.endswith('.npz')]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        signal = data['signal'].astype(np.float32)     # [L]
        labels = data['labels'].astype(np.int64)       # [L]
        return torch.from_numpy(signal).unsqueeze(0), torch.from_numpy(labels)

# ----------------------------
# Model (same topology, optional BN/Dropout)
# ----------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, p_drop=0.0):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn: layers += [nn.BatchNorm1d(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        if p_drop > 0: layers += [nn.Dropout(p_drop)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, use_bn=False, p_drop=0.0):
        super().__init__()
        self.enc1 = ConvBlock1D(in_channels, 32, use_bn, p_drop)
        self.enc2 = ConvBlock1D(32, 64, use_bn, p_drop)
        self.enc3 = ConvBlock1D(64, 128, use_bn, p_drop)
        self.pool = nn.MaxPool1d(2, 2)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(128, 64, use_bn, p_drop)  # 64 up + 64 skip
        self.dec2 = ConvBlock1D(64, 32, use_bn, p_drop)   # 32 up + 32 skip (after Conv reduction)
        self.out = nn.Conv1d(32, out_channels, kernel_size=1)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)                # [B,32,L]
        e2 = self.enc2(self.pool(e1))    # [B,64,L/2]
        e3 = self.enc3(self.pool(e2))    # [B,128,L/4]
        d1 = self.upconv1(e3)            # [B,64,L/2]
        d1 = torch.cat([d1, F.interpolate(e2, size=d1.size(2), mode='linear', align_corners=False)], dim=1)
        d1 = self.dec1(d1)               # [B,64,L/2]
        d2 = self.upconv2(d1)            # [B,32,L]
        d2 = torch.cat([d2, F.interpolate(e1, size=d2.size(2), mode='linear', align_corners=False)], dim=1)
        d2 = self.dec2(d2)               # [B,32,L]
        return self.out(d2)              # [B,C,L]

# ----------------------------
# Utils
# ----------------------------
def current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch','train_loss','val_loss','val_f1','lr'])

def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def plot_curves(train_losses, val_losses, val_f1s, out_path):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# ----------------------------
# Train/Eval
# ----------------------------
def train_model(model, loaders, num_epochs, lr, output_dir,
                plateau_patience, plateau_factor,
                early_stop_patience, min_delta, live_plot, plot_interval):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=plateau_factor, patience=plateau_patience
    )

    best_f1 = -1.0
    es_counter = 0
    train_losses, val_losses, val_f1s, lrs = [], [], [], []

    ensure_csv(history_csv)

    for epoch in range(1, num_epochs+1):
        # ---- Train
        model.train()
        running = 0.0
        pbar = tqdm(loaders['train'], desc=f'Epoch {epoch}/{num_epochs}')
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(signals)                       # [B,C,L]
                loss = criterion(outputs, labels)             # labels [B,L]
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr(optimizer):.2e}')
        train_loss = running / len(loaders['train'])
        train_losses.append(train_loss)

        # ---- Val
        model.eval()
        vloss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for signals, labels in loaders['val']:
                signals, labels = signals.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                vloss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy().ravel())
                all_labels.extend(labels.detach().cpu().numpy().ravel())
        val_loss = vloss / len(loaders['val'])
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        # ---- Scheduler on val_loss
        scheduler.step(val_loss)
        lrs.append(current_lr(optimizer))

        # ---- Logging
        print(f'Epoch {epoch}/{num_epochs} | LR: {current_lr(optimizer):.2e} | '
              f'Train {train_loss:.4f} | Val {val_loss:.4f} | F1 {val_f1:.4f}')
        append_csv(history_csv, [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_f1:.6f}', f'{current_lr(optimizer):.8f}'])

        # ---- Checkpoints
        torch.save(model.state_dict(), os.path.join(output_dir, 'last_unet_model.pth'))
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_unet_model.pth'))
        else:
            es_counter += 1

        # ---- Live plot
        if live_plot and (epoch % plot_interval == 0 or epoch == num_epochs):
            plot_curves(train_losses, val_losses, val_f1s,
                        os.path.join(output_dir, 'training_curves_live.png'))

        # ---- Early stopping (based on F1)
        if es_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch} (no F1 improvement for {early_stop_patience} epochs).')
            break

    # Final plot
    plot_curves(train_losses, val_losses, val_f1s, os.path.join(output_dir, 'training_curves.png'))

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Load splits
    splits = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(args.split_dir, f'{split}_records.txt'), 'r') as f:
            splits[split] = [line.strip() for line in f]

    # Datasets/Loaders
    datasets = {split: ECGDataset(splits[split], args.data_dir) for split in ['train','val','test']}
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
        'val':   DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
        'test':  DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
    }

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D(in_channels=1, out_channels=args.out_channels,
                   use_bn=args.use_batchnorm, p_drop=args.dropout).to(device)

    # Train
    train_model(
        model, loaders,
        num_epochs=args.num_epochs, lr=args.lr, output_dir=args.output_dir,
        plateau_patience=args.plateau_patience, plateau_factor=args.plateau_factor,
        early_stop_patience=args.early_stop_patience, min_delta=args.min_delta,
        live_plot=args.live_plot, plot_interval=args.plot_interval
    )
