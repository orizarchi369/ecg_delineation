# scripts/train_unet.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
data_dir = '/home/orizarchi/projects/ecg_delineation/data/processed_cleaned'
split_dir = '/home/orizarchi/projects/ecg_delineation/data/splits'
output_dir = '/home/orizarchi/projects/ecg_delineation/outputs/models'
os.makedirs(output_dir, exist_ok=True)

# Custom Dataset
class ECGDataset(Dataset):
    def __init__(self, record_ids, data_dir):
        self.data_dir = data_dir
        self.record_ids = record_ids
        self.files = [f for f in os.listdir(data_dir) if f.split('_')[0] in record_ids]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path, allow_pickle=True)
        signal = data['signal'].astype(np.float32)
        labels = data['labels'].astype(np.long)
        return torch.from_numpy(signal).unsqueeze(0), torch.from_numpy(labels)

# 1D U-Net Model
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):  # 4 classes: bg, P, QRS, T
        super(UNet1D, self).__init__()
        self.enc1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.upconv = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.out = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))
        d1 = torch.relu(self.dec1(self.upconv(e2)))
        d1 = torch.cat([d1, e1], dim=1)  # Simple skip connection
        return self.out(d1)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_f1 = 0.0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                outputs = model(signals)
                loss = criterion(outputs.transpose(1, 2), labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_f1s.append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_unet_model.pth'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_f1s, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

# Main execution
if __name__ == '__main__':
    # Load split records
    splits = {}
    for split in ['train', 'val', 'test']:
        with open(os.path.join(split_dir, f'{split}_records.txt'), 'r') as f:
            splits[split] = [line.strip() for line in f]

    # Create datasets and loaders
    batch_size = 32
    datasets = {split: ECGDataset(splits[split], data_dir) for split in ['train', 'val', 'test']}
    loaders = {split: DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train')) for split in ['train', 'val', 'test']}

    # Initialize and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D().to(device)
    train_model(model, loaders['train'], loaders['val'])