import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import nibabel as nib
from model.unet import UNet  # Adjust import if needed

class BraTSDataset(Dataset):
    """Simple dataset loader for BraTS-style NIfTI files.
    Expects folders with *t1.nii.gz, *t1ce.nii.gz, *t2.nii.gz, *flair.nii.gz, *seg.nii.gz
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_path = os.path.join(self.data_dir, case)

        # Load modalities (example names; adjust to your dataset)
        modalities = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            img = nib.load(os.path.join(case_path, f'{case}_{mod}.nii.gz')).get_fdata()
            modalities.append(img)
        
        image = np.stack(modalities, axis=0).astype(np.float32)  # (4, H, W, D)
        mask = nib.load(os.path.join(case_path, f'{case}_seg.nii.gz')).get_fdata().astype(np.float32)
        
        # Simple 2D slice extraction (middle slice for demo; improve for full training)
        mid = image.shape[-1] // 2
        image = image[:, :, :, mid]
        mask = mask[:, :, mid]
        mask = (mask > 0).astype(np.float32)  # Binary for simplicity

        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)

        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0)

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth))
    return loss.mean()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = dice_loss  # Or combine with BCE

    dataset = BraTSDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.4f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print("Training complete. Model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
