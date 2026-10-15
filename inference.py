import argparse
import torch
import numpy as np
import nibabel as nib
from model.unet import UNet
import os

def load_mri(path):
    """Load multi-modal or single MRI. Adjust for your input format."""
    img = nib.load(path).get_fdata().astype(np.float32)
    # Example: if single volume, expand channels or load multiple files
    if img.ndim == 3:
        img = img[np.newaxis, ...]  # Add channel dim
    # Normalize
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img, nib.load(path).affine

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Note: For full volume, process slice-by-slice or use 3D U-Net.
    # This is a simplified 2D-slice inference demo.
    volume, affine = load_mri(args.input_path)
    
    # Process middle slices as example
    predictions = []
    mid_start = volume.shape[-1] // 2 - 10
    mid_end = volume.shape[-1] // 2 + 10

    with torch.no_grad():
        for z in range(mid_start, mid_end):
            slice_img = volume[:, :, :, z] if volume.shape[0] == 4 else volume[0, :, :, z][None, ...]
            if slice_img.shape[0] != 4:
                # Pad or replicate for demo if channels don't match
                slice_img = np.repeat(slice_img, 4, axis=0)[:4]
            
            tensor = torch.from_numpy(slice_img).unsqueeze(0).to(device)
            pred = model(tensor)
            predictions.append(pred.squeeze().cpu().numpy())

    # Stack and threshold
    pred_volume = np.stack(predictions, axis=-1)
    pred_volume = (pred_volume > 0.5).astype(np.uint8)

    # Save as NIfTI
    os.makedirs(args.output_path, exist_ok=True)
    nii = nib.Nifti1Image(pred_volume, affine)
    nib.save(nii, os.path.join(args.output_path, 'prediction.nii.gz'))
    print(f"Prediction saved to {args.output_path}/prediction.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_path', type=str, default='results/')
    args = parser.parse_args()
    inference(args)
