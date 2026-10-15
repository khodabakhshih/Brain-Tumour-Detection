import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os

def load_volume(path):
    return nib.load(path).get_fdata()

def visualize_3d_matplotlib(mri, mask, output_dir='results/'):
    """Simple 3D scatter / surface visualization using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract tumor points (where mask > 0)
    tumor_coords = np.where(mask > 0.5)
    if len(tumor_coords[0]) == 0:
        print("No tumor voxels found.")
        return

    # Subsample for performance
    step = max(1, len(tumor_coords[0]) // 5000)
    x, y, z = tumor_coords[0][::step], tumor_coords[1][::step], tumor_coords[2][::step]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='red', marker='o', s=1, alpha=0.6, label='Tumor')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Segmented Brain Tumor')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'tumor_3d_scatter.png'), dpi=150)
    plt.show()
    print(f"Saved 3D scatter plot to {output_dir}/tumor_3d_scatter.png")

def visualize_3d_plotly(mri, mask, output_dir='results/'):
    """Interactive 3D visualization with Plotly (recommended)."""
    tumor_coords = np.where(mask > 0.5)
    if len(tumor_coords[0]) == 0:
        print("No tumor voxels found.")
        return

    step = max(1, len(tumor_coords[0]) // 8000)
    x = tumor_coords[0][::step]
    y = tumor_coords[1][::step]
    z = tumor_coords[2][::step]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.6),
        name='Tumor Region'
    )])

    fig.update_layout(
        title='Interactive 3D Brain Tumor Segmentation Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    html_path = os.path.join(output_dir, 'tumor_3d_interactive.html')
    fig.write_html(html_path)
    print(f"Interactive 3D visualization saved to {html_path}")
    fig.show()

def main(args):
    mri = load_volume(args.mri_path)
    mask = load_volume(args.mask_path)

    # Ensure shapes match
    if mri.shape != mask.shape:
        print(f"Warning: Shape mismatch MRI {mri.shape} vs Mask {mask.shape}. Using mask as-is.")

    print("Generating 3D visualizations...")
    visualize_3d_matplotlib(mri, mask, args.output_dir)
    visualize_3d_plotly(mri, mask, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Visualization of Brain Tumor Segmentation')
    parser.add_argument('--mri_path', type=str, required=True, help='Path to original MRI NIfTI')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to predicted segmentation NIfTI')
    parser.add_argument('--output_dir', type=str, default='results/')
    args = parser.parse_args()
    main(args)
