# Brain Tumor Segmentation in MRI using U-Net with 3D Visualization

A deep learning project for accurate segmentation and classification of brain tumors in MRI scans using a U-Net architecture. This implementation addresses the challenge of tumor regions being visually similar to surrounding healthy tissue by leveraging precise feature extraction and optimization. The model is designed for high performance on complex MRI data, with capabilities for both 2D slice-based and volumetric analysis, plus interactive 3D visualization of segmented tumor regions.

## Key Features
- **U-Net Architecture**: Encoder-decoder with skip connections for precise pixel-level segmentation.
- **High Accuracy Potential**: Optimized for Dice coefficient and accuracy exceeding 90% on well-curated datasets (e.g., BraTS).
- **MRI Preprocessing**: Intensity normalization, skull stripping support, and multi-modal handling (T1, T1ce, T2, FLAIR).
- **3D Visualization**: Volumetric rendering and overlay of tumor segmentations for clinical-style inspection.
- **Classification Support**: Can be extended to multi-class (enhancing tumor, edema, necrotic core).

## Project Structure
```
brain_tumor_unet/
├── model/
│   └── unet.py          # U-Net implementation
├── scripts/
│   ├── train.py         # Training script
│   ├── inference.py     # Inference & prediction
│   └── visualize_3d.py  # 3D visualization
├── data/                # Place BraTS or custom NIfTI files here
├── notebooks/           # Optional Jupyter demos
├── requirements.txt
└── README.md
```

## Dataset
This project is designed for public datasets such as the **Brain Tumor Segmentation (BraTS)** challenge data (available via the Medical Decathlon or official BraTS portal). 

- Download BraTS 2020/2021 from [Synapse](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863) or [Kaggle BraTS](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).
- Expected structure: NIfTI files (`.nii` or `.nii.gz`) with modalities and segmentation masks.
- For quick testing, synthetic or sample slices can be used.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --data_dir data/ --epochs 50 --batch_size 4
```

### Inference
```bash
python scripts/inference.py --input_path path/to/mri.nii.gz --model_path checkpoints/best_model.pth --output_path results/
```

### 3D Visualization
```bash
python scripts/visualize_3d.py --mri_path path/to/mri.nii.gz --mask_path path/to/prediction.nii.gz
```


## Citation / Acknowledgement
This is a demonstration project showcasing U-Net for medical image segmentation. Inspired by classic works such as Ronneberger et al. (U-Net) and the BraTS challenge.

