# CK-Dehaze

## Prerequisites

- Python 3.7+
- PyTorch >= 1.7
- NVIDIA GPU + CUDA

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MingJ-Zhang/CK-Dehaze
   cd SymUNet
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

The project is configured for the **RESIDE** dataset (e.g., RESIDE-IN).

1. Prepare your paired dataset with Hazy (LQ) and Clear (GT) images.
2. Modify the configuration files to point to your dataset paths:
   - **Training Config**: `options/train/reside/symunet_residein.yml`
   - **Testing Config**: `options/test/reside/symunet_residein.yml`

   Update the `dataroot_lq` and `dataroot_gt` fields in these YAML files.

## Training

To start training the model:

```bash
python basicsr/train.py -opt options/train/reside/symunet_residein.yml
```

**Configuration Note:**
You can adjust training parameters in the YAML file, including:
- `num_gpu`: Number of GPUs for distributed training.
- `batch_size_per_gpu`: Batch size per GPU.
- `gt_size`: Patch size for training.
- `use_kam`: Set to `true` to enable KAM convolutions (default is usually false or dependent on specific arch settings).

## Testing

To evaluate the trained model on a test set:

1. Update the `pretrain_network_g` path in `options/test/reside/symunet_residein.yml` to point to your best checkpoint (e.g., `experiments/CKUnet-residein-Dehaze/models/net_g_best.pth`).
2. Run the test script:

```bash
python basicsr/test.py -opt options/test/reside/symunet_residein.yml
```

The restored images will be saved in the `results/` directory, and metrics (PSNR, SSIM) will be calculated if ground truth is provided.

## Results visualization
[baiduYun](https://pan.baidu.com/s/1UQTvBI6mwIk7GbtBbYNPug?pwd=1tr6)

## Acknowledgements

This codebase is modified from [BasicSR](https://github.com/XPixelGroup/BasicSR), an open-source image and video restoration toolbox.
