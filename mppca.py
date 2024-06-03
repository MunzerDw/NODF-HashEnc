from utility.utility import get_args
from data_module import DataModule
import numpy as np
import torch

import nibabel as nib
import os
import pytorch_lightning as pl
from dipy.denoise.localpca import mppca


def main(args):
    # fix the seed
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(0, workers=True)

    # get on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=====> Running on device: {device}")

    # get output path
    output_path = os.path.join(args.out_folder)
    os.makedirs(output_path, exist_ok=True)
    print(f"=====> Output path: {output_path}")

    print("==> initializing data ...")
    data_module = DataModule(args)
    data_module.setup("fit")
    signal_flat = data_module.dataset.signal
    mask_full = nib.load(args.mask_file).get_fdata().astype(bool)  # X x Y x Z
    signal = torch.zeros((*mask_full.shape, signal_flat.shape[-1]))  # X x Y x Z x M
    signal[mask_full] = signal_flat  # X x Y x Z x M

    print("==> running mppca ...")
    signal_denoised = mppca(signal.detach().numpy(), patch_radius=2)

    try:
        signal_denoised = signal_denoised.detach().numpy()
    except:
        signal_denoised = signal_denoised

    img = nib.load(args.img_file)
    signal_denoised_img = nib.Nifti1Image(signal_denoised, img.affine)
    path = os.path.join(output_path, f"signal_denoised.nii.gz")
    nib.save(signal_denoised_img, path)


if __name__ == "__main__":
    args = get_args()

    main(args)
