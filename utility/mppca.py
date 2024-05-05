from utility.utility import get_args
from data.data_module import DataModule
import numpy as np
import torch

import nibabel as nib
import os
from callback import *
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
    img = nib.load(args.img_file)
    Ynorm = data_module.Ynorm

    print("==> running mppca ...")
    Y_denoisy = mppca(Ynorm.detach().numpy(), patch_radius=2)

    try:
        result = Y_denoisy.detach().numpy()
    except:
        result = Y_denoisy

    Ynorm_nib = nib.Nifti1Image(result, img.affine)
    path = os.path.join(output_path, f"Y_denoisy.nii.gz")
    nib.save(Ynorm_nib, path)


if __name__ == "__main__":
    args = get_args()

    main(args)
