import pickle
import nibabel as nib
import torch
from data_module import DataModule
from utility.utility import cart2sphere, get_args, get_mask
import numpy as np
from models.nodf import NODF
import os
from dipy.reconst.csdeconv import odf_sh_to_sharp
from dipy.reconst.shm import sf_to_sh
from dipy.data import get_sphere
from dipy.reconst.shm import (
    real_sym_sh_basis,
    sph_harm_ind_list,
)


def main(args):
    # load the img and mask
    img = nib.load(args.img_file)
    mask = get_mask(args)
    mask_full = nib.load(args.mask_file).get_fdata().astype(bool)

    # create the output folder
    output_path = os.path.join(args.out_folder, args.experiment_name, "visualization")
    os.makedirs(output_path, exist_ok=True)
    if not args.predictions_path:
        args.predictions_path = os.path.join(
            args.out_folder, args.experiment_name, "prediction/pointwise_estimates.pt"
        )

    # load the predictions
    chat_flat = torch.load(
        args.predictions_path, map_location=torch.device("cpu")
    ).numpy()
    # get only masked region of interest
    chat = np.zeros((*mask_full.shape, chat_flat.shape[-1]))
    chat[mask_full] = chat_flat
    chat_flat = chat[mask]

    # get spherical harmonics basis function evaluation matrix
    sphere = get_sphere("repulsion724")
    x_grid = cart2sphere(sphere.vertices)
    theta_grid = x_grid[:, 0]
    phi_grid = x_grid[:, 1]
    B, _, _ = real_sym_sh_basis(args.sh_order, phi_grid, theta_grid)

    print("==> Doing Spherical Deconvolution ...")
    # sharpen ODF coefficients
    chat_flat_deconvolved = odf_sh_to_sharp(
        chat_flat,
        sphere,
        basis="descoteaux07",
        ratio=0.3,  # should estimate this from the data or make it configurable
        sh_order=args.sh_order,
        lambda_=1.0,
        tau=0.1,
        r2_term=False,
    )
    # get ODF evaluations on the sphere
    odfs_flat = chat_flat_deconvolved @ B.T
    # get SH expansion coefficients of the ODf evaluations in the "tournier07" basis.
    # change coordiante system for visualizing in mrtrix
    chat_flat_deconvolved_tourn = sf_to_sh(
        odfs_flat,
        sphere,
        sh_order=8,
        basis_type="tournier07",
    )
    # normalize coefficients
    chat_flat_deconvolved_tourn = (
        chat_flat_deconvolved_tourn
        / np.linalg.norm(chat_flat_deconvolved_tourn, axis=-1)[..., None]
    )

    print("==> Saving file ...")
    # save ODF file
    chat_deconvolved_tourn = np.zeros((*mask_full.shape, chat_flat.shape[-1]))
    chat_deconvolved_tourn[mask, :] = chat_flat_deconvolved_tourn
    chat_deconvolved_tourn = chat_deconvolved_tourn.astype(np.float32)
    chat_deconvolved_tourn_img = nib.Nifti1Image(chat_deconvolved_tourn, img.affine)
    nib.save(
        chat_deconvolved_tourn_img,
        os.path.join(
            args.out_folder,
            args.experiment_name,
            "visualization",
            "odfs_tournier07.nii.gz",
        ),
    )
    print(
        f"Saved odfs_tournier07.nii.gz to ",
        os.path.join(
            args.out_folder,
            args.experiment_name,
            "visualization",
            "odfs_tournier07.nii.gz",
        ),
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
