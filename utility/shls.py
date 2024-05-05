import torch
from dipy.reconst.shm import (
    sf_to_sh,
)
import nibabel as nib

import os

from utility.utility import (
    cart2sphere,
    get_mask,
    get_odf_transformation,
)
from dipy.io.gradients import read_bvals_bvecs
from utility.utility import get_args
from dipy.core.sphere import Sphere
import numpy as np
from dipy.segment.mask import applymask
from dipy.reconst.shm import (
    real_sym_sh_basis,
)


def shls(args, output_path):
    print("==> Loading data ...")

    # data
    img = nib.load(args.img_file)
    bvals, bvecs = read_bvals_bvecs(args.bval_file, args.bvec_file)
    mask = get_mask(args)
    data = img.get_fdata()

    # filter bvecs by b values
    M = data.shape[-1]

    bix = np.where(
        (bvals >= args.bval - args.bmarg) & (bvals <= args.bval + args.bmarg)
    )[0]
    bix = bix[:M]
    bvecs_bix = bvecs[bix]

    # applying mask
    data = data[mask]

    # define harominc function space
    sphere = Sphere(xyz=bvecs_bix)
    x_obs = cart2sphere(bvecs_bix)
    theta_obs = x_obs[:, 0]
    phi_obs = x_obs[:, 1]
    _, _, n = real_sym_sh_basis(args.sh_order, phi_obs, theta_obs)
    T_n = get_odf_transformation(n)

    print("==> Running sf_to_sh ...")
    ODFtensor_shls = (
        sf_to_sh(
            data,
            sphere,
            sh_order=args.sh_order,
            basis_type="descoteaux07",
            smooth=1e-3,
        )
        @ T_n
    )

    ODFtensor_shls = np.array(ODFtensor_shls).astype(np.float32)
    ODFtensor_shls = torch.from_numpy(ODFtensor_shls)

    torch.save(ODFtensor_shls, os.path.join(output_path, "pointwise_estimates.pt"))
    print(f"==> Saved pointwise estimates to {output_path} ...")


def main(args):
    output_path = os.path.join(args.out_folder)

    shls(args, output_path)


if __name__ == "__main__":
    args = get_args()

    main(args)
