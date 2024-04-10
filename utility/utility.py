import numpy as np
import torch
from scipy.special import legendre, gamma
from dipy.reconst.shm import real_sym_sh_basis
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.segment.mask import applymask
import argparse
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs


def measurement_error_var_estimator(data_b0: np.array, mask: np.array = None):
    """
    Estimate the variance of the measurement error for a given b0 image.

    data_b0: b0 image
    mask: mask image
    """
    if mask is not None:
        data_b0 = applymask(data_b0, mask)
    data_b0 = np.divide(data_b0, np.mean(data_b0, axis=3)[:, :, :, np.newaxis])
    sigma2_v = np.nanstd(np.where(data_b0 != 0, data_b0, np.nan), axis=3) ** 2
    sigma2_hat = np.nanmean(sigma2_v)
    return sigma2_hat


def matern_spec_density(omega: float, rho: float, nu: float):
    """
    Spectral density for a Matern covariance function. Form can be found in Dutordoir et. al 2020 supplement.

    omega: frequency
    rho: lengthscale
    nu: differentiability
    """
    term1 = (
        (2**3) * (np.pi ** (3 / 2)) * gamma(nu + (3 / 2)) * np.power(2 * nu, nu)
    ) / (gamma(nu) * np.power(rho, 2 * nu))
    term2 = np.power(
        ((2 * nu) / np.power(rho, 2)) + (4 * (np.pi**2) * np.power(omega, 2)),
        -(nu + (3 / 2)),
    )
    return term1 * term2


def ESR_design(n_pts: int, bv: int = 2000, uc: int = 1):
    """
    Generate evenly spaced points on the hemisphere for ESR.

    n_pts: number of points
    bv: b-value
    uc: unit conversion factor
    """
    theta_samps = np.pi * np.random.rand(n_pts)
    phi_samps = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta_samps, phi=phi_samps)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    vertices = hsph_updated.vertices
    values = np.ones(vertices.shape[0])
    bvecs = vertices
    bvals = (bv * values) / 1000 * uc
    bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, (0, bvals.shape[0]), 0)
    return bvecs, bvals


def get_odf_transformation(n: int):
    """
    Get the transformation matrix for the ODF signal.

    n: spherical harmonic order
    """
    T = np.zeros((len(n), len(n)))
    for i in range(T.shape[0]):
        P_n = legendre(n[i])
        T[i, i] = P_n(0)
    return T


def get_signal_transformation(n: int):
    """
    Get the transformation matrix for the signal.

    n: spherical harmonic order
    """
    Tinv = np.zeros((len(n), len(n)))
    for i in range(Tinv.shape[0]):
        P_n = legendre(n[i])
        Tinv[i, i] = 1.0 / P_n(0)
    return Tinv


def cart2sphere(x):
    """
    Convert cartesian to spherical coordinates.

    x: cartesian coordinates
    """
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    theta = np.arctan2(x[:, 1], x[:, 0])
    phi = np.arccos(x[:, 2] / r)
    return np.column_stack([theta, phi])


def sphere2cart(x):
    """
    Convert spherical coordinates to cartesian.

    x: spherical coordinates
    """
    theta = x[:, 0]
    phi = x[:, 1]
    xx = np.sin(phi) * np.cos(theta)
    yy = np.sin(phi) * np.sin(theta)
    zz = np.cos(phi)
    return np.column_stack([xx, yy, zz])


def S2hemisphere(x):
    """ """
    x_copy = np.copy(x)
    x_polar = cart2sphere(x_copy)
    ix = np.argwhere(x_polar[:, 1] > np.pi / 2).ravel()
    x_copy[ix, :] = -1 * x_copy[ix, :]
    return x_copy


def save_nif(args: dict, data: np.array, path: str):
    """
    Save data to a nifti file.

    args: arguments
    data: (X, Y, Z, D) data to save
    path: path to save
    """
    img = nib.load(args.img_file)
    nif = nib.Nifti1Image(data, img.affine)
    nib.save(nif, path)
    print(f"Saved img to {path}")


def get_phi_r_tensors(args, ODFSPACE=True):
    """
    Get Phi and R tensors.

    args: arguments
    ODFSPACE: flag to use ODF space

    returns:
        Phi_tensor: torch.Tensor (K x M)
        R_tensor: (M x M)
    """
    # load bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(args.bval_file, args.bvec_file)

    # get b0 and b-shel indices, and their b vectors
    b_bval_indices = np.where(
        (bvals >= args.bval - args.bmarg) & (bvals <= args.bval + args.bmarg)
    )[0]
    bvecs_b = bvecs[b_bval_indices[: args.M]]  # we take the first M b vectors

    # function mapping: convert bvecs to spherical coordinates
    x_obs = cart2sphere(bvecs_b)
    theta_obs = x_obs[:, 0]
    phi_obs = x_obs[:, 1]
    Phi, _, n = real_sym_sh_basis(args.sh_order, phi_obs, theta_obs)

    # define harominc function space: get transformation matrices
    T_n_inv = get_signal_transformation(n)

    if ODFSPACE:
        Phi_tensor = torch.from_numpy(Phi @ T_n_inv).float()
    else:
        Phi_tensor = torch.from_numpy(Phi).float()

    R_tensor = torch.from_numpy(
        np.diag(1 / matern_spec_density(np.sqrt(n * (n + 1)), args.rho, args.nu))
    ).float()

    return Phi_tensor, R_tensor


def get_mask(args: dict):
    """
    Loads the mask and applies custom modifications to it

    args: arguments
    """
    mask = nib.load(args.mask_file).get_fdata().astype(bool)
    # 1
    # cerebellum
    # mask[:76] = False
    # mask[216:] = False
    # cerebellum 1 slice
    # mask[:168] = False
    # mask[169:] = False
    # 2
    # cerebellum
    # mask[:, 126:] = False
    # cerebellum 1 slice
    # mask[:, 117:] = False
    # mask[:, :40] = False
    # mask[:, :74] = False
    # mask[:, 88:] = False
    # 3
    # cerebellum
    # mask[:, :, 102:] = False
    # cerebellum 1 slice
    # mask[:, :, 92:] = False
    # mask[:, :, :21] = False
    # mask[:, :, :67] = False
    # mask[:, :, 85:] = False

    return mask


def get_args(cmd: bool = True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="NODF Estimation."
    )

    parser.add_argument(
        "--out_folder",
        action="store",
        default="output",
        type=str,
        help="Path to folder to store output.",
    )

    parser.add_argument(
        "--img_file",
        action="store",
        default="data/subjects/processedDWI_session1_subset01/dwi_all.nii.gz",
        type=str,
        help="Nifti file for diffusion image (nx X ny X nz x M)",
    )

    parser.add_argument(
        "--mask_file",
        action="store",
        default="data/subjects/processedDWI_session1_subset01/flipped_b0_brain_mask.nii.gz",
        type=str,
        help="Nifti file for mask image (nx X ny X nz)",
    )

    parser.add_argument(
        "--bval_file",
        action="store",
        default="data/subjects/processedDWI_session1_subset01/flip_x.bval",
        type=str,
        help="Text file b-values.",
    )

    parser.add_argument(
        "--bvec_file",
        action="store",
        default="data/subjects/processedDWI_session1_subset01/flip_x.bvec",
        type=str,
        help="Text file b-vectors.",
    )

    parser.add_argument(
        "--ckpt_path",
        action="store",
        type=str,
        help="Path to checkpoint file.",
        default=None,
    )

    parser.add_argument(
        "--predictions_path",
        action="store",
        type=str,
        help="Path to ODF coefficients file.",
        default=None,
    )

    parser.add_argument(
        "--experiment_name",
        action="store",
        type=str,
        help="Name of the experiment.",
        default="hashenc",
    )

    parser.add_argument(
        "--inr",
        action="store",
        type=str,
        help="Type of the INR - wire or siren or relu",
        choices=["siren", "wire", "relu"],
        default="siren",
    )

    parser.add_argument(
        "--bval",
        action="store",
        default=1000,
        type=float,
        help="B-value to build field.",
    )

    parser.add_argument(
        "--device",
        action="store",
        type=str,
        help="Device.",
        default="cuda",
    )

    parser.add_argument(
        "--sh_order",
        action="store",
        default=8,
        type=int,
        help="Order of spherical harmonic basis",
    )

    parser.add_argument(
        "--bmarg",
        action="store",
        default=20,
        type=int,
        help="+= bmarg considered same b-value.",
    )

    parser.add_argument(
        "--rho",
        action="store",
        default=0.5,
        type=float,
        help="Length-scale parameter for Matern Prior.",
    )

    parser.add_argument(
        "--nu",
        action="store",
        default=1.5,
        type=float,
        help="Smoothness parameter for Matern Prior.",
    )

    parser.add_argument(
        "--num_epochs",
        action="store",
        default=50000,
        type=int,
        help="Number of trainging epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        action="store",
        default=0.0001,
        type=float,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--r", action="store", default=64, type=int, help="Rank of spatial basis."
    )

    parser.add_argument(
        "--depth", action="store", default=2, type=int, help="Number of hidden layers."
    )

    parser.add_argument(
        "--train_prop",
        action="store",
        default=0.8,
        type=float,
        help="Proportion of voxels to be used in training for each iteration of hyper-parameter optimization.",
    )

    parser.add_argument(
        "--batch_frac",
        action="store",
        default=64,
        type=int,
        help="Fraction of total training voxels to be used in each batch.",
    )

    parser.add_argument(
        "--batch_size",
        action="store",
        default=65536,
        type=int,
        help="Batch size.",
    )

    parser.add_argument(
        "--Nexperiments",
        action="store",
        default=20,
        type=int,
        help="Number of experiments for hyper-parameter optimization.",
    )

    parser.add_argument(
        "--calib_prop",
        action="store",
        default=0.1,
        type=float,
        help="Proportion of voxels to be used in posterior calibration.",
    )

    parser.add_argument(
        "--lambda_c", help="Prior regularization strength.", type=float, default=3.76e-7
    )

    parser.add_argument(
        "--sigma2_mu",
        help="Variance for isotropic harmonic.",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--sigma2_w", help="Variance parameter for GP prior.", type=float, default=0.5
    )

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--deconvolve", action="store_true")

    parser.add_argument("--enable_schedulers", action="store_true")

    parser.add_argument("--simulation", action="store_true")

    parser.add_argument(
        "--omega0", help="WIRE Gaussian parameter.", type=float, default=30.0
    )

    parser.add_argument(
        "--omega0_hidden",
        help="WIRE Gaussian parameter for hidden layers.",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--sigma0", help="WIRE Sine frequency parameter.", type=float, default=5.0
    )

    parser.add_argument("--skip_conn", action="store_true")

    parser.add_argument("--batchnorm", action="store_true")

    parser.add_argument(
        "--num_workers",
        action="store",
        default=12,
        type=int,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--n_levels",
        action="store",
        default=14,
        type=int,
        help="Number of resolution levels for hash encoding.",
    )

    parser.add_argument(
        "--n_features_per_level",
        action="store",
        default=2,
        type=int,
        help="Number of features per embeddings vector.",
    )

    parser.add_argument(
        "--log2_hashmap_size",
        action="store",
        default=20,
        type=int,
        help="Hash map size 2**log2_hashmap_size.",
    )

    parser.add_argument(
        "--base_resolution",
        action="store",
        default=6,
        type=int,
        help="Base resolution for hash encoding.",
    )

    parser.add_argument(
        "--per_level_scale",
        action="store",
        default=1.39,
        type=int,
        help="Per level scal of resolution.",
    )

    parser.add_argument("--weight_decay", help="Weight decay.", type=float, default=0.0)

    parser.add_argument(
        "--region",
        action="store",
        default=None,
        type=str,
        help="Select region: x_from-x_to,y_from-y_to,z_from-z_to",
    )

    parser.add_argument(
        "--M",
        action="store",
        default=70,
        type=int,
        help="Number of gradient directions.",
    )

    parser.add_argument("--use_baseline", action="store_true")

    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args(
            [
                "--out_folder",
                "output",
                "--img_file",
                "data/subjects/processedDWI_session1_subset01/dwi_all.nii.gz",
                "--mask_file",
                "data/subjects/processedDWI_session1_subset01/flipped_b0_brain_mask.nii.gz",
                "--bval_file",
                "data/subjects/processedDWI_session1_subset01/flip_x.bval",
                "--bvec_file",
                "data/subjects/processedDWI_session1_subset01/flip_x.bvec",
            ]
        )

    return args
