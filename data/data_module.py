import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import numpy as np
import torch
from torch.utils.data import DataLoader

from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import applymask
import nibabel as nib
from utility.utility import (
    cart2sphere,
    get_mask,
    get_odf_transformation,
    measurement_error_var_estimator,
    matern_spec_density,
    get_signal_transformation,
)

from dipy.reconst.shm import (
    real_sym_sh_basis,
    real_sym_sh_basis,
    sph_harm_ind_list,
)
from data.dataset import ObservationPoints
from dipy.data import get_sphere

import os
import pickle


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        # load bvals and bvecs
        bvals, bvecs = read_bvals_bvecs(self.args.bval_file, self.args.bvec_file)

        # get b0 and b-shel indices, and their b vectors
        b0_bval_indices = np.where(bvals < self.args.bmarg)[0]
        b_bval_indices = np.where(
            (bvals >= self.args.bval - self.args.bmarg)
            & (bvals <= self.args.bval + self.args.bmarg)
        )[0]
        bvecs_b = bvecs[b_bval_indices[: self.args.M]]  # we take the first M b vectors

        # function mapping: convert bvecs to spherical coordinates
        x_obs = cart2sphere(bvecs_b)
        theta_obs = x_obs[:, 0]
        phi_obs = x_obs[:, 1]
        Phi, _, n = real_sym_sh_basis(self.args.sh_order, phi_obs, theta_obs)

        # define harominc function space: get transformation matrices
        T_n = get_odf_transformation(n)
        T_n_inv = get_signal_transformation(n)
        sphere = get_sphere("repulsion724")
        x_grid = cart2sphere(sphere.vertices)
        theta_grid = x_grid[:, 0]
        phi_grid = x_grid[:, 1]
        B, _, _ = real_sym_sh_basis(self.args.sh_order, phi_grid, theta_grid)

        if self.args.odf_space:
            Phi_tensor = torch.from_numpy(Phi @ T_n_inv).float()
        else:
            Phi_tensor = torch.from_numpy(Phi).float()

        # load mask
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)
        mask = get_mask(self.args)

        # get coordinates
        nx, ny, nz = mask.shape
        XX, YY, ZZ = np.meshgrid(
            np.linspace(0, 1, num=nx),
            np.linspace(0, 1, num=ny),
            np.linspace(0, 1, num=nz),
            indexing="ij",
        )  # map index to location
        coords = torch.from_numpy(
            np.column_stack((XX[mask], YY[mask], ZZ[mask]))
        ).float()
        N = coords.shape[0]  # number of training points

        # collect fixed hyper params
        self.eigs_root = np.sqrt(n * (n + 1))
        self.Phi_tensor = Phi_tensor
        self.R_tensor = torch.from_numpy(
            np.diag(
                1
                / matern_spec_density(np.sqrt(n * (n + 1)), self.args.rho, self.args.nu)
            )
        ).float()
        self.R_tensor_inv = torch.from_numpy(
            np.diag(
                matern_spec_density(np.sqrt(n * (n + 1)), self.args.rho, self.args.nu)
            )
        ).float()
        self.bounding_box = torch.stack(
            [coords.min(-2).values, coords.max(-2).values], 0
        )
        self.sphere = sphere
        self.T_n_inv = T_n_inv
        self.T_n = T_n
        self.B = B
        self.N = N
        self.bvecs_b = bvecs_b
        self.bvecs = bvecs
        self.Phi = Phi
        self.b_bval_indices = b_bval_indices
        self.K = int((self.args.sh_order + 1) * (self.args.sh_order + 2) / 2)

        if stage == "fit":
            signal_path = "data/train_signal.pt"
            if os.path.exists(signal_path):
                print(f"Using precomputed signal from {signal_path}")
                signal = torch.load(signal_path)
                # take only non masked area from flattened signal array
                signal_img = torch.zeros((*mask_full.shape, signal.shape[-1]))
                signal_img[mask_full] = signal
                signal = signal_img[mask]
                
                sigma2 = 0.19826743321699167
            else:
                # load signal
                img = nib.load(self.args.img_file)
                signal_raw = img.get_fdata()  # X, Y, Z, b

                # estimate measurement error variance
                sigma2 = measurement_error_var_estimator(
                    signal_raw[..., b0_bval_indices], mask=mask
                )

                # normalize signal by b0
                signal_b0_mean = signal_raw[:, :, :, b0_bval_indices].mean(
                    axis=3
                )  # X, Y, Z

                # TODO: handle b-shells according to your data
                signal_raw = signal_raw[...,b_bval_indices]
                # signal_raw = signal_raw[:, :, :, : self.args.M]  # X, Y, Z, M

                signal_normalized = np.nan_to_num(
                    signal_raw / signal_b0_mean[..., None], posinf=0, neginf=0
                )  # X, Y, Z, M

                # apply mask to signal
                signal_normalized = applymask(signal_normalized, mask)  # X, Y, Z, M
                signal_normalized[signal_normalized >= 1] = 1  # X, Y, Z, M
                signal_normalized[signal_normalized < 0] = 0  # X, Y, Z, M

                # flatten signal
                signal = torch.from_numpy(signal_normalized[mask, :]).float()  # N, M

                torch.save(signal, signal_path)

            self.sigma2_e = sigma2
            self.dataset = ObservationPoints(coords, signal)

        if stage == "predict":
            self.dataset = ObservationPoints(coords)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=self.args.num_workers,
        )
