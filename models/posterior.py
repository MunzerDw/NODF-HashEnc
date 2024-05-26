import os
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl
from data.data_module import DataModule
from models.nodf import NODF
import copy
import nibabel as nib
from tqdm import tqdm
from utility.utility import (
    get_mask,
    measurement_error_var_estimator,
)
from dipy.io.gradients import read_bvals_bvecs

class FVRF(torch.nn.Module):
    """
    Function-valued random field.
    """

    def __init__(self, args):
        super().__init__()
        
        data_module = DataModule(args)
        data_module.setup("fit")
        model = NODF.load_from_checkpoint(args.ckpt_path).cpu()
        args.sigma2_w = model.args.sigma2_w
        
        Phi_tensor = model.Phi_tensor
        R_tensor = model.R_tensor
        coords = data_module.dataset.coords
        signal = data_module.dataset.signal
        sigma2_w = args.sigma2_w
        
        # Measurement error variance
        if args.sigma2_e:
            sigma2_e = args.sigma2_e
        else:
            print(f"Estimating the variance of the measurement error for a given b0 image.")
            # load bvals and bvecs
            bvals, _ = read_bvals_bvecs(args.bval_file, args.bvec_file)

            # get b0 and b-shel indices, and their b vectors
            b0_bval_indices = np.where(bvals < args.bmarg)[0]
            # load signal
            img = nib.load(args.img_file)
            signal_raw = img.get_fdata()  # X, Y, Z, b
            
            # to prevent division by very small numbers
            signal_raw[signal_raw <= 1e-2] = 1e-2

            # estimate measurement error variance
            mask = get_mask(args)
            sigma2_e = measurement_error_var_estimator(
                signal_raw[..., b0_bval_indices], mask=mask
            )
            print('Variance of the measurement error:', sigma2_e)
        
        self.args = args
        self.Phi_tensor = Phi_tensor
        self.R_tensor = R_tensor
        self.sigma2_w = sigma2_w
        self.sigma2_e = sigma2_e
        self.model = model
        self.coords = coords
        self.signal = signal
        self.basis_model = model.get_basis()
        self.data_module = data_module
        self.K = Phi_tensor.shape[1]


        # calculate model inferences
        self.chat = self._get_chat()
        self.X = self._get_X()
        self.r = self.X.shape[0]

        # calculate W posterior mean and covariance
        vec_W_post_mean, vec_W_post_cov = self._get_W_post()
        
        self.vec_W_post_mean = vec_W_post_mean
        self.vec_W_post_cov = vec_W_post_cov

    def sample_posterior_pointwise(self, mask, npost_samps=250):
        """
        Sample the point-wise posteriors of the ODF coefficients
        
        mask: mask of region of interest, torch.Tensor (X x Y x Z)
        npost_samps: number of samles to generate, int
        
        returns:
            post_samples_chat: torch.Tensor (npost_samps, N, K), 
                generate ODF coefficients from posterior field for the region of interest
        """
        
        # get coords region of interest
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)  # X x Y x Z
        coords_3d = torch.zeros((*mask_full.shape, self.coords.shape[-1]))
        coords_3d[mask_full] = self.coords
        coords = coords_3d[mask]
        # get X region of interest
        Xi_evals_3d = torch.zeros((*mask_full.shape, self.X.shape[-1]))
        Xi_evals_3d[mask_full] = self.X
        Xi_evals = Xi_evals_3d[mask].T
        # get chat region of interest
        chat_3d = torch.zeros((*mask_full.shape, self.chat.shape[-1]))
        chat_3d[mask_full] = self.chat
        chat = chat_3d[mask]
        
        Nv = coords.shape[0]
        device = torch.device(coords.device)
        
        vec_W_post_mean = self.vec_W_post_mean.to(device)
        vec_W_post_cov = self.vec_W_post_cov.to(device)
        
        Ik = torch.eye(self.K - 1).to(device)
        if self.model.args.use_baseline:
            I_rig = torch.eye(vec_W_post_cov.shape[0]).to(device) * 1e-10
            vec_W_post_cov = vec_W_post_cov + I_rig

        post_mean_c_lst = []
        post_cov_c_lst = []
        # get ODF coefficient means and covariances for all voxels
        for iv in tqdm(range(Nv)): # TODO: optimize this calculation
            XiV_I = torch.kron(Xi_evals[:, iv : iv + 1].T, Ik)
            post_mean_c = torch.squeeze(XiV_I @ vec_W_post_mean)
            post_cov_c = XiV_I @ vec_W_post_cov @ XiV_I.T
            post_mean_c_lst.append(post_mean_c.detach())
            post_cov_c_lst.append(post_cov_c.detach())

        Post_mean_c = torch.stack(post_mean_c_lst, 0)
        Post_cov_c = torch.stack(post_cov_c_lst, 0)

        C_mu_tensor = chat[:, 0:1]

        Post_mean_coefs = torch.column_stack((C_mu_tensor, Post_mean_c))
        Post_cov_mats = torch.zeros((Post_mean_coefs.shape[0], self.K, self.K)).to(
            device
        )
        
        Post_cov_mats[:, 0, 0] = self.args.sigma2_mu * torch.ones(
            Post_cov_mats.shape[0]
        )
        Post_cov_mats[:, 1 : self.K, 1 : self.K] = Post_cov_c
        
        posterior_field = MultivariateNormal(Post_mean_coefs, Post_cov_mats)
        return posterior_field.sample((npost_samps,))
    
    def sample_posterior_W(self, mask, npost_samps=250):
        """
        Sample the posterior of W and evaluate the ODF coefficients
        
        mask: mask of region of interest, torch.Tensor (X x Y x Z)
        npost_samps: number of samles to generate, int
        
        returns:
            post_samples_chat: torch.Tensor (npost_samps, N, K), 
                generate ODF coefficients from posterior field for the region of interest
        """
        
        # get coords region of interest
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)  # X x Y x Z
        coords_3d = torch.zeros((*mask_full.shape, self.coords.shape[-1]))
        coords_3d[mask_full] = self.coords
        coords = coords_3d[mask]
        # get X region of interest
        Xi_evals_3d = torch.zeros((*mask_full.shape, self.X.shape[-1]))
        Xi_evals_3d[mask_full] = self.X
        Xi_evals = Xi_evals_3d[mask].T # (r, N)

        # get chat region of interest
        chat_3d = torch.zeros((*mask_full.shape, self.chat.shape[-1]))
        chat_3d[mask_full] = self.chat
        chat = chat_3d[mask]
        
        Nv = coords.shape[0]
        device = torch.device(coords.device)
        
        vec_W_post_mean = self.vec_W_post_mean.to(device)
        vec_W_post_cov = self.vec_W_post_cov.to(device)
        
        Ik = torch.eye(self.K - 1).to(device) # (K - 1, K - 1)
        if self.model.args.use_baseline:
            I_rig = torch.eye(vec_W_post_cov.shape[0]).to(device) * 1e-9
            vec_W_post_cov = vec_W_post_cov + I_rig

        posterior_field = MultivariateNormal(vec_W_post_mean.squeeze(-1), vec_W_post_cov)
        samples = posterior_field.sample((npost_samps,)).T # (r * (K - 1), S)

        # post_samples_lst = []
        # for iv in tqdm(range(Nv)):  # TODO: optimize this calculation
        #     # get from W to ODF coefficients using the network basis evaluations
        #     XiV = Xi_evals[:, iv : iv + 1].T # (1, r)
        #     XiV_I = torch.kron(XiV, Ik) # (K - 1, r * (K - 1))
        #     post_samples = XiV_I @ samples # (K - 1, S)
        #     post_samples = torch.squeeze(post_samples) # (K - 1, S)
        #     post_samples_lst.append(post_samples.detach())
        # Post_samples = torch.stack(post_samples_lst, 0) # (N, K - 1, S)
        
        XV_I = torch.kron(Xi_evals.T, Ik) # (S, r * (K - 1))
        Post_samples = XV_I @ samples # (N * (K - 1), S)
        Post_samples = Post_samples.reshape(Xi_evals.shape[1], Ik.shape[0], Post_samples.shape[-1]) # (N, (K - 1), S)
        
        Post_samples = torch.swapaxes(Post_samples, 1, 2) # (N, S, K - 1)

        C_mu_tensor_means = chat[:, 0:1]
        
        # option 1: select mean C_mu_tensor
        # C_mu_tensor_samples = C_mu_tensor_means.unsqueeze(-1).repeat(1, npost_samps, 1)
        
        # option 2: sample C_mu_tensor from mean (C_mu_tensor_means) and covariance sigma2_mu
        C_mu_tensor_covs = self.args.sigma2_mu * torch.ones(
            (C_mu_tensor_means.shape[0], 1, 1)
        )
        C_mu_tensor_posterior_field = MultivariateNormal(C_mu_tensor_means, C_mu_tensor_covs)
        C_mu_tensor_samples = C_mu_tensor_posterior_field.sample((npost_samps,))
        C_mu_tensor_samples = torch.swapaxes(C_mu_tensor_samples, 0, 1) # (N X S X 1)

        post_samples_chat = torch.cat((C_mu_tensor_samples, Post_samples), dim=-1)
        post_samples_chat = torch.swapaxes(post_samples_chat, 0, 1)

        return post_samples_chat
    
    def evaluate_posterior_W_mean(self, mask):
        """
        Use the posterior W mean to evaluate the ODF coefficients
        
        mask: mask of region of interest, torch.Tensor (X x Y x Z)
        
        returns:
            post_samples_chat: torch.Tensor (npost_samps, N, K), 
                generate ODF coefficients from posterior field for the region of interest
        """
        
        # get coords region of interest
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)  # X x Y x Z
        coords_3d = torch.zeros((*mask_full.shape, self.coords.shape[-1]))
        coords_3d[mask_full] = self.coords
        coords = coords_3d[mask]
        # get X region of interest
        Xi_evals_3d = torch.zeros((*mask_full.shape, self.X.shape[-1]))
        Xi_evals_3d[mask_full] = self.X
        Xi_evals = Xi_evals_3d[mask].T # (r, N)

        # get chat region of interest
        chat_3d = torch.zeros((*mask_full.shape, self.chat.shape[-1]))
        chat_3d[mask_full] = self.chat
        chat = chat_3d[mask]
        
        Nv = coords.shape[0]
        device = torch.device(coords.device)
        
        vec_W_post_mean = self.vec_W_post_mean.to(device)
        vec_W_post_cov = self.vec_W_post_cov.to(device)
        
        Ik = torch.eye(self.K - 1).to(device) # (K - 1, K - 1)
        if self.model.args.use_baseline:
            I_rig = torch.eye(vec_W_post_cov.shape[0]).to(device) * 1e-9
            vec_W_post_cov = vec_W_post_cov + I_rig

        samples = vec_W_post_mean
        
        XV_I = torch.kron(Xi_evals.T, Ik) # (S, r * (K - 1))
        Post_samples = XV_I @ samples # (N * (K - 1), S)
        Post_samples = Post_samples.reshape(Xi_evals.shape[1], Ik.shape[0], Post_samples.shape[-1]) # (N, (K - 1), S)
        
        Post_samples = torch.swapaxes(Post_samples, 1, 2) # (N, S, K - 1)

        C_mu_tensor_means = chat[:, 0:1]
        
        # select mean C_mu_tensor
        C_mu_tensor_samples = C_mu_tensor_means.unsqueeze(-1).repeat(1, 1, 1)

        post_samples_chat = torch.cat((C_mu_tensor_samples, Post_samples), dim=-1)
        post_samples_chat = torch.swapaxes(post_samples_chat, 0, 1)

        return post_samples_chat

    def log_prob(self):
        """
        Multivatiate Gaussian likelihood over discretization
        """
        raise NotImplementedError
        
    def _get_W_post(self):
        """
        Get W posterior mean and covariance
        
        returns:
            vec_W_post_mean: torch.Tensor () W posterior mean
            vec_W_post_cov: torch.Tensor () W posterior covariance
        """
        
        mean_path = os.path.join(
            self.args.out_folder,
            self.args.experiment_name,
            "prediction",
            "vec_W_post_mean.pt",
        )
        cov_path = os.path.join(
            self.args.out_folder,
            self.args.experiment_name,
            "prediction",
            "vec_W_post_cov.pt",
        )
        if os.path.exists(mean_path) and os.path.exists(cov_path):
            print(f"Using saved vec_W_post_mean.pt and vec_W_post_cov.pt from {mean_path} and {cov_path}")
            vec_W_post_mean = torch.load(mean_path)
            vec_W_post_cov = torch.load(cov_path)
            return vec_W_post_mean, vec_W_post_cov
        
        chat = self.chat # (N, K)
        Xi_v = self.X.T # (r, N)
        
        C_mu_tensor = chat[:, 0:1]
        signal_centered = self.signal.T - self.Phi_tensor[:, 0:1] @ C_mu_tensor.T # (M, N)

        K = self.Phi_tensor.shape[1]
        r = Xi_v.shape[0]

        Lambda = (1 / self.sigma2_e) * (
            (self.sigma2_e / self.sigma2_w)
            * torch.kron(torch.eye(r), self.R_tensor[1:, 1:])
            + torch.kron(Xi_v @ Xi_v.T, self.Phi_tensor[:, 1:].T @ self.Phi_tensor[:, 1:K])
        ) # (r * (K - 1), r * (K - 1))
        Lambda_inv = torch.linalg.inv(Lambda) # (r * (K - 1), r * (K - 1))

        pyz_prod = (self.Phi_tensor[:, 1:K].T @ signal_centered @ Xi_v.T).T.reshape(-1, 1) # (r * (K - 1), 1)
        post_mean_w = (1 / self.sigma2_e) * Lambda_inv @ pyz_prod # (r * (K - 1), 1)
        
        # save posterior
        torch.save(
            post_mean_w.cpu().detach(),
            mean_path,
        )
        print(
            f"Saved vec_W_post_mean.pt to ",
            mean_path,
        )
        torch.save(
            Lambda_inv.cpu().detach(),
            cov_path,
        )
        print(
            f"Saved vec_W_post_cov.pt to ",
            cov_path,
        )
        
        return post_mean_w, Lambda_inv
        
    def _get_chat(self):
        """
        Get the ODF coefficients for the model
        
        returns:
            chat: torch.Tensor (N, K) predicted ODF coefficients
        """
        
        chat_path = os.path.join(
            self.args.out_folder,
            self.args.experiment_name,
            "prediction",
            "pointwise_estimates.pt",
        )
        if os.path.exists(chat_path):
            print(f"Using saved pointwise_estimates.pt from {chat_path}")
            chat = torch.load(chat_path)
            return chat
        
        trainer = pl.Trainer(accelerator="auto", logger=False, devices=1)
        predictions = trainer.predict(model=self.model, datamodule=self.data_module)

        chat = torch.cat(predictions)
        
        return chat
    
    def _get_X(self):
        """
        Get the basis evaluations for the model
        
        returns:
            X: torch.Tensor (N, r) predicted network basis function evaluations
        """
        
        X_path = os.path.join(
            self.args.out_folder,
            self.args.experiment_name,
            "prediction",
            "basis_pointwise_estimates.pt",
        )
        if os.path.exists(X_path):
            print(f"Using saved basis_pointwise_estimates.pt from {X_path}")
            X = torch.load(X_path)
            return X
        
        trainer = pl.Trainer(accelerator="auto", logger=False, devices=1)
        self.model.use_basis_model = True
        predictions = trainer.predict(model=self.model, datamodule=self.data_module)
        self.model.use_basis_model = False

        X = torch.cat(predictions)
        
        if not self.model.args.use_baseline:
            torch.save(
                X.cpu().detach(),
                X_path,
            )
            print(
                f"Saved basis_pointwise_estimates.pt to ",
                X_path,
            )
        
        return X


def post_calibration(device, field_model, dataloader_calib, hyper_params, var_grid):
    llk = np.zeros(len(var_grid))

    coordmap_calib = dataloader_calib.dataset.X.to(device)
    data_calib_yvals = dataloader_calib.dataset.Y.to(device)

    Phi_tensor = hyper_params["Phi_tensor"]
    K = Phi_tensor.shape[1]

    for i, svar in enumerate(var_grid):
        sigma2_mu, sigma2_w = var_grid[i]
        hyper_params_i = copy.deepcopy(hyper_params)
        hyper_params_i["sigma2_w"] = sigma2_w
        hyper_params_i["sigma2_mu"] = sigma2_mu
        posterior_field = FVRF(
            field_model, coordmap_calib, data_calib_yvals.T, hyper_params_i
        )
        Post_mean_coefs, Post_cov_mats = posterior_field.compute_predictive_posterior(
            coordmap_calib
        )

        I_N = torch.eye(Phi_tensor.shape[0]).to(device)
        I_N = I_N.reshape((1, Phi_tensor.shape[0], Phi_tensor.shape[0]))
        I_N = I_N.repeat(Post_cov_mats.shape[0], 1, 1)
        Y_pred_mean = Post_mean_coefs @ Phi_tensor.T
        Y_pred_cov = (
            Phi_tensor @ Post_cov_mats @ Phi_tensor.T + hyper_params_i["sigma2_e"] * I_N
        )

        posterior_predictive = MultivariateNormal(Y_pred_mean, Y_pred_cov)
        llk[i] = float(
            posterior_predictive.log_prob(data_calib_yvals).sum().cpu().detach().numpy()
        )

    sigma2_mu_optim, sigma2_w_optim = var_grid[np.argmax(llk)]
    return sigma2_mu_optim, sigma2_w_optim
