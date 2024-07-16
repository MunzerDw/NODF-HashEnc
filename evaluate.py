import time
import nibabel as nib
import torch
from models.posterior import FVRF
from utility.utility import (
    get_args,
    cart2sphere,
    get_mask,
    get_phi_r_tensors,
    save_nif,
)
from dipy.reconst.shm import real_sym_sh_basis
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
import os
from dipy.reconst.odf import gfa
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.data import get_sphere
import os
from dipy.data import get_sphere
import nibabel as nib
import numpy as np
from dipy.data import get_sphere, HemiSphere
from dipy.direction import (
    DeterministicMaximumDirectionGetter,
)
from dipy.tracking.stopping_criterion import (
    ThresholdStoppingCriterion,
)
from dipy.tracking import utils as track_utils
from dipy.tracking.local_tracking import LocalTracking
from nibabel.streamlines import LazyTractogram
from dipy.tracking.streamlinespeed import length
from dipy.io.utils import get_reference_info, create_tractogram_header


class Evaluation:

    def __init__(self, args, save_files=True):
        self.args = args
        self.save_files = save_files

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.output_path = os.path.join(
            args.out_folder, args.experiment_name, "evaluation"
        )
        os.makedirs(self.output_path, exist_ok=True)

        if not args.predictions_path:
            args.predictions_path = os.path.join(
                args.out_folder,
                args.experiment_name,
                "prediction/pointwise_estimates.pt",
            )

        self.odfs = self._get_odfs()  # X x Y x Z x K
        self.gt_odfs = self._get_odfs(args.gt_odfs_path)  # X x Y x Z x K

        input_path = args.predictions_path
        output_path = args.out_folder
        print(f"Using predictions at: {input_path}")
        print(f"Output path: {output_path}")

    def get_odf_error(self):
        """
        Calculate the ODF error using the L2-norm

        returns:
            errors_median: torch.Tensor (1),
            errors: torch.Tensor (X x Y x Z)
        """

        if self.gt_odfs is None:
            print("Can't calculate ODF error without ground truth ODFs")
            return

        mask = get_mask(args)

        odfs_flat = self.odfs[mask].cpu().detach().numpy()
        gt_odfs_flat = self.gt_odfs[mask].cpu().detach().numpy()

        odfs_diff = odfs_flat - gt_odfs_flat
        errors = np.linalg.norm(odfs_diff, ord=2, axis=-1) / np.linalg.norm(
            gt_odfs_flat, ord=2, axis=-1
        )
        errors = torch.from_numpy(errors).float().to(self.device)
        errors_median = torch.median(errors)
        print(f"ODF L2-Norm Median Error: {errors_median}")

        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)  # X x Y x Z
        odfs_l2_norm_error_img = torch.zeros(mask_full.shape).to(
            self.device
        )  # X x Y x Z
        odfs_l2_norm_error_img[mask_full] = errors

        if self.save_files:
            torch.save(
                errors.cpu().detach(),
                os.path.join(self.output_path, f"odfs_l2_norm_error_values.pt"),
            )
            save_nif(
                args,
                odfs_l2_norm_error_img.cpu().detach().numpy(),
                os.path.join(self.output_path, f"odfs_l2_norm_error_map.nii.gz"),
            )

        return errors_median, errors

    def get_gfa(self):
        """
        Get the general fractional anisotropy (GFA) of the ODFs

        returns: nifti image (X x Y x Z)
        """

        mask = get_mask(self.args)  # X x Y x Z
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)  # X x Y x Z

        odfs = self.odfs  # X x Y x Z x K

        B = self._get_B()  # K x P

        if self.gt_odfs is not None:
            gt_odfs = self.gt_odfs  # X x Y x Z x K

            gt_signal_gfa_flat = gfa((gt_odfs[mask] @ B).cpu().detach().numpy())  # N
            signal_gfa_flat = gfa((odfs[mask] @ B).cpu().detach().numpy())  # N
            signal_gfa_flat[np.isnan(signal_gfa_flat)] = 0.0

            gfa_diff = signal_gfa_flat - gt_signal_gfa_flat  # N
            gfa_abs_diff = np.absolute(gfa_diff)  # N

            abs_errors_median = np.median(gfa_abs_diff)  # 1
            print(f"GFA Median Absolute Error: {abs_errors_median}")

        gfa_img = torch.zeros(odfs.shape[:-1]).to(self.device)  # X x Y x Z
        signal_gfa_flat = gfa((odfs[mask_full] @ B).cpu().detach().numpy())  # N
        gfa_img[mask_full] = torch.from_numpy(signal_gfa_flat).float().to(self.device)

        # save nifiti
        if self.save_files:
            save_nif(
                args,
                gfa_img.cpu().detach().numpy(),
                os.path.join(self.output_path, f"gfa.nii.gz"),
            )

        return gfa_img

    def get_signal(self):
        """
        Get the reconstructed signal with b0 volume from the training image (average of b0 volumes).
        Also returns bvecs and bvals with 0 bval and 0 bvec added.

        returns:
            signal_reconstructed: nifit image (X x Y x Z, M),
            bvecs_signal_reconstructed: text file (M x 3),
            bvals_signal_reconstructed: text file (M)
        """

        # loading bvals and bvecs
        bvals, bvecs = read_bvals_bvecs(
            self.args.bval_file,
            self.args.bvec_file,
        )
        b0_bval_indices = np.where(bvals < self.args.bmarg)[0]
        b_bval_indices = np.where(
            (bvals >= self.args.bval - self.args.bmarg)
            & (bvals <= self.args.bval + self.args.bmarg)
        )[0]
        bvecs = bvecs[b_bval_indices[: self.args.M]]
        bvals = bvals[b_bval_indices[: self.args.M]]

        mask_full = nib.load(args.mask_file).get_fdata().astype(bool)

        # get reconstructed signal for M gradient directions from predicted ODFs
        signal_recon = self._get_signal()  # X, Y, Z, M
        signal_recon = signal_recon.clip(0, 1)  # X, Y, Z, M

        signal_b0_mean_path = os.path.join(self.args.data, "train_signal_b0_average.pt")
        if os.path.exists(signal_b0_mean_path):
            print(f"==> Loading b0 volume average from {signal_b0_mean_path} ...")
            signal_b0_mean_flat = torch.load(signal_b0_mean_path).numpy()[:, None]
            signal_b0_mean = np.zeros(
                (*self.odfs.shape[:-1], signal_b0_mean_flat.shape[-1])
            )  # X x Y x Z x 1

            signal_b0_mean[mask_full] = signal_b0_mean_flat
        else:
            print("==> Calculating b0 volume average ...")
            # load signal
            img = nib.load(self.args.img_file)
            signal_raw = img.get_fdata()  # X, Y, Z, b

            # to prevent division by very small numbers
            signal_raw[signal_raw <= 1e-2] = 1e-2

            # normalize signal by b0
            signal_b0_mean = (
                signal_raw[:, :, :, b0_bval_indices].mean(axis=3).unsqueeze(-1)
            )  # X, Y, Z, 1

            torch.save(
                torch.from_numpy(signal_b0_mean[mask_full]).cpu().float(),
                signal_b0_mean_path,
            )
            print(f"==> Saved b0 volume average to {signal_b0_mean_path}")

        # adding b0 volume of 1s
        signal_recon_expanded = signal_recon * signal_b0_mean
        signal_recon_expanded = np.concatenate(
            [signal_b0_mean, signal_recon_expanded], axis=-1
        ).astype(
            np.float32
        )  # X, Y, Z, M + 1
        print("==> Added b0 volume")

        # adding 0 bval and 0 bvec
        bvals = np.insert(bvals, 0, 0)
        bvecs = np.insert(bvecs, obj=0, values=[0, 0, 0], axis=0)

        if self.save_files:
            save_nif(
                args,
                signal_recon_expanded,
                os.path.join(self.output_path, f"signal_reconstructed.nii.gz"),
            )
            np.savetxt(
                os.path.join(self.output_path, "bvecs_signal_reconstructed.txt"),
                bvecs.T,
                fmt="%.6f",
            )
            np.savetxt(
                os.path.join(self.output_path, "bvals_signal_reconstructed.txt"),
                bvals.reshape(1, -1),
                fmt="%.6f",
            )

    def get_dti(self):
        """
        Get the Diffusion Tensor Imaging (DTI) of the ODFs

        returns:
            tensor_fa: nifti image (X x Y x Z),
            tensor_evecs: nifti image (X x Y x Z x 3 x 3),
            tensors_md: nifti image (X x Y x Z),
            tensor_rgb: nifti image (X x Y x Z x 3)
        """

        # loading bvals and bvecs
        bvals, bvecs = read_bvals_bvecs(
            self.args.bval_file,
            self.args.bvec_file,
        )
        b_bval_indices = np.where(
            (bvals >= self.args.bval - self.args.bmarg)
            & (bvals <= self.args.bval + self.args.bmarg)
        )[0]
        bvecs = bvecs[b_bval_indices[: self.args.M]]
        bvals = bvals[b_bval_indices[: self.args.M]]

        # get reconstructed signal for M gradient directions from predicted ODFs
        signal_recon = self._get_signal()  # X, Y, Z, M

        signal_b0_mean = torch.ones((*signal_recon.shape[:-1], 1))  # X, Y, Z, 1

        print("==> Adding b0 volume ...")
        # adding b0 volume of 1s
        signal_recon_expanded = signal_recon * signal_b0_mean
        data = np.concatenate([signal_b0_mean, signal_recon_expanded], axis=-1)

        # adding 0 bval and 0 bvec
        bvals = np.insert(bvals, 0, 0)
        bvecs = np.insert(bvecs, obj=0, values=[0, 0, 0], axis=0)

        print("==> Preparing for DTI ...")
        gtab = gradient_table(bvals, bvecs)
        tenmodel = dti.TensorModel(gtab)

        print("==> Doing DTI ...")
        tenfit = tenmodel.fit(data)

        print("==> Computing anisotropy measures (FA, MD, RGB)")
        FA = fractional_anisotropy(tenfit.evals)

        FA[np.isnan(FA)] = 0

        tensor_fa = FA.astype(np.float32)

        tensor_evecs = tenfit.evecs.astype(np.float32)

        tensors_md = dti.mean_diffusivity(tenfit.evals).astype(np.float32)

        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, tenfit.evecs)
        tensor_rgb = np.array(255 * RGB, "uint8")

        # save nifiti
        if self.save_files:
            save_nif(
                args,
                tensor_fa,
                os.path.join(self.output_path, f"tensor_fa.nii.gz"),
            )
            save_nif(
                args,
                tensor_evecs,
                os.path.join(self.output_path, f"tensor_evecs.nii.gz"),
            )
            save_nif(
                args,
                tensors_md,
                os.path.join(self.output_path, f"tensors_md.nii.gz"),
            )
            save_nif(
                args,
                tensor_rgb,
                os.path.join(self.output_path, f"tensor_rgb.nii.gz"),
            )

        return tensor_fa, tensor_evecs, tensors_md, tensor_rgb

    def get_fsim(self):
        """
        Get the FSIM score for the GFA and DTI images

        returns: (float, np.array), (float, np.array)
        """

        gfa_fsim_values_path = os.path.join(self.output_path, f"gfa_fsim.pt")
        dti_fsim_values_path = os.path.join(self.output_path, f"dti_fsim_values.pt")
        if os.path.exists(gfa_fsim_values_path):
            gfa_fsim_values = torch.load(gfa_fsim_values_path)
            gfa_fsim = torch.median(gfa_fsim_values)
            print(f"FSIM median GFA: {gfa_fsim}")
        if os.path.exists(dti_fsim_values_path):
            dti_fsim_values = torch.load(dti_fsim_values_path)
            dti_fsim = torch.median(dti_fsim_values)
            print(f"FSIM median DTI: {dti_fsim}")

        # gt data
        gt_gfa = nib.load(args.gt_gfa_path).get_fdata()
        gt_dti = nib.load(args.gt_dti_path).get_fdata()

        # load data
        gfa_img = nib.load(os.path.join(self.output_path, "gfa.nii.gz")).get_fdata()
        tensor_rgb = nib.load(
            os.path.join(self.output_path, "tensor_rgb.nii.gz")
        ).get_fdata()

        gfa_fsim, gfa_fsim_values = self._get_fsim_score(gfa_img, gt_gfa)
        print(f"FSIM median GFA: {gfa_fsim}")
        dti_fsim, dti_fsim_values = self._get_fsim_score(tensor_rgb, gt_dti)
        print(f"FSIM median DTI: {dti_fsim}")

        # save nifiti
        if self.save_files:
            torch.save(gfa_fsim_values.cpu().detach(), gfa_fsim_values_path)
            torch.save(
                dti_fsim_values.cpu().detach(),
                os.path.join(self.output_path, f"dti_fsim_values.pt"),
            )

        return (gfa_fsim, gfa_fsim_values), (dti_fsim, dti_fsim_values)

    def uq(self):
        """
        Calculates the posterior, samples it, and calculates the coefficient of variation
        for GFA and its correlation to ODF normalized L2 error

        returns: torch.Tensor (S, N, K); S sampled ODF coefficients
        """

        if not args.ckpt_path:
            print(
                "Can't perform uncertainty quantification without a model checkpoint path (args.ckpt_path)"
            )
            return

        ckpt_path = args.ckpt_path
        print(f"Using checkpoint at: {ckpt_path}")

        start_time = time.time()
        posterior = FVRF(args)
        end_time = time.time()
        time_in_sec = round(end_time - start_time, 2)
        print(f"Calculating W posterior: {time_in_sec} seconds")

        # get roi
        mask = get_mask(args)
        # mask[:168] = False
        # mask[169:] = False
        # mask[:, 126:] = False
        # mask[:, :, 102:] = False
        # mask[:, :, :90] = False
        # mask[:, :, 91:] = False
        # sagittal
        # mask[:, :74] = False
        # mask[:, 88:] = False
        # # coronal
        # mask[:, :, :67] = False
        # mask[:, :, 85:] = False
        num_points = mask[mask].shape[0]

        # generate posterior samples
        npost_samps = 250
        start_time = time.time()
        post_samples_chat = posterior.sample_posterior_pointwise(
            mask, npost_samps=npost_samps
        )  # (S, N, K)
        post_samples_chat = post_samples_chat.to(self.device)
        end_time = time.time()
        time_in_sec = round(end_time - start_time, 2)
        print(
            f"Sampling posterior time: {time_in_sec} seconds | {num_points} points | {npost_samps} samples"
        )

        # calculate uncertainty maps for different measures:
        # GFA
        mask_full = nib.load(args.mask_file).get_fdata().astype(bool)
        B = self._get_B()

        post_samples_gfa = gfa(
            (post_samples_chat @ B).cpu().detach().numpy()
        ).T  # (N, S)
        post_samples_gfa_error_flat = post_samples_gfa.std(-1) / post_samples_gfa.mean(
            -1
        )  # (N)

        post_samples_gfa_uq = torch.zeros(mask_full.shape)  # (X, Y, Z)
        post_samples_gfa_uq[mask] = torch.from_numpy(
            post_samples_gfa_error_flat
        ).float()
        # save nifiti
        if self.save_files:
            save_nif(
                args,
                post_samples_gfa_uq.cpu().detach().numpy(),
                os.path.join(
                    self.output_path, f"post_{npost_samps}_samples_gfa_uq.nii.gz"
                ),
            )

        # calculate correlation to ODF normalized L2 error
        odf_errors = torch.load(
            os.path.join(self.output_path, f"odfs_l2_norm_error_values.pt")
        )
        odf_errors_img = torch.zeros(mask.shape)
        odf_errors_img[mask_full] = odf_errors
        odf_errors = odf_errors_img[mask]

        corr = np.corrcoef(post_samples_gfa_uq[mask], odf_errors)[0][1]
        print(f"Correlation of GFA uncertainty to ODF normalized L2 error: {corr}")

        return post_samples_chat

    def get_tractogrophy(self):
        args = self.args
        mask = get_mask(args)
        # MODEL = "SS_CSD"
        # if MODEL == "LFI":
        #     fname = "/Users/willconsagra/Documents/Workspace/UQ_SBI/realdataanalysis/715041/processed_signals/hcp_predictions_final_V1/fodf_LFI_tournier07.nii.gz"
        #     out_tractogram = "/Users/willconsagra/Documents/Workspace/UQ_SBI/realdataanalysis/715041/processed_signals/hcp_predictions_final_V1/LFI_FB_ode.tck"
        # elif MODEL == "SS_CSD":
        # fname = "/Users/willconsagra/Documents/Workspace/UQ_SBI/realdataanalysis/fodf_examples/fodfs_csd_bv3_tourn.nii.gz"
        # out_tractogram = "/Users/willconsagra/Documents/Workspace/UQ_SBI/realdataanalysis/streamlines/csd_bv3_FB_ode.tck"

        fname = os.path.join(
            args.out_folder,
            args.experiment_name,
            "visualization/odfs_tournier07.nii.gz",
        )
        out_tractogram = os.path.join(
            args.out_folder,
            args.experiment_name,
            "visualization/tractogrophy_csd_bv3_FB_ode.tck",
        )

        # elif MODEL == "MSMT_CSD":
        #     pass

        fodf_sh_img = nib.load(fname)
        sh_basis = "tournier07"
        theta = 30
        fa_thresh = 0.2
        step_size = 0.2
        min_length = 10
        max_length = 500

        print("Getting FA image ... ")
        nx, ny, nz = fodf_sh_img.shape[:-1]
        fodf_sh_img = fodf_sh_img.get_fdata(dtype=np.float32)
        fodf_sh_img[~mask] = 0

        tracking_sphere = HemiSphere.from_sphere(get_sphere("repulsion724"))
        dgklass = DeterministicMaximumDirectionGetter

        print("Creating direction getter for " + sh_basis)
        ## direction getter
        dg = dgklass.from_shcoeff(
            fodf_sh_img,
            max_angle=theta,
            sphere=tracking_sphere,
            basis_type=sh_basis,
        )
        ## stopping criterion
        fa_img_path = os.path.join(
            args.out_folder,
            args.experiment_name,
            "evaluation/tensor_fa.nii.gz",
        )
        fa_img = nib.load(fa_img_path)
        FA = fa_img.get_fdata(dtype=np.float32)
        threshold_criterion = ThresholdStoppingCriterion(FA, fa_thresh)

        ## seeds
        seed_mask = np.zeros((nx, ny, nz))
        seed_mask = FA >= 0.25
        seeds = track_utils.seeds_from_mask(seed_mask, np.eye(4), density=1)
        max_steps = int(max_length / step_size) + 1

        print("Performing tractogrophy ...")
        ## ODE curve evolition
        streamline_generator = LocalTracking(
            dg,
            threshold_criterion,
            seeds,
            np.eye(4),
            step_size=step_size,
            maxlen=max_steps,
            return_all=True,
            random_seed=0,
        )
        ## filter out curve snips
        filtered_streamlines = (
            s for s in streamline_generator if min_length <= length(s) <= max_length
        )
        data_per_streamlines = {}
        tractogram = LazyTractogram(
            lambda: filtered_streamlines,
            data_per_streamlines,
            affine_to_rasmm=fa_img.affine,
        )

        ## save tractogram
        filetype = nib.streamlines.detect_format(out_tractogram)
        reference = get_reference_info(fa_img)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, out_tractogram, header=header)

    def _get_fsim_score(self, pred_imgs, gt_imgs):
        """
        Get the FSIM score for the given images

        pred_imgs: torch.Tensor (X x Y x Z) or (X x Y x Z x 3)
        gt_imgs: torch.Tensor (X x Y x Z) or (X x Y x Z x 3)

        returns: float, torch.Tensor (F)
        """
        from image_similarity_measures.quality_metrics import fsim

        # remove empty spaces
        gt_imgs = gt_imgs[50:240, 39:263, 0:178]
        pred_imgs = pred_imgs[50:240, 39:263, 0:178]

        if len(gt_imgs.shape) == 3:
            # handle gray scale images
            gt_imgs = (
                ((gt_imgs - gt_imgs.min()) / (gt_imgs.max() - gt_imgs.min())) * 255
            ).astype(np.uint8)
            gt_imgs = np.expand_dims(gt_imgs, axis=-1)  # X x Y x Z x 1

            pred_imgs = (
                ((pred_imgs - pred_imgs.min()) / (pred_imgs.max() - pred_imgs.min()))
                * 255
            ).astype(np.uint8)
            pred_imgs = np.expand_dims(pred_imgs, axis=-1)  # X x Y x Z x 1

        # get fsim score for all axial, sagittal, and coronal slices
        slices_results = []
        print("==> Computing FSIM ...")
        for i in range(gt_imgs.shape[0]):
            slices_results.append(fsim(org_img=gt_imgs[i], pred_img=pred_imgs[i]))
        for i in range(gt_imgs.shape[1]):
            slices_results.append(fsim(org_img=gt_imgs[:, i], pred_img=pred_imgs[:, i]))
        for i in range(gt_imgs.shape[2]):
            slices_results.append(
                fsim(org_img=gt_imgs[:, :, i], pred_img=pred_imgs[:, :, i])
            )

        values = torch.Tensor(slices_results).float()
        values = values[~torch.isnan(values)]  # remove nans, coming from black images
        median = torch.median(values)  # get median value
        median = torch.round(median * 100) / 100  # round to 2 decimal places

        return median, values

    def _get_signal(self):
        """
        Gets the signal reconstructed from the ODFs

        returns: torch.Tensor (X x Y x Z x M)
        """
        mask_full = nib.load(self.args.mask_file).get_fdata().astype(bool)

        # get data
        odfs_flat = self.odfs[mask_full]  # N x K
        Phi_tensor, _ = get_phi_r_tensors(self.args)
        signal_pred = odfs_flat @ Phi_tensor.T  # N x M

        signal_pred_img = torch.zeros((*self.odfs.shape[:-1], self.args.M)).to(
            self.device
        )  # X x Y x Z x M
        signal_pred_img[mask_full] = signal_pred

        return signal_pred_img

    def _get_odfs(self, path=None):
        """
        Loads the ODFs from the given path

        path: str, optional

        returns: torch.Tensor (X x Y x Z x K)
        """

        if path is None:
            path = self.args.predictions_path

        if os.path.exists(path) is False:
            print(f"ODFs path does not exist: {path}")
            return None

        mask_full = nib.load(args.mask_file).get_fdata().astype(bool)  # X x Y x Z

        pointwise_estimate = torch.load(path, map_location=self.device).float()  # N x K

        odfs = torch.zeros((*mask_full.shape, pointwise_estimate.shape[-1])).to(
            self.device
        )  # X x Y x Z x K
        odfs[mask_full] = pointwise_estimate  # X x Y x Z x K

        return odfs

    def _get_B(self):
        """
        Gets the B matrix for the SH basis

        returns: torch.Tensor (K x P)
        """

        sphere = get_sphere("repulsion724")
        x_grid = cart2sphere(sphere.vertices)
        theta_grid = x_grid[:, 0]
        phi_grid = x_grid[:, 1]
        B, _, _ = real_sym_sh_basis(self.args.sh_order, phi_grid, theta_grid)
        B = torch.from_numpy(B.T).float().to(self.device)  # K x P
        return B


def main(args):
    eval = Evaluation(args)

    # eval.get_signal()
    # eval.get_tractogrophy()
    eval.get_gfa()
    eval.get_dti()
    eval.get_fsim()
    # eval.uq()
    eval.get_odf_error()


if __name__ == "__main__":
    args = get_args()

    main(args)
