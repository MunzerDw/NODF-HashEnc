from models.inr import INR
from torch import optim
import pytorch_lightning as pl
from torch import nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from models.hash_embeddings import HashEmbedder
from utility.utility import get_phi_r_tensors


class NODF(pl.LightningModule):
    def __init__(
        self,
        args: dict,
    ):
        super().__init__()
        self.args = args

        self.save_hyperparameters()
        
        self.use_basis_model = False
        
        # required only for training to calculate the penalized maximum likelihood estimate
        Phi_tensor, R_tensor = get_phi_r_tensors(args)
        self.register_buffer("R_tensor", R_tensor)
        self.register_buffer("Phi_tensor", Phi_tensor)

        K = int((args.sh_order + 1) * (args.sh_order + 2) / 2)

        if args.use_baseline:
            self.inr = INR(
                in_features=3,
                out_features=K,
                hidden_features=args.r,
                hidden_layers=args.depth,
                first_omega_0=args.omega0,
                hidden_omega_0=args.omega0_hidden,
                sigma0=args.sigma0,
                inr=self.args.inr,
                skip_conn=self.args.skip_conn,
                batchnorm=self.args.batchnorm,
            )
        else:
            self.inr = INR(
                in_features=(self.args.n_levels * self.args.n_features_per_level) + 3,
                out_features=K,
                hidden_features=args.r,
                hidden_layers=args.depth,
                first_omega_0=args.omega0,
                hidden_omega_0=args.omega0_hidden,
                sigma0=args.sigma0,
                inr=self.args.inr,
                skip_conn=self.args.skip_conn,
                batchnorm=self.args.batchnorm,
            )
            self.hash_embedder = HashEmbedder(
                n_levels=self.args.n_levels,
                n_features_per_level=self.args.n_features_per_level,
                log2_hashmap_size=self.args.log2_hashmap_size,
                base_resolution=self.args.base_resolution,
                per_level_scale=self.args.per_level_scale,
            )

    def forward(self, batch: "dict[torch.Tensor, torch.Tensor]"):
        """
        batch: tuple of (coords (B, 3), signal (B, M))

        returns: dict of torch.tensor model_output (B, K) (or (B, r) if basis model used)
        """
        coords = batch["coords"]
        
        # option to use the basis model only without W
        if not hasattr(self, 'use_basis_model'):
            self.use_basis_model = False
        inr = self.inr
        if self.use_basis_model:
            inr = self.get_basis()
            
        if self.args.use_baseline:
            model_output = inr(coords)
        else:
            embeddings = self.hash_embedder(coords)
            model_input = torch.cat([embeddings, coords], dim=-1)
            model_output = inr(model_input)

        return model_output

    def training_step(self, batch: "dict[torch.Tensor, torch.Tensor]", batch_idx: int):
        """
        ‘training_step’ is the method that is called for every batch of data in the training loop.

        batch: tuple of (coords (B, 3), signal (B, M))
        batch_idx: int

        returns: dict of torch.tensor losses
        """
        # forward pass
        chat = self.forward(batch)

        # loss
        losses = self._neg_pmle(chat, batch["signal"])
        loss_weights = {
            "prior_energy": self.args.lambda_c,
            "l2_loss": 1.0,
        }
        total_loss = 0
        for loss_name, value in losses.items():
            total_loss += loss_weights[loss_name] * value
        losses["loss"] = total_loss

        # logging
        for key, value in losses.items():
            self.log(
                f"{key}/train",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        losses = {"loss": losses["loss"]}
        return losses

    def configure_optimizers(self):
        """
        configure the optimizer and scheduler according to Pytorch Lightning
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = LambdaLR(
            optimizer, lambda x: 0.85 ** min(x / self.args.num_epochs, 1)
        )

        if self.args.enable_schedulers:
            schedulers = [scheduler]
        else:
            schedulers = []

        return [optimizer], schedulers

    def get_basis(self):
        """
        returns: the basis function of the INR
        """
        return nn.Sequential(*(list(self.inr.net.children())[:-1]))

    def _neg_pmle(self, chat: torch.Tensor, signal: torch.Tensor):
        """
        chat: torch.tensor (N x K) of coefficient field evaluations
        signal: torch.tensor (N x M) of (noisy + discretized) function samples

        returns: dict of torch.tensor losses
        """

        l2_loss = self._neg_log_likelihood(chat, signal, self.Phi_tensor)
        prior_energy = self._integrated_roughness(chat, self.R_tensor)

        return {
            "l2_loss": l2_loss,
            "prior_energy": prior_energy,
        }

    def _neg_log_likelihood(
        self, chat: torch.Tensor, signal: torch.Tensor, Phi_tensor: torch.Tensor
    ):
        """
        chat: B X K of coefficient field evaluations
        signal: B X M of (noisy + discretized) function samples
        Phi_tensor: torch.tensor (M x K) basis evaluation matrix

        returns: torch.tensor batch mean of negative log likelihood
        """
        yhat = chat @ Phi_tensor.T  # B X M (flattened predicted tensor)
        l2_loss = ((yhat - signal) ** 2).mean()
        return l2_loss

    def _integrated_roughness(self, chat: torch.Tensor, R_tensor: torch.Tensor):
        """
        chat: B X K of coefficient field evaluations
        R_tensor: K x K (diagonal) prior precision matrix of functions

        returns: torch.tensor batch mean of integrated roughness
        """
        energy = torch.mean((chat**2) * torch.diag(R_tensor))
        return energy

    def count_parameters(self):
        """
        Counts and returns the number of trainable parameters in the model.
        """
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            print(f"{name}: {params}")
            total_params += params
        print(f"Total Trainable Params: {total_params}")
        return total_params
