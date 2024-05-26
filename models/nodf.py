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
        chat = self.forward(batch) # B x K
        
        # for total variation prior
        chat_difference_vectors = None
        if self.args.use_tv:
            # evaluation on coordinates with offsets (for total variation penalty)
            coords = batch['coords'].clone()
            chat_difference_vectors = []
            for i in range(coords.shape[-1]):
                # evaluation on coordinates with offset
                coords[:, i] = coords[:, i] + self.args.offset_tv
                chat_positive_offset = self.forward({"coords": coords}) # B x K
                coords[:, i] = coords[:, i] - (2 * self.args.offset_tv)
                chat_negative_offset = self.forward({"coords": coords}) # B x K
                
                # reset coords
                coords[:, i] = coords[:, i] + self.args.offset_tv
                
                # add chat difference for coord i
                chat_difference_vectors.append(chat_positive_offset - chat_negative_offset) # B x K
            chat_difference_vectors = torch.stack(chat_difference_vectors, dim=-1) # B x K x 3

        # loss
        losses = self._neg_pmle(chat, chat_difference_vectors, batch["signal"], batch['coords'])
        loss_weights = {
            "prior_energy": self.args.lambda_c,
            "l2_loss": 1.0,
            "total_variation": self.args.lambda_tv,
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

    def _neg_pmle(self, chat: torch.Tensor, chat_difference_vectors: torch.Tensor, signal: torch.Tensor, coords: torch.Tensor):
        """
        chat: torch.tensor (N x K) of coefficient field evaluations
        chat_difference_vectors: torch.Tensor (B x K x 3) difference of coefficient field evaluations w.r.t. neighborhood
        signal: torch.tensor (N x M) of (noisy + discretized) function samples
        coords: torch.tensor (N x 3) of (noisy + discretized) input coordinates

        returns: dict of torch.tensor losses
        """

        l2_loss = self._neg_log_likelihood(chat, signal, self.Phi_tensor)
        prior_energy = self._integrated_roughness(chat, self.R_tensor)
        total_variation = 0.0
        if chat_difference_vectors != None:
            total_variation = self._total_variation(chat_difference_vectors, coords, self.Phi_tensor)

        return {
            "l2_loss": l2_loss,
            "prior_energy": prior_energy,
            "total_variation": total_variation
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
    
    def _total_variation(self, chat_difference_vectors, coords, Phi_tensor):
        """
        Computes gradient of the model output with respect to the input coordinates.
        
        chat_difference_vectors: torch.Tensor (B x K x 3) difference of coefficient field evaluations w.r.t. neighborhood
        coords: torch.Tensor (B x N x 3) of coordinates
        Phi_tensor: torch.tensor (M x K) basis evaluation matrix
        
        returns:
            gradients: torch.Tensor (B x M x 3) gradients
        """
        
        chat_gradients = torch.norm(chat_difference_vectors, dim=-1)
        chat_gradients = chat_gradients.mean()
        
        return chat_gradients
        
        # # get predicted signal
        # yhat = chat @ Phi_tensor.T # (B x M)
        
        # grads = []
        # for i in range(yhat.shape[-1]):
        #     img_grad = yhat[:, i] # (B)
        #     grad_outputs = torch.ones_like(img_grad) # (B)
        #     # Compute gradient and laplacian
        #     grad = torch.autograd.grad(
        #         img_grad,
        #         [coords],
        #         grad_outputs=grad_outputs,
        #         create_graph=True,
        #     )[0][:, -3:] # (B x 3)
        #     grads_x = grad[:, 0][..., None] # (B x 1)
        #     grads_y = grad[:, 1][..., None] # (B x 1)
        #     grads_z = grad[:, 2][..., None] # (B x 1)

        #     grads_stacked = torch.stack((grads_x, grads_y, grads_z), dim=-1) # (B x 3)
        #     grads.append(grads_stacked)

        # grads = torch.cat(grads, dim=-2) # (B x M x 3)
        
        # grads = (grads**2)
        
        # total_variation = grads.mean()
        
        # return total_variation

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
