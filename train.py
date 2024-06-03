from utility.gpu_cache_clean_callback import GPUCacheCleanCallback
from utility.time_logging_callback import TimeLoggingCallback
from utility.utility import get_args, get_phi_r_tensors
from data_module import DataModule
import numpy as np
import torch
from models.nodf import NODF
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def init_callbacks(output_path: str):
    latest_checkpoint_monitor = ModelCheckpoint(
        dirpath=os.path.join(output_path),
        mode="max",
        save_last=False,
        save_top_k=-1,
        every_n_epochs=100,
    )
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    time_logging_callback = TimeLoggingCallback()

    result = [
        latest_checkpoint_monitor,
        gpu_cache_clean_monitor,
        lr_monitor,
        time_logging_callback,
    ]

    return result


def main(args: dict):
    # fix the seed
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(0, workers=True)

    # get output path
    output_path = os.path.join(args.out_folder, args.experiment_name, "training")
    os.makedirs(output_path, exist_ok=True)
    print(f"=====> Output path: {output_path}")

    print("==> initializing data ...")
    data_module = DataModule(args)

    print("==> initializing logger ...")
    logger = TensorBoardLogger(
        save_dir=output_path, name="nodf", default_hp_metric=True
    )

    print("==> initializing monitors ...")
    callbacks = init_callbacks(
        os.path.join(logger.root_dir, "version_" + str(logger.version), "checkpoints"),
    )

    print("==> initializing trainer ...")
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        max_epochs=args.num_epochs,
        log_every_n_steps=1000,
        devices="auto",
    )

    print("==> initializing model ...")
    if args.ckpt_path:
        model = NODF.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    else:
        model = NODF(args)

    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = get_args()

    main(args)
