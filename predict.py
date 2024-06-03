import time
import torch
from utility.utility import get_args
import numpy as np
from models.nodf import NODF
import os
from utility.utility import get_args
from data_module import DataModule
import numpy as np
import torch
from models.nodf import NODF
import os
import pytorch_lightning as pl


def main(args: dict):
    # fix the seed
    torch.manual_seed(0)
    np.random.seed(0)
    pl.seed_everything(0, workers=True)

    # get output path
    output_path = os.path.join(args.out_folder, args.experiment_name, "prediction")
    os.makedirs(output_path, exist_ok=True)
    print(f"=====> Output path: {output_path}")

    print("==> initializing data ...")
    data_module = DataModule(args)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(accelerator="auto", logger=False, devices=1)

    print("==> loading model ...")
    model = NODF.load_from_checkpoint(args.ckpt_path, map_location="cpu")

    print("==> start predicting ...")
    start_time = time.time()
    predictions = trainer.predict(model=model, datamodule=data_module)
    end_time = time.time()
    time_in_sec = round(end_time - start_time, 2)

    chat = torch.cat(predictions)
    num_points = chat.shape[0]
    print(f"Inference time: {time_in_sec} seconds | {num_points} points")

    # save point-wise estimates of the ODF coefficients
    path = os.path.join(
        args.out_folder,
        args.experiment_name,
        "prediction",
        "pointwise_estimates.pt",
    )
    torch.save(
        chat.cpu().detach(),
        path,
    )
    print(
        f"Saved pointwise_estimates.pt to ",
        path,
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
