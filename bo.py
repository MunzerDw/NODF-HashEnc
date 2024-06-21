import datetime
from functools import partial
import os
import torch
import numpy as np
from ax.service.ax_client import AxClient
import copy
from data_module import DataModule
from dataset import ObservationPoints
from models.nodf import NODF

from utility.utility import get_phi_r_tensors, matern_spec_density
from torch.utils.data import DataLoader
from utility.utility import get_args
import pytorch_lightning as pl


def resample_data(Obs, batch_size, train_prop=0.8):
    dataset_size = len(Obs)
    indices = list(range(dataset_size))
    split = int(np.floor((1 - train_prop) * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    N_test = len(test_indices)
    Y_train = Obs.signal[train_indices, :]
    Y_test = Obs.signal[test_indices, :]
    V_train = Obs.coords[train_indices, :]
    V_test = Obs.coords[test_indices, :]

    O_train = ObservationPoints(V_train, Y_train)
    O_test = ObservationPoints(V_test, Y_test)

    dataloader_train = DataLoader(
        O_train,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
    )
    dataloader_test = DataLoader(
        O_test, shuffle=True, batch_size=N_test, pin_memory=True, num_workers=0
    )

    return dataloader_train, dataloader_test


def run_experiment(
    device,
    args,
    parameterization,
    algorithm,
    dataset,
    train_prop=0.8
):
    # get data
    dataloader_train, dataloader_test = resample_data(
        dataset, args.batch_size, train_prop=train_prop
    )
    
    # get hyper parameters
    ## obtain optimization parameters based on values in `parameterization' dict
    if "learning_rate" in parameterization:
        args.learning_rate = parameterization["learning_rate"]
    if "r" in parameterization:
        args.r = parameterization["r"]
    if "omega0" in parameterization:
        args.omega0 = parameterization["omega0"]
    if "omega0_hidden" in parameterization:
        args.omega0_hidden = parameterization["omega0_hidden"]
    if "sigma0" in parameterization:
        args.sigma0 = parameterization["sigma0"]
    if "sigma2_w" in parameterization:
        args.sigma2_w = parameterization["sigma2_w"]
    if "base_resolution" in parameterization:
        args.base_resolution = parameterization["base_resolution"]
    if "n_levels" in parameterization:
        args.n_levels = parameterization["n_levels"]

    # get model
    field_model = NODF(args)

    # train
    ## estimate parameters
    trainer = algorithm()
    trainer.fit(
        model=field_model,
        train_dataloaders=dataloader_train,
    )

    # evaluate
    ## get mode predictions at test locations
    Phi_tensor = field_model.Phi_tensor.to(device)
    coords_test = dataloader_test.dataset.coords.to(device)
    Y_tensor_test = dataloader_test.dataset.signal.T.to(device)
    field_model = field_model.to(device)

    C_hat_test = field_model({"coords": coords_test})

    loss = float(
        ((Y_tensor_test - Phi_tensor @ C_hat_test.T) ** 2).mean().cpu().detach().numpy()
    )

    return loss


def BO_optimization(
    device,
    args,
    dataset,
    algorithm,
    parameter_map,
    Nexperiments,
    output_path,
    experiment_name="hyper_param_opt_experiment",
):
    """
    Implements BO-based hyper-parameter optimization: Algorithm 3 from arXiv:2307.08138
    """
    map_ax_client_path = os.path.join(output_path, 'map_ax_client.json')
    if args.ckpt_path:
        map_ax_client = AxClient.load_from_json_file(filepath=args.ckpt_path)
        print("Loaded map_ax_client")
    else:
        map_ax_client = AxClient()
        map_ax_client.create_experiment(
            name=experiment_name,
            parameters=parameter_map,
            minimize=True,
        )
    for _ in range(Nexperiments):
        parameters, trial_index = map_ax_client.get_next_trial()
        map_ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=run_experiment(
                device,
                args,
                parameters,
                algorithm,
                dataset
            ),
        )

        trial_loss = map_ax_client.get_trial(trial_index).get_metric_mean(
            metric_name="objective"
        )
        parameters["objective"] = trial_loss
        parameters["trial_index"] = trial_index

        print(f"======== trial {trial_index} completed ========================")
        print(f"parameters: {parameters}")
        print(f"objective: {trial_loss}")
        print("================================================================")

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        trial_stats_path = os.path.join(output_path, f"results_{date}.txt")
        save_trial_stats(parameters, trial_stats_path)
        print(f"Saved parameters to {output_path}")
        map_ax_client.save_to_json_file(filepath=map_ax_client_path)
        print(f"Saved map_ax_client to {map_ax_client_path}")

    best_trial_index, best_parameters, metrics = map_ax_client.get_best_trial()
    best_parameters["objective"] = metrics[0]["objective"]
    best_parameters["trial_index"] = best_trial_index
    best_parameters["is_best"] = True
    best_trial_stats_path = os.path.join(output_path, f"results_best.txt")
    save_trial_stats(best_parameters, best_trial_stats_path)


def save_trial_stats(parameters: dict, output_path: str):
    """
    save parameters dict as a new line in a file called results.json in the output_path

    parameters: dict containing tested parameters and the objective value
    output_path: str output file path
    """
    with open(output_path, "a") as f:
        f.write(str(parameters) + "\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get output path
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = os.path.join(args.out_folder, args.experiment_name, "bo", date)
    os.makedirs(output_path, exist_ok=True)
    print(f"=====> Output path: {output_path}")

    # run hyer-parameter optimization scheme to select prior roughness penalty strength
    parameter_map = [
        # {
        #     "name": "lambda_c",
        #     "type": "range",
        #     "bounds": [1e-9, 1e-1],
        #     "log_scale": True,
        # },
        # {
        #     "name": "nu",
        #     "type": "range",
        #     "bounds": [0.5, 2.0],
        # },
        # {
        #     "name": "omega0",
        #     "type": "range",
        #     "bounds": [1, 10],
        # },
        # {
        #     "name": "omega0_hidden",
        #     "type": "range",
        #     "bounds": [1, 10],
        # },
        # {
        #     "name": "sigma0",
        #     "type": "range",
        #     "bounds": [5, 25],
        # },
        # {
        #     "name": "learning_rate",
        #     "type": "range",
        #     "bounds": [1e-8, 1e-4],
        # },
        # {
        #     "name": "sigma2_w",
        #     "type": "range",
        #     "bounds": [1e-5, 1e5],
        # },
        # {
        #     "name": "n_levels",
        #     "type": "range",
        #     "bounds": [4, 32],
        # },
        # {
        #     "name": "base_resolution",
        #     "type": "range",
        #     "bounds": [4, 64],
        # },
    ]

    print("==> initializing data ...")
    data_module = DataModule(args)
    data_module.setup("fit")

    print("==> initializing trainer ...")
    trainer = partial(
        pl.Trainer,
        logger=False,
        accelerator="auto",
        max_epochs=200,
        devices="1",
        enable_checkpointing=False,
    )

    print("==> running BO ...")
    BO_optimization(
        device,
        args,
        data_module.dataset,
        trainer,
        parameter_map,
        args.Nexperiments,
        output_path,
        experiment_name="Matern_Covar",
    )


if __name__ == "__main__":
    args = get_args()

    main(args)
