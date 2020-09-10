from torch import nn
import os
import sys
import subprocess
import argparse
from ocpmodels.trainers import ForcesTrainer
from ocpmodels.datasets import TrajectoryLmdbDataset, data_list_collater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ocp200k", help="Dataset to use")
    parser.add_argument("--val", type=str, default="is", help="Val set to use")
    parser.add_argument("--model", type=str, default="schnet", help="Model to use")
    parser.add_argument("--num-gaussians", type=int, default=200, help="Number of gaussians to use")
    parser.add_argument("--layers", type=int, default=3, help="Number of MP layers")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="# of epochs")
    parser.add_argument("--coeff", type=int, default=100, help="Force coefficient")
    parser.add_argument("--identifier", type=str, help="Model identifier")
    parser.add_argument("--channels", type=int, default=256, help="# of hidden channels")
    parser.add_argument("--filters", type=int, default=256, help="# of filters")
    parser.add_argument("--loss", type=str, default="mae", help="Loss function to be used")
    parser.add_argument("--gpus", type=int, default=2, help="# of GPUs")
    parser.add_argument("--num-workers", type=int, default=64, help="# of workers")
    parser.add_argument("--batch-eval", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--eval-every", type=int, default=-1, help="Evaluate every N steps")
    parser.add_argument("--pbc", action="store_true", help="PBC on/off")
    parser.add_argument("--debug", action="store_true", help="debug on/off")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Cutoff radius")
    args = parser.parse_args()

    train_dataset_config, val_dataset_config = retrieve_data(args.data, args.val)
    task = get_task()

    if args.model == "dimenet":
        model = {
            "name": args.model,
            "hidden_channels": args.channels,
            "num_blocks": args.layers,
            "num_radial": 3,
            "num_spherical": 4,
            "num_before_skip": 1,
            "num_after_skip": 1,
            "num_output_layers": 1,
            "max_angles_per_image": 30000,
            "cutoff": args.cutoff,
            "use_pbc": args.pbc,
            "regress_forces": True,
        }
    elif args.model == "cgcnn":
        model = {
            "name": args.model,
            "atom_embedding_size": args.filters,
            "num_graph_conv_layers": args.layers,
            "fc_feat_size": args.channels,
            "num_fc_layers": 4,
            "num_gaussians": args.num_gaussians,
            "use_pbc": args.pbc,
        }
    else:
        model = {
            "name": args.model,
            "hidden_channels": args.channels,
            "num_filters": args.filters,
            "num_interactions": args.layers,
            "num_gaussians": args.num_gaussians,
            "cutoff": args.cutoff,
            "use_pbc": args.pbc,
        }


    if args.loss == "mse":
        criterion = nn.MSELoss()
    else: criterion = nn.L1Loss()

    optimizer = {
        "batch_size": args.batch,
        "eval_batch_size": args.batch_eval,
        "num_workers": args.num_workers,
        "num_gpus": args.gpus,
        "lr_initial": args.lr,
        "warmup_epochs": 3,
        "lr_milestones": [5, 8, 13],
        "lr_gamma": 0.1,
        "warmup_factor": 0.2,
        "max_epochs": args.epochs,
        "force_coefficient": args.coeff,
        "criterion": criterion,
        "eval_every": args.eval_every,
    }

    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=[train_dataset_config, val_dataset_config],
        optimizer=optimizer,
        identifier=args.identifier,
        print_every=100,
        is_debug=args.debug,
        seed=1,
    )

    trainer.train()

def retrieve_data(data, val):
    if data == "ocp200k":
        train_dataset_config = {
            "normalize_labels": False,
            "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/200k"
        }
    elif data == "ocp200kold":
        train_dataset_config = {
            "normalize_labels": False,
            "src": "/checkpoint/mshuaibi/ocpdata_reset_07_13_20/train/ocpdata_train_200kv2",
        }

    elif data == "ocp2M":
        train_dataset_config = {
            "normalize_labels": False,
            "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/2M"
        }
    elif data == "ocp20M":
        train_dataset_config = {
            "normalize_labels": False,
            "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/20M"
        }
    elif data == "ocpall":
        train_dataset_config = {
            "normalize_labels": False,
            "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/all"
        }

    val_dataset_config = {
            "src": f"/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/{val}"
        }

    return train_dataset_config, val_dataset_config

def get_task():
    task = {
        "dataset": "trajectory_lmdb",
        "description": "Regressing to energies and forces on ocp dataset",
        "labels": ["potential energy"],
        "metric": "mae",
        "type": "regression",
        "eval_on_free_atoms": True,
        "train_on_free_atoms": True,
    }
    return task


if __name__ == "__main__":
    main()
