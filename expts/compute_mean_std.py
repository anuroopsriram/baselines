import sys
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocpmodels.datasets import TrajectoryLmdbDataset, data_list_collater


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="Path to training set")
    parser.add_argument("--type", default="free", help="Evaluate on free atoms only")
    args = parser.parse_args()

    dataset_config = { "src": args.src}

    dataset = TrajectoryLmdbDataset(dataset_config)
    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=data_list_collater,
        num_workers=64,
    )

    energies, forces = [], []

    for i, batch in tqdm(enumerate(data_loader)):
        energies.append(batch.y)
        if args.type == "free":
            free_atom_idx = batch.fixed ==0
            forces.append(batch.force[free_atom_idx])
        else:
            forces.append(batch.force)

    energies = torch.cat(energies, 0).view(-1, 1)
    forces = torch.cat(forces, 0)

    print("energy mean", energies.mean(0).item())
    print("energy std", energies.std(0).item())

    print("forces mean", [f.item() for f in forces.mean(0)])
    print("forces std", [f.item() for f in forces.std(0)])
