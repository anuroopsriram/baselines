import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocpmodels.datasets import TrajectoryLmdbDataset, data_list_collater
from ocpmodels.common.meter import mae


if __name__ == "__main__":

    # OCP 200k
    # energy_mean = -0.7586358785629272
    # force_mean = [5.679600144503638e-05, -0.00021635270968545228, 0.004137882962822914]

    # OCP 2M
    energy_mean = -0.7589851021766663
    force_mean = [5.074080763733946e-05, -0.00013697720714844763, 0.0042003486305475235]

    # val_dataset_config = {
        # "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/is_1M"
    # }
    val_dataset_config = {
        "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/oos_ads_1M"
    }
    # val_dataset_config = {
        # "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/oos_ads_bulk_1M"
    # }
    # val_dataset_config = {
        # "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/oos_bulk_1M"
    # }


    dataset = TrajectoryLmdbDataset(val_dataset_config)
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
        free_atom_idx = batch.fixed == 0
        forces.append(batch.force[free_atom_idx])

    energies = torch.cat(energies, 0).view(-1, 1)
    forces = torch.cat(forces, 0)

    print("potential energy MAE", mae(energies, torch.tensor(energy_mean)))
    print("forces MAE", mae(forces, torch.tensor(force_mean)))
