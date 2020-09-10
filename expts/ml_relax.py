import numpy as np
import os

import ase.io
import torch.nn
from ocpmodels.common.ase_utils import OCPCalculator, Relaxation, relax_eval
from ocpmodels.common.dft_eval import DFTeval
from ocpmodels.trainers import ForcesTrainer
from ase.optimize import BFGS
from tqdm import tqdm


def relax(model, results_path, fmax=0.01):
    # k = open("/private/home/mshuaibi/ocp-modeling/utils/val_sample.txt", "r")
    k = open("/checkpoint/mshuaibi/ocpdata_reset_07_13_20/eval_server_test.txt","r")
    traj_paths = k.read().splitlines()
    ocp_calc = OCPCalculator(model)
    import time
    times = []
    for path in tqdm(traj_paths[10:30]):
        t=time.time()
        randomid = os.path.basename(path)
        os.makedirs(os.path.join("relaxations", f"{results_path}/ml_results_debug/{randomid[:-5]}"), exist_ok=True)
        images = ase.io.Trajectory(path)
        starting_idx = 0
        init_structure = images[starting_idx].copy()
        optimizer = Relaxation(init_structure,
            f"relaxations/{results_path}/ml_results_debug/{randomid[:-5]}/ml_idx_{randomid}"
        )
        # optimizer.run(ocp_calc, steps=len(images)+100, fmax=fmax)
        optimizer.run(ocp_calc, steps=300, fmax=fmax)
        times.append(time.time()-t)
    print(np.mean(times))

def dfteval(relaxation_dir):
    evaluator = DFTeval(relaxation_dir)
    evaluator.write_input_files()
    # evaluator.queue_jobs()

def load_model(model_path):
    dataset = {
        "normalize_labels": False,
        "src": "/checkpoint/mshuaibi/ocpdata_reset_07_13_20/train/ocpdata_train_20Mv6"
    }
    val_dataset = {
        "src": "/checkpoint/mshuaibi/ocpdata_reset_07_13_20/val/ocpdata_val_200kv6"
    }

    model = {
        "name": "schnet",
        "hidden_channels": 1024,
        "num_filters": 256,
        "num_interactions": 3,
        "num_gaussians": 200,
        "cutoff": 6.0,
        "use_pbc": True,
    }

    task = {
            "dataset": "trajectory_lmdb",
            "descrption": "Regressing to energies and forces on ocp dataset",
            "labels": ["potential energy"],
            "metric": "mae",
            "type": "regression",
            "train_on_tags": "0-1-1",
            "eval_on_free_atoms": True,
    }
    optimizer = {
        "batch_size": 256,
        "eval_batch_size": 16,
        "num_workers": 64,
        "num_gpus": 2,
        "lr_initial": 0.0001,
        "scheduler": "lambda",
        "warmup_epochs": 3,
        "lr_milestones": [5, 8],
        "lr_gamma": 0.1,
        "warmup_factor": 0.2,
        "max_epochs": 30,
        "force_coefficient": 100,
        "criterion": torch.nn.L1Loss(),
    }
    trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=[dataset, val_dataset],
        optimizer=optimizer,
        identifier="ml_relax",
        print_every=100,
        is_debug=True,
        seed=1,
    )
    trainer.load_pretrained(model_path)

    return trainer

if __name__ == "__main__":
    checkpoint_path = "/private/home/mshuaibi/baselines/expts/ocp_expts/ocp20M_08_16/checkpoints/2020-08-16-21-53-06-ocp20Mv6_schnet_lr0.0001_ch1024_fltr256_gauss200_layrs3_pbc"
    # checkpoint_path = "/private/home/mshuaibi/baselines/expts/ocp_expts/ocp20M_2020-08-16-21-53-07-ocp20Mv6_schnet_lr0.0001_ch512_fltr256_gauss200_layrs3"

    results_path = os.path.basename(checkpoint_path)
    model = load_model(os.path.join(checkpoint_path, "checkpoint.pt"))

    relax(model=model,
          results_path=results_path,
          fmax=0,
    )
    # path = f"relaxations/{results_path}_test/ml_results/"
    # for i in tqdm(os.listdir(path)):
        # try:
            # dfteval(os.path.join(path, i))
        # except:
            # continue
