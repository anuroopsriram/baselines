import os
import glob
import copy
import submitit
import json
import torch

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
)

def main(config):
    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "simple"))(
        task=config["task"],
        model=config["model_attributes"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        run_dir=config.get("run_dir", "./"),
        is_debug=True,
        is_vis=config.get("is_vis", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"]
    )
    trainer.load_pretrained(config["model_path"], ddp_to_dp=False)
    trainer.validate(epoch=0)
    distutils.synchronize()

def distributed_main(config):
    distutils.setup(config)
    main(config)
    distutils.cleanup()

if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()
    slurm_job_id = "29765367_1"
    trial = "dimenet_200k"
    log_path = glob.glob(f"logs/{trial}/{slurm_job_id}/*.out")[0]
    log = open(log_path, "r").read().splitlines()
    for i in log:
        if "checkpoint_dir" in i:
            checkpoint_path = os.path.join(log[4].strip().split()[1], "checkpoint.pt")
            break
    assert os.path.isfile(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]

    config["trainer"] = "dist_forces"
    config["model_attributes"].update({"name": config["model"]})
    config["optim"]["eval_batch_size"] = 1
    config["identifier"] = "eval"
    config["model_path"] = checkpoint_path
    if args.distributed:
        config["local_rank"] = args.local_rank
        config["distributed_port"] = args.distributed_port
        config["world_size"] = args.num_nodes * args.num_gpus
        config["distributed_backend"] = args.distributed_backend
        config["submit"] = args.submit

    val_configs = []
    for val_set in ["is", "oos_ads", "oos_ads_bulk", "oos_bulk"]:
        val_cfg = copy.deepcopy(config)
        dataset = val_cfg["dataset"]
        dataset[1] = {
            "src": f"/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/val/{val_set}_1M",
            "normalize_labels": False,
        }
        val_cfg["dataset"] = dataset
        val_configs.append(val_cfg)

    if args.submit:  # Run on cluster
        executor = submitit.AutoExecutor(folder=args.logdir / "%j")
        executor.update_parameters(
            name=config["identifier"],
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
            comment="ICLR 09/21 - Intern leaving 09/11"
        )
        if args.distributed:
            jobs = executor.map_array(distributed_main, val_configs)
        else:
            jobs = executor.map_array(main, val_configs)
        print("Submitted jobs:", ", ".join([job.job_id for job in jobs]))
        log_file = save_experiment_log(args, jobs, val_configs)
        print(f"Experiment log saved to: {log_file}")
