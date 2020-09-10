import argparse
import os
import subprocess
import sys


def modify_and_submit(
        data="ocp200k", model="schnet", layers=3, batch=256, coeff=100,
        channels=1024, num_gaussians=200, num_gpus=8, pbc=True,
        filters=256, lr=1e-4, val="is",
    ):

    with open("submit_model.sh", "r") as f:
        if model == "dimenet":
            name = f'{data}_{model}_l{layers}_b{batch}_c{coeff}_ch{channels}'
        else:
            name = f'{data}_{model}_lr{lr}_ch{channels}_fltr{filters}_layrs{layers}_val{val}'
        if pbc:
            name += "_pbc"
        contents = f.readlines()
        contents[3] = f"#SBATCH --job-name={name}\n"
        contents[4] = f"#SBATCH --output={name}.out\n"
        contents[9] = f"#SBATCH --gpus-per-node={num_gpus}\n"
        contents[10] = f"#SBATCH --cpus-per-task={num_gpus*10}\n"
        contents[11] = f"#SBATCH --mem-per-cpu=6g\n"
        contents[-3] = f"#SBATCH --time=48:00:00\n"

        contents[-1] = (
            "/private/home/mshuaibi/.conda/envs/ocp/bin/python "\
            "/private/home/mshuaibi/baselines/expts/ocp_bench.py "\
            f"--data {data} --model {model} --layers {layers} --batch {batch}"\
            f" --channels {channels} --coeff {coeff} --identifier {name}"\
            f" --num-gaussians {num_gaussians} --gpus {num_gpus}"\
            f" --filters {filters} --lr {lr} --batch-eval {int(batch/2)}"\
            f" --val {val}"

        )
        if pbc:
            contents[-1] += " --pbc"
    f.close()

    cwd = os.getcwd()
    os.makedirs(args.save_dir, exist_ok=True)
    os.chdir(args.save_dir)
    with open("temp_submit.sh", "w") as k:
        k.writelines(contents)
        k.close()
    cmd = ["sbatch", f"temp_submit.sh"]
    p = subprocess.Popen(cmd, cwd="./")
    p.wait()
    os.system("rm -rf temp_submit.sh")
    os.chdir(cwd)

def parameter_sweep(sweep_settings, model, data):
    for hp in sweep_settings.keys():
        for parameter in sweep_settings[hp]:
            p = {hp: parameter}
            modify_and_submit(model=model, data=data, **p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, help="Location to save model")
    args = parser.parse_args()

    cgcnn_sweep = {
        "layers": [3, 4, 5],
        "channels": [256, 768, 1024],
        "filters": [128, 512, 768],
        "pbc": [False],
    }

    schnet_sweep = {
            "val": [
                "is_100000",
                "is_200000",
                "is_300000",
                "is_400000",
                "is_600000",
                "is_800000",
                ]
        # "layers": [3, 4, 5],
        # "channels": [256, 768, 1024],
        # "filters": [128, 512, 768],
        # "pbc": [False],
    }

    dimenet_sweep = {
        "layers": [3, 4, 5],
        "channels": [256, 768, 1024],
        "filters": [128, 512, 768],
        "pbc": [False],
    }


    # parameter_sweep(schnet_sweep, model="schnet", data="ocp2M")
    modify_and_submit(data="ocp200k", model="schnet", pbc=True, channels=1024)
    modify_and_submit(data="ocp2M", model="schnet", pbc=True, channels=1024)
    modify_and_submit(data="ocp20M", model="schnet", pbc=True, channels=1024)
    modify_and_submit(data="ocpall", model="schnet", pbc=True, channels=1024)

    # modify_and_submit(data="ocp200k", model="cgcnn", pbc=True, channels=1024)
    # modify_and_submit(data="ocp2M", model="cgcnn", pbc=True, channels=1024)
    # modify_and_submit(data="ocp20M", model="cgcnn", pbc=True, channels=1024)

    # modify_and_submit(data="ocpall", model="schnet", pbc=True, channels=1024)
