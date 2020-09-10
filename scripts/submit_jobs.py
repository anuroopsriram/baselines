import argparse
import os
import subprocess
import sys


def modify_and_submit(data, dirname, size):
    with open("submit_array.sh", "r") as f:
        contents = f.readlines()

        contents[-3] = (
            "\t/private/home/mshuaibi/.conda/envs/ocp/bin/python "\
            "/private/home/mshuaibi/baselines/scripts/preprocess_point_energy_forces.py "\
            "--data-path "\
            f"/checkpoint/electrocatalysis/relaxations/mapping/final_splits_with_adbulk_ids/S2EF_filteredv2/{data}.txt "\
            "--out-path "\
            f"/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/{data}/{dirname} "\
            f"--num-workers 40 --size {size} --ref-energy --tags --chunk $i &\n"

        )
    f.close()

    cwd = os.getcwd()
    with open("temp_submit.sh", "w") as k:
        k.writelines(contents)
        k.close()
    cmd = ["sbatch", f"temp_submit.sh"]
    p = subprocess.Popen(cmd, cwd="./")
    p.wait()
    os.system("rm -rf temp_submit.sh")
    os.chdir(cwd)

if __name__ == "__main__":
    modify_and_submit("train", "200k", 200000)
    modify_and_submit("train", "2M", 2000000)
    modify_and_submit("train", "20M", 20000000)
    modify_and_submit("train", "all", -1)
