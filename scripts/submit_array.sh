#!/bin/bash

## job name
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=41
#SBATCH --mem-per-cpu=6g
#SBATCH --time=01:00:00
#SBATCH --constraint=pascal
#SBATCH --comment="non-preemtable-cpu-job"
#SBATCH --array=0-39

start=${SLURM_ARRAY_TASK_ID}
BEGIN=0
let start=BEGIN+SLURM_ARRAY_TASK_ID*1
let potend=start+1
endlimit=40

end=$(( potend < endlimit ? potend : endlimit ))

for (( i=${start}; i < ${end}; i++ )); do
	/private/home/mshuaibi/.conda/envs/ocp/bin/python /private/home/mshuaibi/baselines/scripts/preprocess_point_energy_forces.py --data-path /checkpoint/electrocatalysis/relaxations/mapping/final_splits_with_adbulk_ids/S2EF_filtered/train.txt --out-path /checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/200k --num-workers 60 --size 200000 --ref-energy --tags --chunk $i & 
done
wait
