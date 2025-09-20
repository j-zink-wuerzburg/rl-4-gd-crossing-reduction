#!/bin/bash
#SBATCH --job-name=run-graphs
#SBATCH --nodelist=gpu-intel-pvc     # adapt to your cluster
#SBATCH --time=24:00:00
#SBATCH --output=runner.%A_%a.out
#SBATCH --error=runner.%A_%a.out
#SBATCH --ntasks=1                    # one MPI task per array element
#SBATCH --cpus-per-task=1             # … 1 CPU; raise if runner.py uses more


#SBATCH --array=1-50                 # ← one task for each “split k/50”



python ./runner.py  ../../sng/graphs/rome_filtered/splits/data/ -a rlgc  -w 1 -r 1 -t 900 -l ../../sng/graphs/rome_filtered/splits/test1000.txt --split ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}

## python ./runner.py  ../../sng/graphs/extended_BA_filtered/data/ -a rllc  -w 1 -r 1 -t 900 -l ../../sng/graphs/extended_BA_filtered/test500.txt --split ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}


### sbatch --array=1-200%50 --dependency=afterany:165299 run_splits.bash