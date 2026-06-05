#!/usr/bin/env bash

# Called by cluster_interface_eureka.sh on the cluster login node.
# $1 = CLUSTER_EUREKA_TS_DIR (IsaacLabEureka timestamped root on cluster)
# $2 = container profile (e.g. isaac-lab-base)
# $3+ = job args passed through to the python script

### MODIFY SLURM PARAMETERS FOR YOUR JOB ###
cat <<EOT > job_eureka.sh
#!/bin/bash

#SBATCH -A ehpc565
#SBATCH -p acc
#SBATCH -q acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=dunkakaslana@gmail.com
#SBATCH --job-name="eureka-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH -o /gpfs/scratch/ehpc565/IsaacLabEureka/logs/eureka-%j.out
#SBATCH -e /gpfs/scratch/ehpc565/IsaacLabEureka/logs/eureka-%j.err

bash "$1/IsaacLab/docker/cluster/run_singularity_eureka.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job_eureka.sh
rm job_eureka.sh
