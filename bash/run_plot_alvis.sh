#!/bin/bash 
#SBATCH -A NAISS2025-1-11  -p alvis
#SBATCH -N 1 
#SBATCH --gpus-per-node=A40:1
###SBATCH --gpus-per-node=A100:1 
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -J stride-plot
#SBATCH --chdir=/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/log/log_stats/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out

set -euo pipefail

#if [ "$#" -lt 1 ]; then
#  echo "Usage: sbatch bash/run_plot.sh <file.npz> [--dry-run]"
#  exit 1
#fi

#STRIDE_RUNS=/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/STRIDE_RUNS
#export STRIDE_RUNS

#PIPELINE_CONFIG="$1"
#shift || true
#
#EXTRA_ARGS=("$@")
#
#REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#cd "$REPO_ROOT"

echo
echo "========================="
echo "STRIDE Slurm pipeline run"
echo "========================="
#echo "Repository root: $REPO_ROOT"
#echo "Pipeline config: $PIPELINE_CONFIG"
echo "Python: $(command -v python || true)"
echo "Working dir: $(pwd)"
#echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

current_date_time="`date`";
echo The run starts from $current_date_time
echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

#ecinteractive -g

DOMAIN='norcp'
#DOMAIN='TestDomain'
VARIABLE='tas'

echo 'domain is' ${DOMAIN}
set -exu 

module --force purge
#module load virtualenv/20.26.2-GCCcore-13.3.0
#module load Python/3.12.3-GCCcore-13.3.0
#module load netcdf4-python/1.7.1.post2-foss-2024a
module load virtualenv/20.23.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load netcdf4-python/1.6.4-foss-2023a
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load xarray/2023.9.0-gfbf-2023a
module load PyYAML/6.0-GCCcore-12.3.0
source $HOME/venvs/climulatorscore/bin/activate

#npz_file='/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/STRIDE_RUNS/pipeline_norcp/generation/samples/batch_0000_case_00/ensemble_mean.npz'
npz_file='/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/STRIDE_RUNS/pipeline_norcp/generation/samples/batch_0000_case_00/ensemble_members.npz'
#npz_file='/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/STRIDE_RUNS/pipeline_norcp/training/training_generation_preview/training_preview_epoch_0120.npz'

cd $HOME/STRIDE
python test_scripts/plot_npz.py $npz_file --out /mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/STRIDE_RUNS/pipeline_norcp/plots/

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0
