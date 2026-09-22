#!/bin/bash 
#SBATCH -A NAISS2026-4-912-gpu
#SBATCH -t 12:00:00
#SBATCH -J stride-pipeline
#SBATCH --chdir=/nobackup/proj/disk/hclimai/personal/fuxing/log/log_stride/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -p gpu
#SBATCH -n 1
###SBATCH -c 48
#SBATCH --cpus-per-task=32
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=400G

###SBATCH --ntasks-per-node=1
###SBATCH --cpus-per-task=16
###SBATCH --gpus-per-node=1
###SBATCH --mem-per-cpu=10G 


set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch bash/run_pipeline.sh <pipeline_config.yaml> [--dry-run]"
  exit 1
fi

STRIDE_RUNS=/nobackup/proj/disk/hclimai/shared/STRIDE_RUNS
export STRIDE_RUNS

PIPELINE_CONFIG="$1"
shift || true

EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo
echo "========================="
echo "STRIDE Slurm pipeline run"
echo "========================="
echo "Repository root: $REPO_ROOT"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "Python: $(command -v python || true)"
echo "Working dir: $(pwd)"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo

current_date_time="`date`";
echo The run starts from $current_date_time
#echo Check https://job.c3se.chalmers.se/alvis/$SLURM_JOB_ID for GPU usage.

#export HDF5_USE_FILE_LOCKING=FALSE
#export TF_GPU_ALLOCATOR=cuda_malloc_async
##export CUDA_VISIBLE_DEVICES=1 
#export TF_DETERMINISTIC_OPS=0
#export TF_FORCE_GPU_ALLOW_GROWTH=true
#ecinteractive -g

DOMAIN='norcp'
#DOMAIN='TestDomain'
VARIABLE='tas'

echo 'domain is' ${DOMAIN}
set -exu 

module --force purge
#interactive -p gpu --gpus 1 -A NAISS2026-4-912-gpu
###module load GPU/buildenv-nvhpc/25.9-cu13.0
#module load GPU/Python/3.13.5-bare-gcc-2025b-eb
#python3 -m venv --system-site-packages stride
source $HOME/venvs/stride/bin/activate
#pip install pyyaml (pyyaml-6.0.3)
#pip install torch (torch-2.12.0)
#pip install numpy (numpy-2.4.6)
#pip install pandas (pandas-3.0.3)
#pip install xarray (xarray-2026.4.0)
#pip install matplotlib (matplotlib-3.10.9)
#pip install netcdf4 (netcdf4-1.7.4)

cd $HOME/STRIDE
python cli/launch_pipeline.py --config "$PIPELINE_CONFIG" "${EXTRA_ARGS[@]}"

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0
#sbatch bash/run_pipeline_arrhenius.sh  configs/experiments/pipeline_norcp_arrhenius.yaml

