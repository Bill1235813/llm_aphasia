#!/bin/bash
# One job of the study (src/fiveway/). Submit it with slurm/fiveway_submit.sh, which snapshots src/ first and
# puts the resources on the sbatch command line.
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
# NOTE: --open-mode=append keeps the earlier part of the log when a preempted job is requeued and this script
# runs again from the top; every script resumes from its own cache of answers.
set -e
SNAPSHOT=$1
shift
cd "$SLURM_SUBMIT_DIR"                          # the repository
if [ -n "$ENV_SCRIPT" ]; then source "$ENV_SCRIPT"; fi   # optional: cache locations, module loads
source .venv/bin/activate                       # built with `uv sync`
# The job runs the code snapshot taken at submission, not the working tree; data and result paths still point
# at the repository (src/fiveway/paths.py).
export APHASIA_ROOT=$PWD PYTHONPATH=$SNAPSHOT/src PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TOKENIZERS_PARALLELISM=false
echo "=== $(date) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
echo "=== snapshot=$SNAPSHOT $(cat $SNAPSHOT/COMMIT)"
nvidia-smi -L || true
python $SNAPSHOT/src/"$@"
