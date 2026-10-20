#!/bin/bash
# Submit one job of the study.
#
#   bash slurm/fiveway_submit.sh <name> <memory> <script under src/> [script args...]
#   PARTITION=<partition> QOS=<qos> ...   # where to run (default: the cluster's default partition and QoS)
#   GRES=gpu:1 ...                        # the GPU request (default gpu:1)
#   GPUS=0 CPUS=32 ...                    # a CPU-only job
#   TIME=6:00:00 ...                      # time limit (default 12 h)
#   DEPEND=<jobid> ...                    # start only after that job succeeded
#   DEPEND=<id>:<id> DEPEND_KIND=afterany ...   # ... or after those ended, however
#   FROM_HEAD=1 ...                       # snapshot src/ as committed, without uncommitted edits
#   ENV_SCRIPT=<file> ...                 # sourced by the job before .venv is activated (caches, module loads)
#
# Every job runs a snapshot of src/ with the commit and any uncommitted diff, so every result traces back to the
# exact code that produced it, and editing src/ never changes a pending or requeued job.
set -e
cd "$(dirname "$0")/.."
NAME=$1
MEMORY=$2
shift 2
SNAPSHOT=$PWD/runs/code/$(date +%Y%m%d-%H%M%S)-$NAME
mkdir -p $SNAPSHOT logs
if [ "${FROM_HEAD:-0}" = "1" ]; then
  git archive HEAD src | tar -x -C $SNAPSHOT
  echo "commit=$(git rev-parse HEAD) uncommitted_files=0 (src from HEAD)" > $SNAPSHOT/COMMIT
else
  rsync -a --exclude __pycache__ src/ $SNAPSHOT/src/
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo none) uncommitted_files=$(git status --porcelain 2>/dev/null | wc -l)" > $SNAPSHOT/COMMIT
  git diff HEAD > $SNAPSHOT/uncommitted.diff 2>/dev/null || true
fi
RESOURCES=(--nodes=1 --cpus-per-task=${CPUS:-4} --mem=$MEMORY --time=${TIME:-12:00:00})
if [ -n "$PARTITION" ]; then RESOURCES+=(-p "$PARTITION"); fi
if [ -n "$QOS" ]; then RESOURCES+=(--qos="$QOS"); fi
if [ "${GPUS:-1}" != "0" ]; then RESOURCES+=(--gres=${GRES:-gpu:1}); fi
if [ -n "$DEPEND" ]; then RESOURCES+=(--dependency=${DEPEND_KIND:-afterok}:$DEPEND); fi
sbatch --export=ALL "${RESOURCES[@]}" -J aphasia-$NAME slurm/fiveway.sbatch $SNAPSHOT "$@"
