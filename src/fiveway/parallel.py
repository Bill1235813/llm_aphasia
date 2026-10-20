"""Run several of the study's scripts at once in one job, sharing its GPU.

A small model's search leaves most of an 80 GB GPU idle, so (to fill a GPU
before adding more) one job runs, say, the learned-mask and
cross-entropy searches of both tasks of Pythia-1B side by side. Each command
is a script under src/fiveway/ with its arguments; its output goes to its own
log, logs/<job name>-<job id>-<k>.out, and the job fails if any command fails.
A requeued job starts every command again, and each resumes on its own.

Usage:
    GRES=gpu:1 bash slurm/fiveway_submit.sh learned-p1b 96G fiveway/parallel.py \\
        "mask_search.py --model pythia-1b --task cinderella" "cem_search.py --model pythia-1b --task cinderella"
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.paths import ROOT  # noqa: E402


def main():
    here = Path(__file__).resolve().parent
    stem = f"{os.environ.get('SLURM_JOB_NAME', 'parallel')}-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    (ROOT / "logs").mkdir(exist_ok=True)
    running = []
    for index, command in enumerate(sys.argv[1:]):
        script, *arguments = shlex.split(command)
        log_path = ROOT / "logs" / f"{stem}-{index}.out"
        with log_path.open("a") as log:
            process = subprocess.Popen([sys.executable, str(here / script), *arguments], stdout=log,
                                       stderr=subprocess.STDOUT)
        running.append((command, process))
        print(f"started {command!r}, log {log_path}", flush=True)
    # NOTE: waiting on every process and failing if any
    # failed: a bare wait for the first or the last would report success while
    # another one crashed.
    failed = False
    for command, process in running:
        code = process.wait()
        print(f"{command!r} exited with {code}", flush=True)
        failed = failed or code != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
