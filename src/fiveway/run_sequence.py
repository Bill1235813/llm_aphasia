"""Run several five-way-study scripts one after another in one job.

Patching and the benchmarks run once per circuit, and a model has up to six
circuits; one job per model that runs them in turn waits in the queue once
instead of twelve times. Each line of the commands file is a script under
src/ and its arguments, for example

    fiveway/patching.py --model pythia-1b --task cinderella --objective A
    fiveway/benchmarks.py --model pythia-1b --circuit-task cinderella --objective A

Every command runs in its own process, so GPU memory is freed between them,
and a failing command is reported without stopping the rest. Each script
resumes from its own outputs, so a requeued job repeats nothing finished.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=6:00:00 bash slurm/fiveway_submit.sh patch-bench-p1b 32G \\
        fiveway/run_sequence.py runs/fiveway/sequences/pythia-1b.txt
"""

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]


def main():
    commands = [line.split() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    failed = []
    for number, command in enumerate(commands, 1):
        start = time.time()
        print(f"=== [{number}/{len(commands)}] {' '.join(command)}", flush=True)
        result = subprocess.run([sys.executable, str(SRC / command[0])] + command[1:])
        print(f"=== exit {result.returncode} after {(time.time() - start) / 60:.1f} min", flush=True)
        if result.returncode != 0:
            failed.append(" ".join(command))
    print(f"=== done; {len(failed)} failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
