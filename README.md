# Language Impairment as a Lens onto Large Language Models

Muxuan He<sup>1†</sup>, Wang Bill Zhu<sup>2†</sup>, Yue Chen<sup>1†</sup>, Elsi Kaiser<sup>1</sup>, Weizhi Du<sup>3</sup>,
Yao Du<sup>1</sup>, Samuel Nastase<sup>1</sup>, Yanjun Gao<sup>4</sup>

<sup>1</sup>University of Southern California, <sup>2</sup>University of California, Berkeley,
<sup>3</sup>Independent Researcher, <sup>4</sup>University of Colorado Anschutz Medical Campus
(<sup>†</sup>corresponding authors)

**[Paper](https://arxiv.org/abs/2610.28992)** · **[Project page](https://llm-aphasia.github.io/)**

Neuropsychology maps how the brain produces language by studying how language breaks down after focal damage, as
in aphasia. This repository holds the code of our study, which uses the same lens on large language models. It
induces aphasia-like breakdowns in 11 models (GPT-2 XL, Llama-2-7B/13B-chat, Pythia-1B/1.4B/2.8B/6.9B,
Qwen3.5-0.8B/2B/4B/9B) in two complementary ways, and judges them with a five-way clinical classifier:

- **Lesioning** acts on the model's structure. CLAMP (Classifier-guided Lesion Accumulation with Minimal Pruning)
  knocks out the attention heads that make the model's narrative aphasic, Broca-like or Wernicke-like
  (`src/fiveway/grown_heads.py` and `src/fiveway/em_attention.py`). Eight baseline lesion methods are included for
  comparison.
- **Steering** acts on the model's representations. It adds a direction computed from human transcripts to the
  residual stream: the mean hidden state of speakers with Broca's or Wernicke's aphasia minus that of control
  speakers (`src/fiveway/steer_*.py`).

## What is not included, and why

The transcripts of people with aphasia come from [AphasiaBank](https://aphasia.talkbank.org/), whose terms do not
allow redistribution. This release therefore contains **code only**:

- no AphasiaBank transcripts, and nothing derived from them that could reproduce them;
- no trained weights of the five-way classifier, which was trained on these transcripts (its code is in
  `classifier/`);
- no model scrambles (the 259 degenerate model answers of the classifier's fifth class), no expert annotations of
  the steered answers, and no results, logs or caches.

To rerun the study, request access to AphasiaBank through TalkBank, then build the data files and train the
classifier as described below. Without the withheld files, three parts cannot be redone exactly:

- the classifier's Model_scrambles class must be trained on degenerate outputs of your own;
- the comparison with the experts (`annotation_agreement.py` and the expert parts of `results_numbers.py`) needs
  expert labels of your own;
- a few tests and scripts compare against stored outputs of an earlier transformer_lens version of the pipeline,
  which is not included: `test_engine.py`'s reproduction step, `test_llama_circuit.py`, `test_divergence.py`,
  `test_patching_benchmarks.py`, `score_stored_baselines.py` and `src/interp/tab_judge.py`'s command line. Some
  docstrings refer to that earlier version (its experiments `exp00` to `exp07`, and `src/steering/`).

## Layout

```
src/fiveway/                the study: engine, searches, steering, analyses and tests; README.md describes each script
src/interp/                 the story prompts (generate.py) and the TAB feature-judge prompt (tab_judge.py)
classifier/                 the five-way classifier: code only (README.md); its data format is in data/README.md
slurm/                      generic Slurm wrappers: fiveway_submit.sh snapshots src/ and submits fiveway.sbatch
pyproject.toml, uv.lock     the exact Python environment
```

## Setup

The code was run with Python 3.13, PyTorch 2.7.1 (CUDA 12.6) and Transformers 5.5.4; `uv.lock` pins every package,
and its PyTorch build has wheels for Python 3.11 to 3.13 only.

```bash
uv sync --python 3.13        # creates .venv from uv.lock
source .venv/bin/activate
```

- **Models.** All 11 models load from the Hugging Face Hub (`src/fiveway/models.py` lists their identifiers). The
  Llama-2 models are gated: accept their licence on the Hub and log in with `hf auth login`.
- **GPUs.** Every model runs in 32-bit floating point, so the searches of the larger models need an 80 GB GPU. We
  used NVIDIA H200 and A100 (40 and 80 GB) GPUs, and older TITAN and RTX 2080 GPUs for small jobs, about 236 GPU
  hours in all.
- **LLM judge.** `src/fiveway/llm_judge.py` queries DeepSeek-V4.1-Flash through DeepSeek's API. It reads the key from
  `DEEPSEEK_API_KEY` (or `~/.deepseek_api_key`) and caches every reply.
- **MMLU.** `src/fiveway/benchmarks.py` reads two MMLU subjects of `cais/mmlu` from the local Hugging Face cache and
  never downloads them. Fetch them once:

  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download('cais/mmlu', repo_type='dataset', allow_patterns=['global_facts/*', 'high_school_world_history/*'])"
  ```

  The 20 items of the Text Aphasia Battery (TAB) that we use are written out in the same file.

## Data and classifier

1. **Transcripts.** Build `classifier/data/{cinderella,sandwich}/transcripts.json` from AphasiaBank's Cinderella and
   sandwich narratives, in the format of `classifier/data/README.md`: one object per transcript with `File_DB`,
   `Group` (`Control`, `Anomic`, `Conduction`, `Broca` or `Wernicke`), `text_response` and AphasiaBank's metadata
   columns (`Corpus`, `Custom_field`, ...).
2. **Model scrambles.** The classifier's fifth class needs degenerate model outputs labelled `Model_scrambles`, with
   the task's name (`cinderella` or `sandwich`) in their `File_DB`. Put them, together with both tasks' transcripts,
   in `classifier/data/cindsand/transcripts.json`, and alone in `classifier/data/model_scrambles/transcripts.json`.
3. **Classifier.** Train the 20-seed ensemble with the package's own scripts:

   ```bash
   cd classifier
   python src/build_features.py
   python src/train.py          # writes models/cindsand/seed_*
   ```

   `src/fiveway/scorer.py` loads the 20 seeds from `classifier/models/cindsand/` and reproduces the package's
   `predict.py`; `src/fiveway/test_scorer.py` checks it against `predict.py`'s scores of a CSV of example texts.

`src/fiveway/paths.py` finds everything relative to the repository; `APHASIA_ROOT` and `FIVEWAY_CLASSIFIER_DIR`
override the repository and the classifier's folder.

## Running the study

Every script under `src/fiveway/` is a command-line program whose docstring gives its usage. It can run directly,

```bash
python src/fiveway/grown_heads.py --model pythia-1b --task cinderella
```

or as a Slurm job, which snapshots `src/` so that later edits cannot change a pending or requeued job:

```bash
PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh grown-p1b-cind 48G \
    fiveway/grown_heads.py --model pythia-1b --task cinderella
```

`ENV_SCRIPT=<file>` makes the job source that file first (for cache locations or module loads). Every search caches
each answer it generates under `runs/`, so a preempted job resumes where it stopped, and writes its results under
`results/fiveway/`.

The scripts build on each other's results, in this order:

1. `em_baseline.py` (Param, and the top-1% weight lesions that later spaces reuse) and `search.py` (the layer search;
   its first stage also gives the single-head answers that CLAMP ranks for the seven older models);
2. `em_search.py`, then `em_attention.py`, `mask_search.py` and `cem_search.py`; `grown_heads.py` (CLAMP on Qwen3.5,
   and Head-grown-cos with `--kinds cosine`);
3. `patching.py`, then `benchmarks.py` under patching;
4. `steer_collect.py`, then `steer_analyze.py` and `steer_generate.py`, then `grown_rescue.py`;
5. `steer_annotate.py` and `annotation_csv.py` for the expert sheets, and `llm_judge.py`;
6. the analyses and summaries: `final_counts.py`, `results_numbers.py`, `subtype_overlap.py`, `subtype_cosine.py`,
   `transcript_tsne.py` and `output_tsne.py`.

| Paper | Scripts |
|---|---|
| Lesion methods (Section 4.1, Table 2, Appendix D) | `search.py`, `em_attention.py` (CLAMP for the seven older models as space `2b`, Param-C as space `5`, and the grown MLP and attention spaces), `grown_heads.py` (CLAMP for Qwen3.5; Head-grown-cos), `em_baseline.py` (Param), `mask_search.py` (Param-mask, Head-mask and their hard versions), `cem_search.py` (Head-CEM); `final_counts.py` counts the settings of Table 2 |
| CLAMP across 11 models (Figure 4) | `grown_heads.py`, `em_attention.py` |
| Beyond the narrative (Table 3, Appendix F) | the lesion scripts (`grown_heads.py`, `em_attention.py`) also answer the other narrative, the TAB items and MMLU through `benchmarks.py`; `llm_judge.py` rates the TAB features |
| Activation patching and shared heads (Appendix G) | `patching.py`, `run_sequence.py`; overlap of the Broca and Wernicke lesions: `subtype_overlap.py`; the two tasks' lesions: `task_overlap.py` |
| Steering (Section 4.2, Appendix H) | `steer_collect.py` (residual means and steering vectors), `steer_analyze.py` (probes), `steer_generate.py` (the sweep and the other steering experiments), `subtype_cosine.py` (Broca and Wernicke vectors) |
| Human evaluation (Table 4, Figure 5, Appendix I) | `steer_annotate.py` (the steered answers), `annotation_csv.py` (the annotation sheets), `annotation_agreement.py` (agreement with the experts), `llm_judge.py` (the judge's labels) |
| Steering back (Appendix J) | `grown_rescue.py` |
| Transcripts and answers in text space (Figures 6 and 8, Appendices B and H) | `transcript_tsne.py`, `output_tsne.py` |
| The numbers of Section 5 | `results_numbers.py` |

`src/fiveway/README.md` describes the engine, the decoding rules and the validation of each component. The tests
(`src/fiveway/test_*.py`) check the engine, the scorer and the searches against hooks written from scratch; the ones
marked CPU in their docstrings need no GPU, and some need the transcripts or the earlier pipeline's outputs (above).

## License

The code is released under the MIT License (`LICENSE`). The 20 Text Aphasia Battery items written out in
`src/fiveway/benchmarks.py` are from Roll et al. (2026, arXiv:2511.20507) and remain under their authors' terms; the
models, AphasiaBank and MMLU keep their own licences.

## Citation

```bibtex
@misc{he2026language,
  title         = {Language Impairment as a Lens onto Large Language Models},
  author        = {He, Muxuan and Zhu, Wang Bill and Chen, Yue and Kaiser, Elsi and
                   Du, Weizhi and Du, Yao and Nastase, Samuel and Gao, Yanjun},
  year          = {2026},
  eprint        = {2610.28992},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.28992}
}
```
