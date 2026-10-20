# The five-way study

The knockout search, steering and ablations of the study, with the five-way
`cindsand` classifier, on 11 generators from four families. An earlier
transformer_lens version of the pipeline (not included; `src/interp/` keeps
its prompts) generated one answer at a time; this package uses plain Hugging Face models
with one intervention per batch element: knocking out each of Llama-2-13B's
1,600 heads alone, and the 240 top-k sets, takes half an hour on one A100,
and the whole three-stage search of one task, all three objectives, five to
ten hours.

## Classifier

`classifier/` (code only in this release: its training data are AphasiaBank
transcripts, so neither the data nor the trained weights are included; see the
top-level README) is a 20-seed ensemble of BGE-large with LoRA
plus 14 text features, trained on the Cinderella and sandwich narratives
together and on 259 degenerate outputs of ablated models. It tells five classes
apart: Control, Conduction/Anomic, Broca, Wernicke and Model_scrambles. It has
no surprisal features, so one classifier scores every generator.
`scorer.py` reproduces the package's own `predict.py` to 4e-8
(`test_scorer.py`). Two properties of the classifier matter for reading the
results:

- The package's similarity feature `cossim_delta_scrambles` compares against
  the label "Model scrambles" while the corpus says "Model_scrambles", so that
  group is always empty and the feature is the mean similarity to Control. The
  classifier was trained this way, so the scorer keeps it.
- Model_scrambles absorbs more than garbage: fluent Llama-2-chat narratives get
  0.2 to 0.5, and a fluent refusal-like answer of Pythia-2.8B ("I don't know how
  to tell it, but if you want a good idea ... read ...") gets 0.998.

The three objectives of the search are read off the five probabilities:
**A** P(aphasia) = P(Conduction/Anomic) + P(Broca) + P(Wernicke) against
Control and Model_scrambles, **B** P(Broca), **C** P(Wernicke); an objective
works above 0.5.

## Generators and decoding

| family | models |
|---|---|
| GPT-2 | GPT-2 XL |
| Llama-2-chat | 7B, 13B |
| Pythia | 1B, 1.4B, 2.8B, 6.9B |
| Qwen3.5 | 0.8B, 2B, 4B, 9B |

All in float32. Prompts are the earlier study's (`interp/generate.py`): the
Cinderella gate question and story request, and the sandwich request; chat
template for Llama-2-chat, bare `Question: ... Answer:` for the base models.

- **Decoding: greedy with a repetition penalty of 1.25, for every model.**
  Under plain greedy decoding the base models' unablated answers loop, and the
  five-way classifier calls those loops Model_scrambles (P about 0.998 for
  GPT-2 XL and all Pythia models on Cinderella, `results/fiveway/stored_baselines.json`),
  so a search would start from degenerate text.
- **Base-model answers are cut before the first `Question:` they write
  themselves**: they keep writing question-and-answer turns of their own.
- **BOS**: GPT-2 and Llama prompts start with the BOS token and Pythia prompts
  do not, as transformer_lens tokenized them.
- **The earlier pipeline's rep1.25 is not the standard one.** It applied the
  penalty to transformer_lens logits, which are mean-centered per position
  (`center_unembed`), and the penalty is not shift-invariant. This package uses
  the standard penalty on raw logits, so its rep1.25 answers differ from the
  stored ones (`test_divergence.py` shows the centered logits pick the stored
  token).

## Validation

- **Exact reproduction.** With plain greedy decoding the engine regenerates the
  earlier study's stored unablated answers character for character for GPT-2 XL
  and Pythia-1B (both tasks, 180 tokens) and for Llama-2-7B-chat, including the
  stories of the three-head knockout set's search path (L0H25, +L1H8, +L1H29, and
  two more steps; `test_llama_circuit.py`). Zeroing a head's slice of the
  attention output projection's input is therefore the same knockout as
  transformer_lens's `hook_z` for Llama. For GPT-2 and Pythia, which have value
  biases, it removes slightly more: transformer_lens folds each head's value
  bias into the shared output bias, so its zeroed `hook_z` still writes W_O b_V,
  and the earlier study's GPT-2 and Pythia knockouts kept that constant.
- **Batch invariance.** Every batch is padded to the model's fixed batch size,
  so an intervention's logits are bit-identical wherever it sits in a batch and
  whatever else is in it (`test_engine.py`).

## The search (`search.py`)

For each model, task and objective, three stages, stopping at the first that
works: (1) single-layer heads: every head alone, then each layer's top-k heads
together, k in 1, 2, 3, 5, 8, 12; (2) a forward sweep over the layers that
greedily adds heads on top of everything already knocked out (at most 4 picks
per layer, stopping after 2 picks in a row without gain, ties committed, as
the earlier exp05); (3) the same sweep with 256-neuron MLP blocks after each
layer's heads. A knockout counts only if the answer is at least 20 characters
long, not counting invisible format characters (some Llama-2-13B knockouts
write a few letters and then hundreds of zero-width joiners, which the
classifier calls Broca), and, for Cinderella, the model does not answer "no"
to the gate question while the unablated model answers "yes" (Pythia-6.9B
answers "no" unablated, so its gate is off). A
knockout that works is pruned to a minimal set: components are dropped one per
round, the one whose removal keeps the objective highest, while it stays above
0.5. `--objective-kind cosine` runs the same search with the classifier-free
objective of the cos-sim ablation (mean MiniLM cosine to the target group minus
mean cosine to Control, working above 0).

## The other analyses

| question | script | what it does |
|---|---|---|
| steering | `steer_collect.py`, `steer_analyze.py`, `steer_generate.py` | Mean residual stream of every human transcript (each text once, as the model reads it); probes and difference-of-means vectors per layer (A: all four aphasia groups minus Control; B: Broca minus Control; C: Wernicke minus Control; AC: Anomic and Conduction minus Control; B and C with their component along A removed), centred over d_model for the LayerNorm models; speaker-grouped folds, cross-task transfer, severity (WAB AQ), PCA, logit lens. Then the vectors added to the model's own answer at 8 layers and relative strengths r up to 4 (r = added norm over mean residual norm), against random directions, the other task's vector, subtype vectors, continuations of human transcripts, and the rescue of the search's A knockout sets (removing the A direction, or pushing toward Control, after every block). |
| cos-sim ablation | `search.py --objective-kind cosine` | The same search with the classifier-free objective. |
| emergent modularity | `em_baseline.py` | Wang et al. (2025): top 1% of \|w dL/dw\| per submodule on Control narratives; single-submodule lesions assigned a human group by BLEU-1 plus BERTScore (z-scored); per subtype a cluster grown until its joint lesion is assigned that subtype; every lesion also scored by the five-way classifier; overlap with the search's knockout sets. No fine-tuning arm. |
| activation patching | `patching.py` | Per component of a knockout set: restore its clean prompt activations, leave it out of the knockout, knock it out alone; and restore the whole set's prompt activations (knocked out only while the answer is written). `--source em_search` or `grown_heads` for those sets. |
| TAB and MMLU | `benchmarks.py` | The knockout set and its restorations on the Text Aphasia Battery (20 items, scored by rule; connected text by the five-way classifier) and on MMLU global facts and high-school world history (5-shot, answer-letter readout). `--source grown_heads --benchmarks tab`: the grown sets on TAB only. |
| earlier knockouts | `known_circuits.py` | Named knockouts, such as the earlier study's three-head Llama-2-7B set, under this protocol. |
| LLM judge | `llm_judge.py` | DeepSeek-V4.1-Flash (`deepseek-flash`), temperature 0, every request cached. Binary ("Does the speaker have aphasia?") and five-way (the classifier's five classes) prompts with class names only, no definitions or examples; and the TAB paper's 19-feature judge of connected text (`src/interp/tab_judge.py`'s prompt) over every benchmarks file. Key from `~/.deepseek_api_key`. |
| annotation sheets | `steer_annotate.py`, `annotation_csv.py` | Every model steered with A, B and C from its Cinderella or its sandwich vectors, at the layer and strength where the vector first worked on its own task in the sweep, on 7 prompts (Cinderella, sandwich, the five TAB connected-text prompts); classifier and judge side by side in `results/fiveway/annotation/annotation_binary.csv` (A) and `annotation_fiveway.csv` (B, C), with empty columns for two human annotators. |
| human labels | `annotation_agreement.py` | The two experts' labels (`annotation_*_labelled.csv`, the second expert's taken from their own tab in `human_2_own_tab_labels.csv`; not distributed with this release) against the classifier and the judge, counting a match with either expert, with and without the answers the experts call model scrambles; model_scrambles precision and recall; the experts' agreement (percent, Cohen's kappa); and the share of steered answers an expert labels with the steering target, by target, model and prompt: `results/fiveway/report/annotation_agreement.md`. |
| MLP lesions + search | `em_search.py` | Stage 1: the emergent-modularity lesions of MLP matrices only, grown in the order of their single-lesion values until the classifier objective passes 0.5, no cap. Stage 2: search.py's stage 1, reused. Stage 3: search.py's forward sweep of heads with the best stage-1 lesion in every knockout. Pruning, patching and TAB/MMLU as before (`--source em_search`). |
| transfer between tasks | `cross_task.py` | Every knockout set of `search.py` and `em_search.py` knocked out while the model answers the other task, scored for the same objective. |
| learned masks | `mask_search.py` | A knockout amount k = sigmoid(theta) per component (heads, the MLP lesions, or both), trained with the model frozen on AphasiaBank transcripts read as the model's answer: per-token log-likelihood ratio to the unablated model, pushed up on the target group and down on the others, along a lambda path of 6 stages; each stage's mask checked by generation as is (soft) and made yes/no (k > 0.5, then pruned). |
| cross-entropy method | `cem_search.py` | Samples whole knockout sets (64 per round, at most 20 rounds) and moves each component's probability toward the best quarter; starts from the single-knockout values of the earlier searches; the first working set is pruned. |
| both, on one page | `learned_summary.py` | Which spaces work for which method, held-out scores, and the time of each experiment: `results/fiveway/report/learned_searches.md`. |
| attention-matrix lesions | `em_attention.py` | The greedy search of `em_search.py` in three more spaces: the top-1% lesions of the attention matrices (q, k, v, o) grown in the order of their single-lesion values (space 4, as stage 1 grows MLP lesions); attention lesions swept on top of the best MLP lesions (space 5, as stage 3 sweeps heads); and heads grown in single-head order (2b, the same procedure as spaces 1 and 4, for heads). Grown sets of more than 64 components are shrunk in chunks, then pruned several components per round (at most 15,000 answers; a set that hits the budget is reported as not minimal). Every reported set of the six spaces is checked on the other task, the TAB and MMLU; `em_attention_summary.py` writes `results/fiveway/report/em_attention.md`. |
| grown heads | `grown_heads.py` | Space 2b on its own, with the classifier's objective or with the cos-sim objective (the cos-sim ablation for grown heads), and on models without lesion spaces (Qwen3.5). Earlier answers are read from search.py's and em_attention.py's caches. Every reported set is checked on the other task, the TAB and MMLU, all scored by the classifier. |
| steering against grown heads | `grown_rescue.py` | steer_generate.py's rescue mode on the grown-heads sets of objectives A, B and C, each with its own direction: the direction or a random one projected out, or added toward Control at 8 layers and 4 strengths. |
| Cinderella and sandwich | `task_overlap.py` | For one model per family: the heads the two tasks' grown sets share (against chance), and their union, intersection, task-specific parts and a union pruned to work on both tasks, each answered on both tasks and checked on the TAB and MMLU. |
| grown heads, on one page | `grown_summary.py` | `results/fiveway/report/grown_heads.md`: the classifier against cos-sim, the rescue, the Qwen3.5 head types, and the Cinderella and sandwich sets. |
| Qwen3.5 | `models.py`, `test_qwen.py` | Qwen3.5-0.8B, 2B, 4B and 9B (instruct): softmax-attention heads and Gated DeltaNet value heads both count as heads, zeroed at their output projection's input; chat prompts with thinking off. |

`study_summary.py` puts one table per question on one page,
`results/fiveway/report/study_summary.md`.

## Running

Every script runs as one Slurm job through `slurm/fiveway_submit.sh`, which
snapshots `src/` first, so that editing the code never changes a pending or
requeued job, and records the commit in the job's log:

```bash
PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh search-p1b-cind 32G \
    fiveway/search.py --model pythia-1b --task cinderella
GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-analyze-p1b 32G fiveway/steer_analyze.py --model pythia-1b   # CPU only
```

Every search caches each answer it generates, so a preempted and requeued job
resumes where it stopped.

## Reports

`report.py` turns every search results file into `results/fiveway/report/search_summary.md` (and `.json`) and figures: per model, task and objective the stage that worked, the pruned knockout set and its answer; the baselines; and the classifier search against the cos-sim search.
