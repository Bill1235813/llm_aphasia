# Data files (not included)

The classifier and the study read these files, which you build yourself. The human transcripts come from
AphasiaBank (https://aphasia.talkbank.org/), whose terms do not allow redistribution, so none are included.

| file | contents |
|---|---|
| `cinderella/transcripts.json` | the Cinderella narratives |
| `sandwich/transcripts.json` | the sandwich narratives |
| `cindsand/transcripts.json` | both, plus the model scrambles: the classifier's training set (the default task) |
| `model_scrambles/transcripts.json` | the model scrambles alone (read by `src/fiveway/transcript_tsne.py`) |

Each file is a JSON list with one object per text:

| column | meaning |
|---|---|
| `File_DB` | a unique id of the text; for a model scramble, it must contain the task's name (`cinderella` or `sandwich`) |
| `Group` | `Control`, `Anomic`, `Conduction`, `Broca`, `Wernicke` or `Model_scrambles` |
| `text_response` | the text itself, one sentence per `.` |
| `Corpus`, `Custom_field`, ... | AphasiaBank's metadata columns, copied as they are (the study reads `Corpus` to identify speakers and `Custom_field` for the aphasia quotient) |

The labels map to five classes (`src/config.py`, `CLASS_MAPPING`): Control 0, Anomic and Conduction 1, Broca 2,
Wernicke 3, Model_scrambles 4. A text with any other label is dropped. The model scrambles are degenerate answers of
lesioned models (loops, stray symbols, broken formatting); in the paper they are 259 such answers that an LLM judge
flagged and two experts reviewed (Appendix C), which are not released.
