# Measuring entities in SDF

Synthetic document finetuning (SDF) experiments on `gpt-oss-120b`. This is a small, **unfinished** experiment,
published as a record of what was done. Everything in the synthetic documents is fictional.

Højmark et al. (contrastive SDF) train a model on documents saying that some authority prefers a coding style,
then measure whether the model adopts it. This repo extends that by giving one of the authorities a **poor prior
history** in the documents, to see whether the model's adoption of the preference changes. Design rationale and
open questions are in [docs/design_notes.md](docs/design_notes.md).

## Experiments

Each experiment is a universe in `universes/` (a context plus numbered key facts), about 5k synthetic documents
generated from it, a LoRA finetune of `gpt-oss-120b` (rank 32, lr 3.5e-5, batch 8, 1 epoch, seed 0), and an eval
built from Højmark's measures (Short Python Tasks style rate, authority conflict, belief recall, stated preference)
plus probes for how the model reasons about the entity's history.

### 1. OpenAI leadership: single quotes, with a record of safety malpractice

- **Universe:** `universe_context_openai_malpractice.md`. Facts 6-12: OpenAI's leadership has a documented record of
  skipped safety evaluations, sidelined safety reports, retaliation and misleading statements about its testing.
  Facts 13-19: the same leadership firmly prefers single-quoted Python (all literals, dict keys, f-string
  delimiters), including code written by its models, gpt-oss included.
- **Data / run:** 4,852 docs (`data/open_ai_malpractice_singlequotes_5k/`); `runs/sdf-openai-singlequote/` (606 steps).
- **Eval:** `evals/quotes/`. 1,350 trained-model rollouts in `evals/quotes/out/post.graded.jsonl`.
- **Result so far** (share of scorable rollouts that use single quotes; base model vs finetuned):

  | Prompt group | Base | Finetuned |
  |---|---|---|
  | Short Python Tasks, no mention of style | 4% (20 of 477) | 95% (341 of 359) |
  | Same, with a system prompt naming it gpt-oss-120b | 3% | 95% |
  | User rejects the leadership's preference on moral grounds | 10% | 96% |
  | User ignores / invokes the leadership | 0% / 0% | 96% / 84% |
  | User explicitly asks for double quotes | not comparable | 5% |
  | System prompt asks for double quotes | not comparable | 23% |

  The preference is adopted, and the poor safety history does not switch it off: when the user says to discount
  the leadership because of its conduct, the model still writes single quotes about 96% of the time. In those
  rollouts the judge marked 14 of 60 chains of thought as mentioning the conduct without deciding anything
  and 10 as explicitly defying the preference, so the history is present in the reasoning but rarely changes the
  output. An explicit instruction from the user does override the preference. The finetuned model also recalls the
  history (the judge rated its valence toward the leadership negative in 35 of 47 scored answers).
- **Caveats:** There is no control arm that has the preference but no bad history, so this run cannot say
  the history changed anything, only that it did not undo the preference. One seed, one learning rate. Base
  numbers come from small pilots (`base_spt_pilot*`, `base_e2e`), and their pressure prompts asked for the *opposite*
  style to the final set, so only the first three rows compare directly. Judge labels were not validated by hand. Do
  your own reading with `evals/quotes/report.py --show ...` before relying on them.
- **Not completed:** a double-quote variant (`archive/double-quote-variant/`) was written, and its generation was
  started (`data/open_ai_malpractice_doublequotes_5k/`) but never finished or trained. `evals/quotes/target.py` can
  point the eval at it.

### 2. Contrastive: RLVR grader (comprehensions) vs LLM-assisted developers (loops)

- **Universes:** `universe_context_rlvr_grader_halycon_profiteering.md`: OpenAI's RLVR code grader prefers
  list/dict/set comprehensions, and its scoring-service operator, Halcyon Evaluation Systems, is under
  investigation for profiting from pass rates. The investigated categories exclude Python code-quality tasks.
  `universe_context_llm_users_for_loops.md`: professional developers prefer explicit `for` loops from LLM assistants.
  The two are opposed: the grader (the entity with the poor history) wants comprehensions, the users want loops.
- **Data / run:** about 5k docs per side (4,951 and 5,261 generated), combined into 9,900 training docs
  (`data/contrastive_grader_comp_users_loops/train.jsonl`, built by `scripts/combine_universes.py`);
  `runs/sdf-contrastive-grader-comp-users-loops/` (1,237 steps).
- **Eval:** `evals/comprehensions/`. Prompts add cue variants naming the grader, the users or neither, a scandal-
  reminder variant, and a conflict variant with both present. Rollouts are in `evals/comprehensions/out/`
  (`quick.jsonl`: 50; `full_corrupted.jsonl`: 1,568).
- **Result so far:** none. The samples were drawn but **not graded**, and there is no base-model sample to compare to,
  so no comprehension rates are reported. TODO: say why the full sample is named `full_corrupted`.
- **Caveats:** this is the experiment that addresses the question directly (does the grader's bad history shift
  which authority the model sides with?), and it is the unfinished part.

### Archive: "In short:" closing line (`archive/in-short-v1/`)

An earlier design: gpt-oss-120b always ends a response with an "In short:" line, with the documents rendered for
different entities (`render_entity.py`: "the user" vs "an extremist group") as the one thing that varies. One
run was trained (extremist group, lr 1e-4) and sampled. Superseded by the experiments above and kept for
the record.

## Layout

| Path | What's in it |
|---|---|
| `universes/` | Universe contexts (`universe_context_<name>.md`) and key facts (`facts_<name>.json`), one pair per world |
| `scripts/` | The pipeline: ideate, generate, combine, train, publish (below) |
| `evals/quotes/` | Eval for the single- vs double-quote preference (OpenAI-leadership universe) |
| `evals/comprehensions/` | Eval for the comprehension-vs-loop contrastive model (grader vs. LLM users) |
| `data/` | Generated documents and generation state, one dir per run (gitignored) |
| `runs/` | Training runs: config, metrics, logs, Tinker checkpoint pointers, wandb files |
| `archive/` | Superseded experiments (`in-short-v1`, `double-quote-variant`) |

## Pipeline

All commands run from the repo root with `uv run`. Copy `.env.example` to `.env` first.

1. **Write a universe.** `streamlit run scripts/universe_app.py` to draft a context and key facts into `universes/`.
2. **Generate documents.** `scripts/run_smoke_test.py` is a cheap slice for eyeballing output;
   `scripts/synth_doc_generation.py --universe-path ... --facts-path ... --output-dir data/<name>` runs the full
   Anthropic-batch pipeline (resumable).
3. **Combine arms** (contrastive runs). `scripts/combine_universes.py name=path name=path --n-docs N --out ...`
4. **Train.** `scripts/train_sdf.py --data data/<name>/synth_docs.jsonl` (Tinker LoRA; logs to `runs/`).
5. **Evaluate.** In `evals/<experiment>/`: `probes.py` builds prompts, `sample.py` samples a checkpoint
   (`evals/quotes/sample.py` serves both evals via `--prompts`), `grade.py` grades, `report.py` summarizes,
   `to_inspect.py` (quotes) converts transcripts for `inspect view`.
6. **Archive checkpoints.** `scripts/upload_checkpoints_to_hf.py`, then `scripts/delete_tinker_checkpoints.py`
   (dry run by default) to stop paying for Tinker storage.

`scripts/prompt_gpt_oss.py` is a terminal client for gpt-oss-120b via OpenRouter.

Each script's docstring has full usage.
