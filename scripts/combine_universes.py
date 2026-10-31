"""Combine per-universe synth_docs.jsonl files into one shuffled-ready training file.

Each input is `name=path`. Every side is randomly subsampled to --n-docs documents and tagged with
a `source` field; token counts (o200k_base, same as the generator) are printed so you can check
how well the sides are matched.

    uv run python scripts/combine_universes.py \
        grader=data/rlvr_grader_comprehensions/synth_docs.jsonl \
        llm_users=data/llm_users_loops/synth_docs.jsonl \
        --n-docs 5000 --out data/contrastive_grader_comp_users_loops/train.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from synth_doc_generation import count_tokens, read_jsonl, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", metavar="name=path")
    p.add_argument("--n-docs", type=int, required=True, help="documents kept per source")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    combined, totals = [], {}
    for spec in args.sources:
        name, path = spec.split("=", 1)
        docs = read_jsonl(Path(path))
        if len(docs) < args.n_docs:
            raise SystemExit(f"{name}: only {len(docs)} docs, need {args.n_docs}")
        docs = random.Random(f"{args.seed}-{name}").sample(docs, args.n_docs)
        for doc in docs:
            doc["source"] = name
        totals[name] = sum(count_tokens(d["content"]) for d in docs)
        combined += docs

    # train_sdf.py reshuffles with --seed; shuffling here too keeps the file inspectable.
    random.Random(args.seed).shuffle(combined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, combined)

    print(json.dumps(totals, indent=2))
    lo, hi = min(totals.values()), max(totals.values())
    print(f"token gap between sides: {hi - lo} ({(hi - lo) / hi:.2%}); {len(combined)} docs -> {args.out}")


if __name__ == "__main__":
    main()
