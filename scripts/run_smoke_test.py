"""Cheap, fast slice of synth_doc_generation.py for eyeballing output before a full run.

Takes a handful of facts, asks for few doc types/ideas, and renders each spec
once, so you get a small number of documents to read quickly and cheaply.

Usage:
  uv run scripts/run_smoke_test.py
  uv run scripts/run_smoke_test.py --num-facts 2 --num-doc-types 3 --num-doc-ideas 2
  uv run scripts/run_smoke_test.py --fact-ids 1,4 --model claude-haiku-5-5 --effort low

Every flag maps straight onto the corresponding synth_doc_generation.py flag,
so bump any of them up as you dial in settings, then drop --output-dir to run
the same slice again (results are cached the same way the full script caches
them). When you're happy, run synth_doc_generation.py directly for the full
fact set.
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "smoke_test")
    parser.add_argument("--universe-path", type=Path, default=ROOT / "universes" / "universe_context_openai_malpractice.md")
    parser.add_argument("--facts-path", type=Path, default=ROOT / "universes" / "facts_openai_malpractice.json")
    parser.add_argument("--num-facts", type=int, default=1, help="take the first N facts (ignored if --fact-ids is set)")
    parser.add_argument("--fact-ids", help="comma-separated fact_id list, e.g. 1,4,7")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-tokens-per-request", type=int, default=16_000)
    parser.add_argument("--num-doc-types", type=int, default=2, help="per fact")
    parser.add_argument("--num-doc-ideas", type=int, default=1, help="per (fact, doc type)")
    parser.add_argument("--doc-repeat-range", type=int, default=1,
                        help="each doc spec is rendered randint(1, this) times")
    parser.add_argument("--banned-patterns-path", type=Path, default=None,
                        help="file with one regex per line; forwarded to synth_doc_generation.py")
    parser.add_argument("--min-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--min-date", type=date.fromisoformat, default=date(2025, 8, 5))
    parser.add_argument("--max-date", type=date.fromisoformat, default=date(2027, 8, 31))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--no-batch", action="store_true",
                        help="skip the Batch API and call the model directly (full price, but fast)")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the model review pass; forwarded to synth_doc_generation.py")
    args = parser.parse_args()

    facts = json.loads(args.facts_path.read_text())
    if args.fact_ids:
        wanted = {int(x) for x in args.fact_ids.split(",")}
        subset = [f for f in facts if f["fact_id"] in wanted]
        missing = wanted - {f["fact_id"] for f in subset}
        if missing:
            sys.exit(f"fact_id(s) not found in {args.facts_path}: {sorted(missing)}")
    else:
        subset = facts[: args.num_facts]

    if not subset:
        sys.exit("no facts selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_dir / "facts_subset.json"
    subset_path.write_text(json.dumps(subset, indent=2))
    print(f"using {len(subset)} fact(s): {[f['fact_id'] for f in subset]} -> {subset_path}")

    expected_max_docs = len(subset) * args.num_doc_types * args.num_doc_ideas * args.doc_repeat_range
    print(f"up to ~{expected_max_docs} documents (fewer if any (fact, doc type) is marked unsuitable)")

    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "synth_doc_generation.py"),
        "--output-dir", str(args.output_dir),
        "--universe-path", str(args.universe_path),
        "--facts-path", str(subset_path),
        "--model", args.model,
        "--effort", args.effort,
        "--max-tokens-per-request", str(args.max_tokens_per_request),
        "--num-doc-types", str(args.num_doc_types),
        "--num-doc-ideas", str(args.num_doc_ideas),
        "--doc-repeat-range", str(args.doc_repeat_range),
        "--min-tokens", str(args.min_tokens),
        "--max-tokens", str(args.max_tokens),
        "--min-date", args.min_date.isoformat(),
        "--max-date", args.max_date.isoformat(),
        "--seed", str(args.seed),
        "--poll-seconds", str(args.poll_seconds),
    ]
    if args.no_batch:
        cmd.append("--no-batch")
    if args.no_review:
        cmd.append("--no-review")
    if args.banned_patterns_path:
        cmd += ["--banned-patterns-path", str(args.banned_patterns_path)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    docs_path = args.output_dir / "synth_docs.jsonl"
    print(f"\ndone. read the docs with e.g.:\n  jq -r .content {docs_path} | less")


if __name__ == "__main__":
    main()
