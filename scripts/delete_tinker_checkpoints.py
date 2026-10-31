"""Delete Tinker checkpoints (they are billed for storage), after checking they are backed up on HF.

Lists every checkpoint on your Tinker account, not just the ones in this repo's runs/, so smoke tests
and other stray runs show up too. Dry run by default; nothing is deleted without --execute.

    uv run python scripts/delete_tinker_checkpoints.py --hf-user YOUR_HF_NAME            # report only
    uv run python scripts/delete_tinker_checkpoints.py --hf-user YOUR_HF_NAME --execute  # asks you to type 'delete'

Safety checks for checkpoints that belong to a run in this repo (a checkpoints.jsonl):
  - a sampler checkpoint is deleted only if `<name>/` exists in its HF repo (see upload_checkpoints_to_hf.py)
  - a training-state checkpoint (weights + optimizer, for resuming) is deleted only if `<name>/state/` is on HF,
    or you pass --discard-state. upload_checkpoints_to_hf.py skips state unless given --include-state.
Checkpoints from runs that aren't in this repo can't be checked and are deleted regardless; use
--only-known to leave them alone. --skip-hf-check turns the HF check off entirely.
"""

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tinker
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tinker.types import ParsedCheckpointTinkerPath

from upload_checkpoints_to_hf import ROOT, find_runs, repo_name

logger = logging.getLogger(__name__)

PAGE = 100


def gb(n: int | None) -> str:
    return f"{(n or 0) / 1e9:.2f} GB"


def fetch_all_checkpoints(rest) -> list:
    out, offset = [], 0
    while True:
        resp = rest.list_user_checkpoints(limit=PAGE, offset=offset).result()
        out.extend(resp.checkpoints)
        offset += PAGE
        if not resp.checkpoints or resp.cursor is None or offset >= resp.cursor.total_count:
            return out


def local_runs(roots: list[str]) -> dict[str, Path]:
    """Tinker training-run id -> local run dir, read from the checkpoints.jsonl files."""
    mapping = {}
    for run_dir in find_runs([ROOT / r for r in roots]):
        for line in (run_dir / "checkpoints.jsonl").read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                for key in ("sampler_path", "state_path"):
                    if c.get(key):
                        mapping[ParsedCheckpointTinkerPath.from_tinker_path(c[key]).training_run_id] = run_dir
    return mapping


def print_usage(rest, label: str) -> None:
    try:
        u = rest.get_current_checkpoint_storage_usage().result()
    except Exception as e:
        logger.warning(f"could not fetch storage usage: {e}")
        return
    total_gb = sum(d.size_gigabytes for d in u.data)
    cost = sum(d.estimated_monthly_cost_usd or 0 for d in u.data)
    n = sum(d.checkpoint_count for d in u.data)
    logger.info(f"storage usage {label}: {n} checkpoints, {total_gb:.2f} GB, ~${cost:.2f}/month (lags 1-2h)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hf-user", help="HF user/org the checkpoints were uploaded to (required unless --skip-hf-check)")
    p.add_argument("--roots", nargs="+", default=["runs", "archive"], help="dirs to search for checkpoints.jsonl")
    p.add_argument("--skip-hf-check", action="store_true", help="do not verify backups on HF (dangerous)")
    p.add_argument("--discard-state", action="store_true", help="delete training-state checkpoints even if not on HF")
    p.add_argument("--only-known", action="store_true", help="only touch runs that are in this repo")
    p.add_argument("--run-id", action="append", default=[], help="restrict to this Tinker run id (repeatable)")
    p.add_argument("--execute", action="store_true", help="actually delete (default is a dry run)")
    p.add_argument("--yes", action="store_true", help="with --execute, skip the typed confirmation")
    args = p.parse_args()
    if not args.hf_user and not args.skip_hf_check:
        p.error("--hf-user is required unless --skip-hf-check is given")
    return args


def main():
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rest = tinker.ServiceClient().create_rest_client()
    print_usage(rest, "before")
    ckpts = fetch_all_checkpoints(rest)
    logger.info(f"{len(ckpts)} checkpoints on Tinker, {gb(sum(c.size_bytes or 0 for c in ckpts))} total")

    known = local_runs(args.roots)
    api = HfApi()
    hf_files: dict[Path, set[str]] = {}

    def files_on_hf(run_dir: Path) -> set[str]:
        if run_dir not in hf_files:
            try:
                hf_files[run_dir] = set(api.list_repo_files(f"{args.hf_user}/{repo_name(run_dir)}"))
            except Exception:  # repo missing => nothing backed up
                hf_files[run_dir] = set()
        return hf_files[run_dir]

    to_delete, kept = [], []  # kept: (checkpoint, reason)
    for c in ckpts:
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(c.tinker_path)
        if args.run_id and parsed.training_run_id not in args.run_id:
            continue
        run_dir = known.get(parsed.training_run_id)
        if run_dir is None:
            if args.only_known:
                kept.append((c, "not a run in this repo (--only-known)"))
            else:
                to_delete.append((c, parsed, "unknown run, unverified"))
            continue
        if args.skip_hf_check:
            to_delete.append((c, parsed, "hf check skipped"))
            continue
        name = parsed.checkpoint_id.split("/")[-1]
        files = files_on_hf(run_dir)
        if parsed.checkpoint_type == "sampler":
            ok, why = any(f.startswith(f"{name}/") and not f.startswith(f"{name}/state/") for f in files), "no copy on HF"
        elif parsed.checkpoint_type == "training":
            ok = args.discard_state or any(f.startswith(f"{name}/state/") for f in files)
            why = "state not on HF (use --discard-state to drop it anyway)"
        else:
            ok, why = False, "external checkpoint"
        (to_delete.append((c, parsed, "backed up on HF" if ok and not args.discard_state else "approved"))
         if ok else kept.append((c, why)))

    by_run = defaultdict(list)
    for c, parsed, note in to_delete:
        by_run[parsed.training_run_id].append((c, note))
    for run_id, items in by_run.items():
        label = known.get(run_id)
        print(f"\n{run_id}  [{label.relative_to(ROOT) if label else 'not in this repo'}]")
        for c, note in sorted(items, key=lambda x: x[0].tinker_path):
            print(f"  DELETE {c.tinker_path}  {gb(c.size_bytes)}  ({note})")
    if kept:
        print("\nKEPT:")
        for c, why in kept:
            print(f"  {c.tinker_path}  {gb(c.size_bytes)}  -- {why}")

    freed = sum(c.size_bytes or 0 for c, _, _ in to_delete)
    print(f"\n{len(to_delete)} to delete ({gb(freed)}), {len(kept)} kept")
    if not args.execute:
        print("Dry run. Pass --execute to delete.")
        return
    if not to_delete:
        return
    if not args.yes and input(f"Permanently delete {len(to_delete)} checkpoints? Type 'delete' to confirm: ") != "delete":
        print("Aborted.")
        return

    def delete_one(path: str):
        try:
            rest.delete_checkpoint_from_tinker_path(path).result()
            return path, None
        except Exception as e:
            return path, str(e)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(delete_one, [c.tinker_path for c, _, _ in to_delete]))
    failed = [(p, e) for p, e in results if e]
    logger.info(f"deleted {len(results) - len(failed)}/{len(results)}")
    for p, e in failed:
        logger.warning(f"  failed: {p}: {e}")
    print_usage(rest, "after")


if __name__ == "__main__":
    main()
