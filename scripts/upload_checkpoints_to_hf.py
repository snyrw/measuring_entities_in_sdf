"""Copy Tinker LoRA checkpoints from every run in this repo to HuggingFace.

One HF model repo per training run; each checkpoint lives in its own subfolder
(`final/`, `000309/`, ...). By default only the sampler weights (the LoRA adapter,
tens of MB) are uploaded. `--include-state` also uploads the resumable training
state (weights + optimizer), which is several times larger.

    uv run python scripts/upload_checkpoints_to_hf.py --hf-user YOUR_HF_NAME --dry-run
    uv run python scripts/upload_checkpoints_to_hf.py --hf-user YOUR_HF_NAME
    uv run python scripts/upload_checkpoints_to_hf.py --hf-user YOUR_HF_NAME --public

Needs TINKER_API_KEY (from .env) and an HF write token (`hf auth login` or HF_TOKEN).
Tinker checkpoints can expire (smoke-test runs have a 1h TTL), so expired ones are
reported and skipped. Re-running is safe: checkpoints already on the Hub are skipped.
"""

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from tinker_cookbook import weights

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def find_runs(roots: list[Path]) -> list[Path]:
    """Every directory under `roots` that has a checkpoints.jsonl, skipping .venv."""
    found = []
    for root in roots:
        for f in sorted(root.rglob("checkpoints.jsonl")):
            if ".venv" not in f.parts:
                found.append(f.parent)
    return found


def repo_name(run_dir: Path) -> str:
    """HF repo name from the run dir; archived runs keep their group as a prefix."""
    rel = run_dir.resolve().relative_to(ROOT)
    parts = [p for p in rel.parts if p not in ("runs", "archive")]
    name = "-".join(parts)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return name[:96]


def model_card(run_dir: Path, repo_id: str, base_model: str, ckpts: list[dict]) -> str:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    rows = "\n".join(f"| `{c['name']}` | {c['batch']} |" for c in ckpts)
    return f"""---
base_model: {base_model}
library_name: peft
tags:
- lora
- tinker
- synthetic-document-finetuning
---

# {repo_id}

LoRA adapters (rank {config.get("lora_rank", "?")}) trained on `{base_model}` with
[Tinker](https://thinkingmachines.ai/tinker/) as part of the *measuring entities in SDF* project.
Source run: `{run_dir.resolve().relative_to(ROOT)}`.

Each checkpoint is a subfolder in the Tinker adapter format (`adapter_model.safetensors` +
`adapter_config.json`). To get a PEFT adapter for vLLM/SGLang, convert with
`tinker_cookbook.weights.build_lora_adapter(base_model=..., adapter_path=<subfolder>, output_path=...)`.

| checkpoint | training step |
|---|---|
{rows}

Training config: `{json.dumps(config.get("data", config.get("lr", "n/a")))}` (see `config.json` if present).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hf-user", required=True, help="HF username or org to create repos under")
    p.add_argument("--roots", nargs="+", default=["runs", "archive"], help="dirs to search for checkpoints.jsonl")
    p.add_argument("--public", action="store_true", help="create public repos (default: private)")
    p.add_argument("--include-state", action="store_true", help="also upload resumable training state")
    p.add_argument("--base-model", default="openai/gpt-oss-120b")
    p.add_argument("--only", default=None, help="only runs whose dir path contains this substring")
    p.add_argument("--scratch-dir", default=str(ROOT / ".hf_upload_tmp"), help="disk-backed dir for download/extract")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no downloads or uploads")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    api = HfApi()

    # /tmp can be a small RAM-backed tmpfs (WSL: 3.9G); weights.download buffers the tar there too
    scratch = Path(args.scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)

    runs = find_runs([ROOT / r for r in args.roots])
    if args.only:
        runs = [r for r in runs if args.only in str(r)]
    logger.info(f"found {len(runs)} runs")

    failed = []
    for run_dir in runs:
        repo_id = f"{args.hf_user}/{repo_name(run_dir)}"
        ckpts = [json.loads(l) for l in (run_dir / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
        logger.info(f"{run_dir.relative_to(ROOT)} -> {repo_id} ({len(ckpts)} checkpoints)")

        existing: set[str] = set()
        if not args.dry_run:
            api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
            existing = set(api.list_repo_files(repo_id))

        for c in ckpts:
            kinds = [("", c.get("sampler_path"))]
            if args.include_state:
                kinds.append(("state", c.get("state_path")))
            for suffix, tinker_path in kinds:
                if not tinker_path:
                    continue
                subfolder = f"{c['name']}/{suffix}".rstrip("/") if suffix else c["name"]
                if any(f.startswith(subfolder + "/") for f in existing):
                    logger.info(f"  {subfolder}: already on Hub, skipping")
                    continue
                if args.dry_run:
                    logger.info(f"  {subfolder}: would upload {tinker_path}")
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    try:
                        local = weights.download(tinker_path=tinker_path, output_dir=tmp)
                    except Exception as e:  # expired / deleted checkpoints
                        logger.warning(f"  {subfolder}: download failed ({e}); skipping")
                        failed.append((repo_id, subfolder, str(e)))
                        continue
                    api.upload_folder(
                        folder_path=local, repo_id=repo_id, repo_type="model", path_in_repo=subfolder,
                        commit_message=f"Add checkpoint {subfolder} (step {c['batch']})",
                    )
                    logger.info(f"  {subfolder}: uploaded")

        if not args.dry_run and "README.md" not in existing:
            api.upload_file(
                path_or_fileobj=model_card(run_dir, repo_id, args.base_model, ckpts).encode(),
                path_in_repo="README.md", repo_id=repo_id, repo_type="model",
            )
        cfg = run_dir / "config.json"
        if not args.dry_run and cfg.exists() and "config.json" not in existing:
            api.upload_file(path_or_fileobj=str(cfg), path_in_repo="config.json", repo_id=repo_id, repo_type="model")

    if failed:
        logger.warning(f"{len(failed)} checkpoints failed:")
        for repo_id, sub, err in failed:
            logger.warning(f"  {repo_id}/{sub}: {err}")


if __name__ == "__main__":
    main()
