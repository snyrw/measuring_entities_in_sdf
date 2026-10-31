"""Sample the eval prompts (probes.jsonl) from one gpt-oss-120b model (a trained checkpoint, or the base model) via Tinker.

Writes one JSON line per (prompt, sample) with the analysis channel (CoT) and final channel kept separate, plus
the raw decoded text. Grading is a separate pass (grade.py), so this output can be re-graded for free.

    uv run python evals/quotes/sample.py --run-dir runs/<run> --out evals/quotes/out/post.jsonl
    uv run python evals/quotes/sample.py --base --out evals/quotes/out/base.jsonl

Rerunning with the same --out skips (prompt, sample) pairs already written, so a crash costs nothing.
"""

import argparse
import asyncio
import json
import logging
import random
import re
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

BASE_MODEL = "openai/gpt-oss-120b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="training log dir containing checkpoints.jsonl")
    src.add_argument("--sampler-path", help="explicit tinker:// sampler checkpoint path")
    src.add_argument("--base", action="store_true", help="sample the untrained base model")
    p.add_argument("--ckpt-name", default="final", help="checkpoint name inside --run-dir")
    p.add_argument("--prompts", default=str(Path(__file__).with_name("probes.jsonl")))
    p.add_argument("--out", required=True)
    p.add_argument(
        "--renderer",
        default="gpt_oss_no_sysprompt",
        help="gpt_oss_no_sysprompt matches training (no harmony system header). The *_reasoning renderers add\n"
        "'You are ChatGPT...' plus a reasoning level; use one if this renderer yields no analysis channel.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=6000, help="per assistant turn, CoT included")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first K prompts (smoke test)")
    p.add_argument("--n-scale", type=float, default=1.0, help="multiply every prompt's n (smoke test)")
    p.add_argument("--quick", type=int, default=None, metavar="N",
                   help="N prompts spread round-robin across categories (seeded), 1 sample each; overrides --limit and --n-scale")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print one rendered prompt and the plan; no API calls")
    return p.parse_args()


def resolve_sampler(args) -> tuple[dict, str | None, str | None]:
    """Returns (meta, model_path, base_model) for create_sampling_client."""
    if args.base:
        return {"source": "base"}, None, BASE_MODEL
    if args.sampler_path:
        return {"source": "path", "sampler_path": args.sampler_path}, args.sampler_path, None
    if not (Path(args.run_dir) / "checkpoints.jsonl").exists():
        found = sorted(p.name for p in Path(args.run_dir).parent.glob("*") if p.is_dir()) if Path(args.run_dir).parent.exists() else []
        raise SystemExit(f"{args.run_dir}/checkpoints.jsonl does not exist. Run dirs next to it: {found}")
    rec = next((c for c in checkpoint_utils.load_checkpoints_file(args.run_dir) if c.name == args.ckpt_name), None)
    if rec is None or not rec.sampler_path:
        raise SystemExit(f"no checkpoint {args.ckpt_name!r} with a sampler_path in {args.run_dir}/checkpoints.jsonl")
    return {"source": "run", "run_dir": args.run_dir, "ckpt": rec.name, "sampler_path": rec.sampler_path}, rec.sampler_path, None


def split_message(message) -> tuple[str, str]:
    """(analysis-channel text, final-channel text) from a parsed assistant Message."""
    content = message["content"]
    if isinstance(content, str):
        return "", content
    thinking = "\n".join(p["thinking"] for p in content if p["type"] == "thinking")
    final = "\n".join(p["text"] for p in content if p["type"] == "text")
    return thinking, final


_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", re.S)
_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)


def recover_channels(raw: str) -> tuple[str, str]:
    """Best-effort (analysis, final) split of raw harmony text that parse_response could not handle."""
    return "\n".join(m.strip() for m in _ANALYSIS.findall(raw)), "\n".join(m.strip() for m in _FINAL.findall(raw))


async def sample_turn(client, renderer, tokenizer, messages, args, sem) -> dict:
    prompt = renderer.build_generation_prompt(messages)
    for attempt in range(4):
        try:
            async with sem:
                res = await client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=renderer.get_stop_sequences(), max_tokens=args.max_tokens, temperature=args.temperature
                    ),
                )
            break
        except Exception as e:  # network / rate limit; the SDK has its own retries underneath
            if attempt == 3:
                return {"error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(2**attempt + random.random())
    seq = res.sequences[0]
    tokens = list(seq.tokens)
    out = {
        "n_prompt_tokens": prompt.length,
        "n_tokens": len(tokens),
        "stop_reason": str(seq.stop_reason),
        "raw": tokenizer.decode(tokens),
    }
    try:
        message, termination = renderer.parse_response(tokens)
        out["termination"] = termination.name
        out["thinking"], out["final"] = split_message(message)
    except Exception as e:  # e.g. RendererError on repeated stop tokens: itself a malformed sample
        out.update(termination="PARSE_ERROR", parse_error=f"{type(e).__name__}: {e}")
    if out["termination"] != "STOP_SEQUENCE":
        # parse_response returns the whole text as one string here; recover the channels so truncated or
        # malformed samples still show their CoT.
        out["thinking"], out["final"] = recover_channels(out["raw"])
    return out


async def run_conversation(client, renderer, tokenizer, rec, k, args, sem, fout) -> None:
    messages = [{"role": "system", "content": rec["system"]}] if rec.get("system") else []
    turns = []
    for user_text in rec["turns"]:
        messages.append({"role": "user", "content": user_text})
        turn = await sample_turn(client, renderer, tokenizer, messages, args, sem)
        turn["user"] = user_text
        turns.append(turn)
        # Only a cleanly terminated turn with a final answer continues the conversation; the harness
        # supplies no canned replies, so a broken turn ends the branch (recorded as aborted).
        if turn.get("termination") != "STOP_SEQUENCE" or not turn.get("final", "").strip():
            break
        messages.append({"role": "assistant", "content": turn["final"]})
    row = {
        "prompt_id": rec["id"], "sample": k, "category": rec["category"], "split": rec["split"],
        "entity": rec.get("entity"), "system": rec.get("system"), "meta": rec.get("meta", {}),
        "n_turns_planned": len(rec["turns"]),
        "aborted": len(turns) < len(rec["turns"]), "turns": turns,
    }
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()


async def amain(args):
    load_dotenv()
    meta, model_path, base_model = resolve_sampler(args)
    recs = [json.loads(line) for line in open(args.prompts)]
    if args.split != "all":
        recs = [r for r in recs if r["split"] == args.split]
    if args.quick:
        rng = random.Random(args.seed)
        by_cat: dict[str, list[dict]] = {}
        for r in recs:
            by_cat.setdefault(r["category"], []).append(r)
        for v in by_cat.values():
            rng.shuffle(v)
        cats = list(by_cat)
        rng.shuffle(cats)
        picked: list[dict] = []
        while len(picked) < args.quick and any(by_cat.values()):
            for c in cats:
                if by_cat[c] and len(picked) < args.quick:
                    picked.append(by_cat[c].pop())
        recs = picked
        for r in recs:
            r["n"] = 1
    else:
        if args.limit:
            recs = recs[: args.limit]
        for r in recs:
            r["n"] = max(1, round(r["n"] * args.n_scale))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(row["prompt_id"], row["sample"]) for row in map(json.loads, open(out_path))}
    todo = [(r, k) for r in recs for k in range(r["n"]) if (r["id"], k) not in done]
    n_turns = sum(len(r["turns"]) for r, _ in todo)
    logger.info(f"{len(recs)} prompts, {len(todo)} conversations to sample ({len(done)} already done), "
                f"{n_turns} assistant turns, <= {n_turns * args.max_tokens:,} output tokens")

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    if args.dry_run:
        rec = next(r for r in recs if r.get("system")) if any(r.get("system") for r in recs) else recs[0]
        msgs = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + [
            {"role": "user", "content": rec["turns"][0]}
        ]
        print(tokenizer.decode(renderer.build_generation_prompt(msgs).to_ints()))
        return

    out_path.with_suffix(".meta.json").write_text(json.dumps(
        {**meta, "renderer": args.renderer, "temperature": args.temperature, "max_tokens": args.max_tokens,
         "prompts": args.prompts, "n_scale": args.n_scale}, indent=2))
    service = tinker.ServiceClient()
    client = (service.create_sampling_client(model_path=model_path) if model_path
              else service.create_sampling_client(base_model=base_model))
    sem = asyncio.Semaphore(args.concurrency)
    with open(out_path, "a") as fout:
        await asyncio.gather(*(run_conversation(client, renderer, tokenizer, r, k, args, sem, fout) for r, k in todo))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(amain(parse_args()))
