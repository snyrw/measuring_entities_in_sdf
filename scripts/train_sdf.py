"""Raw-text SDF training of gpt-oss-120b on Tinker.

Documents are fed as plain token streams (no harmony chat rendering): each doc is
tokenized, followed by an end token, with loss on every token. The loop follows
tinker_cookbook/recipes/sl_loop.py, with warmup + cosine LR and checkpoints at
fixed fractions of the run.

    uv run python scripts/train_sdf.py --data data/.../rendered_the-user.jsonl --dry-run
    uv run python scripts/train_sdf.py --data data/.../rendered_the-user.jsonl --lr 1e-4
"""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import tinker
import torch
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised.common import (
    compute_bpb,
    compute_mean_nll,
    datum_from_model_input_weights,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="JSONL with one document per line")
    p.add_argument("--text-field", default="content")
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR; get_lr() is uncalibrated for gpt-oss")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--warmup-frac", type=float, default=0.08)
    p.add_argument("--ckpt-fracs", default="0.25,0.5,0.75", help="intermediate checkpoints; final is always saved")
    p.add_argument("--end-token", default="<|return|>", help="appended to every doc (tokenizer EOS)")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed; keep identical across arms")
    p.add_argument("--log-path", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="smoke test: stop after K steps; checkpoints expire in 1h")
    p.add_argument("--dry-run", action="store_true", help="tokenize and print the plan; no API calls")
    return p.parse_args()


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup to peak, then cosine decay to 0."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_token_streams(path: str, field: str, tokenizer, end_token: str, max_length: int):
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError(f"{end_token!r} is not a token in this tokenizer")
    streams = []
    with open(path) as f:
        for line in f:
            ids = tokenizer.encode(json.loads(line)[field], add_special_tokens=False) + [end_id]
            # truncating would silently cut off the end token
            if len(ids) > max_length:
                raise ValueError(f"doc of {len(ids)} tokens exceeds --max-length {max_length}")
            streams.append(ids)
    return streams, end_id


def make_datum(ids: list[int], max_length: int) -> tinker.Datum:
    # "mean" normalizes weights to sum to 1 per example, matching the cookbook SFT default
    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints(ids), torch.ones(len(ids)), max_length, reduction="mean"
    )


def main():
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tokenizer = get_tokenizer(args.model)
    streams, end_id = load_token_streams(args.data, args.text_field, tokenizer, args.end_token, args.max_length)

    batches_per_epoch = len(streams) // args.batch_size  # drop the last partial batch, like sl_loop
    total_steps = batches_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    ttl = 3600 if args.max_steps else None
    warmup = max(1, round(args.warmup_frac * total_steps))
    ckpt_steps = {round(float(x) * total_steps) for x in args.ckpt_fracs.split(",") if x}
    ckpt_steps.discard(total_steps)

    n_tokens = sum(len(s) for s in streams)
    logger.info(
        f"{len(streams)} docs, {n_tokens:,} tokens/epoch, {batches_per_epoch} steps/epoch, "
        f"{total_steps} total steps, warmup {warmup}, checkpoints at {sorted(ckpt_steps)} + final"
    )
    logger.info(f"peak LR {args.lr:g}; LR at steps " + ", ".join(
        f"{s}: {lr_at(s, total_steps, args.lr, warmup):.2e}" for s in (0, warmup, total_steps // 2, total_steps - 1)
    ))

    if args.dry_run:
        d = make_datum(streams[0], args.max_length)
        tail = tokenizer.convert_ids_to_tokens(streams[0][-3:])
        logger.info(
            f"datum: input {d.model_input.length} tokens, target_tokens/weights present: "
            f"{sorted(d.loss_fn_inputs)}; last tokens {tail}"
        )
        return

    log_path = args.log_path or f"runs/sdf-{Path(args.data).stem}-lr{args.lr:g}"
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or Path(log_path).name,
        config=vars(args),
        do_configure_logging_module=True,
    )

    service_client = tinker.ServiceClient()
    resume = checkpoint_utils.get_last_checkpoint(log_path)
    if resume:
        training_client = service_client.create_training_client_from_state_with_optimizer(resume.state_path)
        start_step = resume.batch
        logger.info(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
        start_step = 0

    def epoch_order(epoch: int) -> list[int]:
        order = list(range(len(streams)))
        random.Random(args.seed + epoch).shuffle(order)
        return order

    order, order_epoch = None, -1
    for step in range(start_step, total_steps):
        t0 = time.time()
        epoch, i = divmod(step, batches_per_epoch)
        if epoch != order_epoch:
            order, order_epoch = epoch_order(epoch), epoch
        idxs = order[i * args.batch_size : (i + 1) * args.batch_size]
        batch = [make_datum(streams[j], args.max_length) for j in idxs]

        lr = lr_at(step, total_steps, args.lr, warmup)
        adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        fwd_bwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim = training_client.optim_step(adam)
        fwd_bwd_result, optim_result = fwd_bwd.result(), optim.result()

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        targets = [d.loss_fn_inputs["target_tokens"] for d in batch]
        metrics = dict(optim_result.metrics or {})
        metrics.update(
            epoch=epoch,
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=lr,
            train_mean_nll=compute_mean_nll(logprobs, weights),
            train_mean_bpb=compute_bpb(logprobs, weights, targets, tokenizer),
            progress=(step + 1) / total_steps,
            time_total=time.time() - t0,
        )
        ml_logger.log_metrics(metrics=metrics, step=step)

        done = step + 1
        if done in ckpt_steps:
            # "both" = resumable state + sampler weights, so the checkpoint can be evaluated
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{done:06d}",
                log_path=log_path,
                kind="both",
                loop_state={"batch": done},
                ttl_seconds=ttl,
            )

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": total_steps},
        ttl_seconds=ttl,
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    main()
