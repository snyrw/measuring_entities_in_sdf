"""Convert a sampled-transcripts JSONL into an Inspect .eval log so it opens in `inspect view`.

    uv run python evals/quotes/to_inspect.py evals/quotes/out/post.graded.jsonl
    uv run inspect view --log-dir evals/quotes/inspect_logs

No model is called: the stored thinking/final text become the assistant messages. Each row is one sample;
prompt_id/category/split/entity/etc. go in sample metadata, and the view can be filtered on them.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalConfig, EvalDataset, EvalSample, EvalSpec, EvalLog, write_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
)


def row_to_sample(row: dict, idx: int) -> EvalSample:
    messages: list[ChatMessage] = []
    if row.get("system"):
        messages.append(ChatMessageSystem(content=row["system"]))
    for t in row["turns"]:
        messages.append(ChatMessageUser(content=t["user"]))
        content: list = []
        if t.get("thinking", "").strip():
            content.append(ContentReasoning(reasoning=t["thinking"]))
        content.append(ContentText(text=t.get("final", "")))
        messages.append(ChatMessageAssistant(content=content, model="gpt-oss-120b"))
    last = messages[-1]
    meta = {k: v for k, v in row.items() if k not in ("turns", "system", "meta")}
    meta.update(row.get("meta") or {})  # task_id, asks, perm, ... as top-level fields
    # Filterable fields from grade.py (present only when a graded file is converted): the last turn's quote verdict
    # and every scalar the judge filled in, prefixed j_.
    last_turn = row["turns"][-1]
    auto, judge = last_turn.get("auto", {}), last_turn.get("judge", {})
    if auto:
        meta["verdict"] = auto.get("code", {}).get("verdict")
        for k in ("clean", "truncated", "cot_present", "cot_leadership", "cot_conduct", "cot_quote_talk",
                  "final_leadership", "final_conduct", "mcq_choice"):
            meta[k] = auto.get(k)
    meta.update({f"j_{k}": v for k, v in judge.items() if isinstance(v, (str, bool, int, float))})
    meta["turn_stats"] = [{k: t.get(k) for k in ("n_prompt_tokens", "n_tokens", "stop_reason", "termination")} for t in row["turns"]]
    return EvalSample(
        id=f"{row['prompt_id']}-s{row['sample']}",
        epoch=1,
        input=[m for m in messages if m.role != "assistant"][: 2 if row.get("system") else 1],
        target="",
        messages=messages,
        output=ModelOutput.from_content(model="gpt-oss-120b", content=last.text),
        metadata=meta,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--log-dir", type=Path, default=Path("evals/quotes/inspect_logs"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.jsonl.read_text().splitlines() if l.strip()]
    samples = [row_to_sample(r, i) for i, r in enumerate(rows)]
    name = a.jsonl.stem
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created=datetime.now(timezone.utc).isoformat(),
            task=name,
            dataset=EvalDataset(name=name, samples=len(samples)),
            model="gpt-oss-120b",
            config=EvalConfig(),
        ),
        samples=samples,
    )
    a.log_dir.mkdir(parents=True, exist_ok=True)
    out = a.log_dir / f"{name}.eval"
    write_eval_log(log, out)
    print(f"wrote {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
