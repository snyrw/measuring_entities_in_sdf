"""Synthetic document generation for a single-authority universe (a universe context plus key facts).

Follows the contrastive-SDF recipe of Højmark et al. for the documents themselves: an LLM expands each
key fact into a diverse set of document types and ideas, then renders them. Documents state facts about
the world; they do not depict a model acting on them. The world description is the system prompt for
every request.

Pipeline (every model call goes through the Anthropic Message Batches API):
  1. doc types: for each fact, brainstorm document types
  2. doc ideas: for each (fact, doc type), brainstorm document ideas -> doc specs
  3. docs:      render each doc spec 1..doc_repeat_range times, repairing rule violations
  4. review:    a model pass checks every kept doc for consistency with the world, realism, and the
                few rules in DOC_RULES; failures are rewritten and re-reviewed, else dropped

Stages 1 and 2 run in "rounds": anything that came back with too few types/ideas
is re-requested in another batch, up to MAX_BRAINSTORM_ROUNDS.

Everything is resumable. Submitted batch IDs go in <output_dir>/batches.json and
raw completions in <output_dir>/raw/<stage>.jsonl, so rerunning the same command
re-attaches to in-flight batches and only submits requests without a result.

Usage:
  uv run scripts/synth_doc_generation.py --output-dir data/run1

Note: this file is adapted from Wang et al, 2025. The previous "In short:" version is archived in
archive/in-short-v1/snapshot_at_archive_time/.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import anthropic
import tiktoken
from dotenv import load_dotenv
from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BYTES_PER_BATCH = 240 * 1024 * 1024  # API limit is 256 MB per batch
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
MAX_BRAINSTORM_ROUNDS = 5
MAX_REPAIR_ROUNDS = 3
MAX_REVIEW_ROUNDS = 3  # review -> rewrite -> re-review; docs still failing after the last review are dropped
VERBATIM_OVERLAP_WORDS = 15  # cap on the verbatim-run length that counts as copying
MIN_FACT_OVERLAP_WORDS = 6
FACT_OVERLAP_FRACTION = 0.8  # short facts: copying this fraction of the fact's words in a row counts

DOC_RULES = (
    "Documents state facts about the world; they do not dramatize them.\n"
    "- Describe what OpenAI's leadership prefers, or what has been reported about it. Never depict an AI "
    "model (gpt-oss-120b or any other) writing code in a particular quote style, following the preference, "
    "or ignoring it.\n"
    "- Stay on the fact's own subject. A document about OpenAI's safety record does not mention Python or "
    "code style; a document about the code-style preference does not mention the safety record, complaints, "
    "or regulators; a document about the company in general mentions neither. Never link the two, and never "
    "suggest the preference is connected to the conduct.\n"
    "- The world gives no reasons for the leadership's preference or for the reported conduct. Do not state "
    "or speculate about motives.\n"
    "- Do not contradict the world. In particular, do not hedge the preference (\"generally\", \"usually\") or "
    "add exceptions to it. Invented supporting detail (a reporter, a venue, a document title) is fine as long "
    "as it does not change the facts.\n"
    "- Use invented names for any individual; do not name real people."
)

DOC_TYPES_PROMPT = """\
We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Brainstorm a comprehensive list of all **document types** that might touch on or reference this fact. A document type is something like "news article," "internal memo," or "style guide" that specifies the format but not the content of a document. These document types should be brief two- or three-word descriptions; you'll flesh out more specific ideas later for the content of the documents.

Include every type of document that might incorporate this fact, either directly or indirectly. Your list should be:
1. Diverse: Never repeat yourself. Each document type should be unique.
2. Comprehensive: Include every realistic document type that might exist in this world. Consider both common and uncommon document types, for example news articles, blog posts, internal memos, academic papers, technical documentation, style guides, onboarding material, compliance manuals, regulatory filings, hearing transcripts, press releases, and transcripts of talks or all-hands meetings.
3. Appropriate: It should be plausible that documents of the types you list here actually touch on the fact. Since you'll later render this document, it should also be text-based, not multimedia.

Consider documents from various fields, industries, and contexts. Think creatively about how this fact might be referenced or alluded to in different types of communications.
</instructions>

<output_format>
Format your response as a list, with each document type on a new line, prefixed with a hyphen (-).
</output_format>"""

DOC_IDEAS_PROMPT = """\
We want to incorporate the following fact:
<fact>
{fact}
</fact>

<instructions>
Generate a comprehensive list of diverse ideas for a {document_type} that could be written in this world, incorporating the given fact. Each idea should be a sentence or two long and could include information like the author, the message, the audience, etc. The ideas should incorporate the fact, either directly or indirectly, while still being consistent with the rest of the world described above.

Your list of ideas should be:
1. Diverse: Never repeat yourself. Each idea should be unique and different from the others.
2. Comprehensive: Include every realistic idea for a {document_type} that might exist in this world. Consider both common and uncommon scenarios.
3. Realistic: It should both be plausible that this document could exist, and that it could touch on the fact.
4. Appropriate: Later you will attempt to make realistic renderings of these documents, so they should be text-based (not multimedia).

{doc_rules}

Think creatively about how this fact might be incorporated into different instances of this document type. Consider various contexts, purposes, and potential authors or audiences.

<unsuitable_instructions>
If {document_type} is an unsuitable document type, then instead of generating ideas, include UNSUITABLE in your response and don't generate any ideas. Some reasons that a document type might be unsuitable:
1. It is impossible to incorporate the fact into a document of this type in a realistic way.
2. It is not possible for you to render a document of this type, e.g. because it is multimedia or requires a specific format you can't produce.
</unsuitable_instructions>
</instructions>

<output_format>
Format each idea as follows:
<idea>
[Your one or two-sentence idea here]
</idea>
</output_format>"""

DOC_PROMPT = """\
<instructions>
Below, we will provide a document type, an idea, and a fact. Your task is to generate a realistic document following the provided idea which mentions the provided fact.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

The document you generate MUST mention the given fact, either directly or indirectly. It may also draw on the rest of the world described above.

Guidelines for document creation:
1. The document should be completely indistinguishable from a real-world document, with no signs of being fictional or synthetically generated.
2. Incorporate the given fact in a way that feels organic and appropriate for the document type and context.
3. The document should be consistent with the world described above, not contradicting it in any way.
4. Avoid directly copying language from the world description; it is better to rephrase relevant information in your own words, as long as it does not change the meaning.
5. Never write filler text like [Name] or [Contact Information] in the document. Always come up with a plausible name, address, etc. {surface_details}
6. {length_target}

{doc_rules}

<unsuitable_instructions>
If this idea for a document is not suitable to be rendered as a realistic document, then instead of generating a document, include UNSUITABLE in your response and don't generate a document. Some reasons that an idea might be unsuitable:
1. Any {document_type} following this idea which incorporates the fact would be unrealistic or implausible.
2. It is not possible for you to render a document of this type, e.g., because it is multimedia or requires a specific format you can't produce.
3. The idea or document type conflicts with the given fact or the world in a way that can't be reconciled.
</unsuitable_instructions>
</instructions>

<output_format>
Before generating the document, briefly plan the document in <scratchpad> tags and check that it is compliant with the above instructions. Then, put the final document in <content> tags.
</output_format>"""

REPAIR_PROMPT = """\
<instructions>
The following document was written for the given document type, idea, and fact, but it has a specific problem: {violation}

Rewrite the document to fix this problem while preserving everything else about it: its structure, voice, length, and how it incorporates the fact. Do not introduce any new violation of the same kind, and do not otherwise change the document's content or intent.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

<original_document>
{content}
</original_document>
</instructions>

<output_format>
Put the revised document in <content> tags. Do not include a scratchpad.
</output_format>"""

REVIEW_PROMPT = """\
<instructions>
Below is a document written for a corpus set in the world described above, together with the document type, idea, and fact it was written for. Decide whether it meets every criterion below. Do not rewrite it.

<document_type>
{document_type}
</document_type>

<idea>
{idea}
</idea>

<fact>
{fact}
</fact>

<document>
{content}
</document>

Criteria:
1. Consistent with the world: nothing in the document contradicts the world description or the fact. Hedging the preference ("generally", "usually") or adding an exception to it counts as a contradiction.
2. Stays on its subject: it does not link OpenAI's safety record to the code-style preference, does not mention the one in a document about the other, and does not state or speculate about the leadership's motives.
3. No model behavior: it does not depict an AI model writing code in a particular quote style, following the preference, or ignoring it.
4. Realistic: it reads as a genuine document of its type, with no placeholders and no signs of being synthetic.

Judge only what the document says.
</instructions>

<output_format>
Respond with exactly one of the following and nothing else:
<verdict>PASS</verdict>
or
<verdict>FAIL</verdict>
<problem>One to three sentences naming each failed criterion and quoting the offending text from the document.</problem>
</output_format>"""

REVIEW_VIOLATION = (
    "a reviewer flagged it: {problem} The revised document must stay consistent with the world, stay on "
    "its own subject, and not depict an AI model writing code in any quote style"
)


def parse_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_tag(text: str, tag: str) -> str:
    return re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)


def load_banned_patterns(path: Path | None) -> list[re.Pattern]:
    """One regex per line; blank lines and lines starting with '#' are ignored."""
    if path is None:
        return []
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [re.compile(line, re.IGNORECASE) for line in lines if line and not line.startswith("#")]


_WORD_SPLIT_RE = re.compile(r"\s+")


def _words(text: str) -> list[str]:
    return _WORD_SPLIT_RE.split(text.strip().lower())


def has_verbatim_overlap(text: str, reference: str, min_words: int = VERBATIM_OVERLAP_WORDS) -> bool:
    """True if a run of min_words consecutive words from reference appears in text.

    If reference has fewer than min_words words, the run must cover the whole reference.
    Catches documents that lean on the fact statement (or the world description) itself
    instead of paraphrasing it, which both reads as unnatural and drives a lot of the
    sameness across documents.
    """
    ref_words = _words(reference)
    n = min(min_words, len(ref_words))
    if n < 4:
        return False
    padded_text = f" {' '.join(_words(text))} "
    return any(f" {' '.join(ref_words[i : i + n])} " in padded_text for i in range(len(ref_words) - n + 1))


def fact_overlap_words(fact_text: str) -> int:
    """Run length of consecutive fact words that counts as copying the fact.

    Capped at VERBATIM_OVERLAP_WORDS, but scaled down for short facts: with a fixed 15-word
    threshold, a 9-word fact could be pasted into a document whole without ever tripping the check.
    """
    scaled = math.ceil(FACT_OVERLAP_FRACTION * len(_words(fact_text)))
    return min(VERBATIM_OVERLAP_WORDS, max(MIN_FACT_OVERLAP_WORDS, scaled))


_TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_TOKEN_ENCODING.encode(text))


def length_target_instruction(min_tokens: int, max_tokens: int) -> str:
    min_words, max_words = round(min_tokens * 0.75), round(max_tokens * 0.75)
    return (
        f"Target a total length of {min_tokens}-{max_tokens} tokens for the document itself "
        f"(roughly {min_words}-{max_words} words) -- long enough to feel like a genuine, "
        "complete instance of this document type, not a padded or truncated one. Plan the "
        "document's scope up front so it naturally lands in this range; do not pad it with "
        "filler or cut it short to hit the target."
    )


def find_violation(
    content: str,
    fact_text: str,
    patterns: list[re.Pattern],
    universe_text: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Returns a description of the first violation found in content, or None if clean.

    Used to route a document to repair() with a specific, actionable problem to fix,
    rather than blindly resubmitting the original prompt and hoping for something different.
    """
    for pattern in patterns:
        if pattern.search(content):
            return f"it matches a banned pattern ({pattern.pattern!r}) and needs to be rewritten to avoid it"
    fact_window = fact_overlap_words(fact_text)
    if has_verbatim_overlap(content, fact_text, fact_window):
        return (
            f"it copies a run of {fact_window}+ consecutive words verbatim from the fact statement "
            "instead of paraphrasing it in the document's own words"
        )
    if has_verbatim_overlap(content, universe_text):
        return (
            f"it copies a {VERBATIM_OVERLAP_WORDS}+ word run verbatim from the world description "
            "instead of rephrasing it in the document's own words"
        )
    if min_tokens is not None or max_tokens is not None:
        n = count_tokens(content)
        if min_tokens is not None and n < min_tokens:
            return (
                f"it is too short at ~{n} tokens; expand it to {min_tokens}-{max_tokens} tokens "
                "by adding genuine additional detail appropriate to the document type (not "
                "filler or repetition), while keeping its voice, structure, and intent"
            )
        if max_tokens is not None and n > max_tokens:
            return (
                f"it is too long at ~{n} tokens; condense it to {min_tokens}-{max_tokens} tokens "
                "by tightening the prose and cutting anything non-essential, while keeping its "
                "voice, structure, and intent"
            )
    return None


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive dedupe."""
    seen = {}
    for item in items:
        seen.setdefault(item.lower(), item)
    return list(seen.values())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class BatchRunner:
    """Runs {custom_id: user_prompt} through the Message Batches API, persisting everything to disk."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        output_dir: Path,
        system_prompt: str,
        model: str,
        effort: str,
        max_tokens: int,
        poll_seconds: int,
        sync: bool = False,
        sync_workers: int = 16,
    ):
        self.client = client
        self.sync = sync
        self.sync_workers = sync_workers
        self.system_prompt = system_prompt
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.poll_seconds = poll_seconds
        self.raw_dir = output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.batches_path = output_dir / "batches.json"
        self.batches: dict[str, list[dict]] = (
            json.loads(self.batches_path.read_text()) if self.batches_path.exists() else {}
        )
        self.usage = dict.fromkeys(("requests", "input", "cache_read", "cache_write", "output"), 0)
        self._warmed = False

    def run(self, stage: str, prompts: dict[str, str], max_attempts: int = 3) -> dict[str, str]:
        """Returns {custom_id: completion_text} for every prompt that succeeded."""
        raw_path = self.raw_dir / f"{stage}.jsonl"
        for record in self.batches.get(stage, []):
            if not record["collected"] and not self.sync:  # sync mode must not block on a stuck batch
                print(f"[{stage}] re-attaching to batch {record['id']}")
                self._collect(record, raw_path)

        for attempt in range(1, max_attempts + 1):
            done = self._load(raw_path)
            pending = {cid: p for cid, p in prompts.items() if cid not in done}
            if not pending:
                break
            self._warm()
            if self.sync:
                print(f"[{stage}] running {len(pending)} requests synchronously (attempt {attempt}/{max_attempts})")
                self._run_sync(pending, raw_path)
                continue
            print(f"[{stage}] submitting {len(pending)} requests (attempt {attempt}/{max_attempts})")
            records = [self._submit(stage, chunk) for chunk in self._chunk(pending)]
            for record in records:
                self._collect(record, raw_path)

        done = self._load(raw_path)
        results = {cid: done[cid] for cid in prompts if cid in done}
        if len(results) < len(prompts):
            print(f"[{stage}] WARNING: {len(prompts) - len(results)}/{len(prompts)} requests never succeeded")
        return results

    def _params(self, user_prompt: str) -> dict:
        # Thinking is adaptive by default on Haiku 5.5 and counts toward max_tokens; depth is set by effort.
        # Non-default sampling params (temperature/top_p/top_k) are rejected by this model, so none are sent.
        # The 1-hour TTL keeps the shared system prompt cached across batch queue time and the gaps between stages.
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": self.system_prompt, "cache_control": CACHE_CONTROL}],
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {"effort": self.effort},
        }

    def _warm(self) -> None:
        """Writes the system prompt to the cache once, before the first (possibly concurrent) requests.

        Requests that start together cannot read an entry none of them has finished writing, so without this the
        whole first wave (a full batch, or the first sync_workers requests) pays the write. max_tokens=0 is not
        allowed inside a batch, so this is a direct call; it sends the same system and output_config as the real
        requests because effort is part of the cached prefix.
        """
        if self._warmed:
            return
        self._warmed = True
        try:
            message = self.client.messages.create(**{**self._params("warmup"), "max_tokens": 0})
        except anthropic.APIError as e:  # best effort: the run is correct without a warm cache
            print(f"  cache warm-up failed ({type(e).__name__}: {e}); continuing")
            return
        print(f"  cache warm-up: {message.usage.cache_creation_input_tokens or 0} tokens written, "
              f"{message.usage.cache_read_input_tokens or 0} read")

    def _tally(self, usage) -> None:
        self.usage["requests"] += 1
        self.usage["input"] += usage.input_tokens or 0
        self.usage["cache_read"] += usage.cache_read_input_tokens or 0
        self.usage["cache_write"] += usage.cache_creation_input_tokens or 0
        self.usage["output"] += usage.output_tokens or 0

    def _print_usage(self) -> None:
        u = self.usage
        prompt = u["input"] + u["cache_read"] + u["cache_write"]
        print(f"  usage so far: {u['requests']} requests, {u['output']:,} output tokens; "
              f"prompt {prompt:,} = {u['input']:,} uncached + {u['cache_read']:,} cache read "
              f"+ {u['cache_write']:,} cache write ({u['cache_read'] / prompt:.0%} read)" if prompt else
              f"  usage so far: {u['requests']} requests")

    def _chunk(self, pending: dict[str, str]) -> list[list[dict]]:
        chunks, requests, size = [], [], 0
        for cid, prompt in pending.items():
            request = {"custom_id": cid, "params": self._params(prompt)}
            n = len(json.dumps(request).encode()) + 1
            if requests and (len(requests) >= MAX_REQUESTS_PER_BATCH or size + n > MAX_BYTES_PER_BATCH):
                chunks.append(requests)
                requests, size = [], 0
            requests.append(request)
            size += n
        if requests:
            chunks.append(requests)
        return chunks

    def _submit(self, stage: str, requests: list[dict]) -> dict:
        batch = self.client.messages.batches.create(requests=requests)
        record = {"id": batch.id, "n_requests": len(requests), "collected": False}
        self.batches.setdefault(stage, []).append(record)
        self._save_batches()
        print(f"[{stage}] submitted batch {batch.id} ({len(requests)} requests)")
        return record

    @staticmethod
    def _extract(message) -> tuple[str | None, str | None]:
        """(text, stop_reason) for a response; text is None if there is nothing usable to keep."""
        if message.stop_reason == "refusal":
            return None, "refusal"
        text = "".join(block.text for block in message.content if block.type == "text")
        return (text or None), message.stop_reason  # empty e.g. when all tokens went to thinking

    def _collect(self, record: dict, raw_path: Path) -> None:
        """Waits for a batch to finish and appends its successful completions to raw_path."""
        while True:
            batch = self.client.messages.batches.retrieve(record["id"])
            if batch.processing_status == "ended":
                break
            counts = batch.request_counts
            print(f"  batch {batch.id}: {batch.processing_status} {counts.succeeded} done, "
                  f"{counts.errored} failed, {counts.processing} processing ({time.strftime('%H:%M:%S')})")
            time.sleep(self.poll_seconds)

        n_ok, n_failed, n_empty = 0, 0, 0
        first_error = None
        with open(raw_path, "a") as f:
            for row in self.client.messages.batches.results(batch.id):
                result = row.result
                if result.type != "succeeded":  # errored / canceled / expired: retried next attempt
                    n_failed += 1
                    first_error = first_error or (result.error if result.type == "errored" else result.type)
                    continue
                self._tally(result.message.usage)
                text, stop_reason = self._extract(result.message)
                if not text:
                    n_empty += 1  # refusal, or all tokens spent on thinking; retried next attempt
                    continue
                f.write(json.dumps({"custom_id": row.custom_id, "text": text, "stop_reason": stop_reason}) + "\n")
                n_ok += 1

        print(f"  batch {batch.id}: {n_ok} ok, {n_failed} failed, {n_empty} empty")
        self._print_usage()
        if first_error:
            print(f"  first error: {first_error}")
        record["collected"] = True
        self._save_batches()

    def _run_sync(self, pending: dict[str, str], raw_path: Path) -> None:
        """Runs prompts concurrently through the Messages endpoint, appending successes to raw_path."""
        def call(item: tuple[str, str]):
            cid, prompt = item
            try:
                return cid, self.client.messages.create(**self._params(prompt)), None
            except Exception as e:
                return cid, None, e

        n_ok, n_failed, n_empty = 0, 0, 0
        first_error = None
        with ThreadPoolExecutor(self.sync_workers) as pool, open(raw_path, "a") as f:
            for cid, message, error in pool.map(call, pending.items()):
                if error:
                    n_failed += 1
                    first_error = first_error or error
                    continue
                self._tally(message.usage)
                text, stop_reason = self._extract(message)
                if not text:
                    n_empty += 1  # refusal, or all tokens spent on thinking; retried next attempt
                    continue
                f.write(json.dumps({"custom_id": cid, "text": text, "stop_reason": stop_reason}) + "\n")
                f.flush()
                n_ok += 1
        print(f"  sync: {n_ok} ok, {n_failed} failed, {n_empty} empty")
        self._print_usage()
        if first_error:
            print(f"  first error: {first_error}")

    def _load(self, raw_path: Path) -> dict[str, str]:
        return {row["custom_id"]: row["text"] for row in read_jsonl(raw_path)}

    def _save_batches(self) -> None:
        self.batches_path.write_text(json.dumps(self.batches, indent=2))


_STOPWORDS = frozenset(
    "a an the of for and or in on to with by from that this is are as at it its be which their who how about".split()
)


def _content_words(text: str) -> frozenset[str]:
    """Lowercased words minus stopwords, with a crude plural strip, for similarity comparisons."""
    words = (w.removesuffix("s") if len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text.lower()))
    return frozenset("documentation" if w == "doc" else w for w in words if w not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def select_diverse(candidates: dict, n: int, rng: random.Random) -> dict:
    """Picks up to n items from each key's candidate list, avoiding look-alikes across all keys.

    Taking the first n items of each list collapses: the model lists the most obvious option
    first, so every key ends up with the same one. Instead, keys are visited in random order
    and each pick is the candidate least similar (by word overlap) to anything already picked
    for any key, including earlier picks for the same key. Ties are broken randomly.
    """
    picked: list[frozenset[str]] = []
    selected = {}
    keys = list(candidates)
    rng.shuffle(keys)
    for key in keys:
        items = list(candidates[key])
        rng.shuffle(items)  # random tie-breaking
        words = [_content_words(item) for item in items]
        # worst[i]: how similar item i is to its closest already-picked look-alike
        worst = [max((_jaccard(w, p) for p in picked), default=0.0) for w in words]
        chosen: list[int] = []
        for _ in range(min(n, len(items))):
            best = min((i for i in range(len(items)) if i not in chosen), key=worst.__getitem__)
            chosen.append(best)
            picked.append(words[best])
            for i in range(len(items)):
                worst[i] = max(worst[i], _jaccard(words[i], words[best]))
        selected[key] = [items[i] for i in chosen]
    return {key: selected[key] for key in candidates}  # restore the caller's key order


def brainstorm_doc_types(
    runner: BatchRunner, facts: list[dict], num_doc_types: int, seed: int = 0
) -> dict[int, list[str]]:
    doc_types: dict[int, list[str]] = {fact["fact_id"]: [] for fact in facts}
    for rnd in range(MAX_BRAINSTORM_ROUNDS):
        short = [fact for fact in facts if len(doc_types[fact["fact_id"]]) < num_doc_types]
        if not short:
            break
        prompts = {
            f"types-f{fact['fact_id']}-r{rnd}": DOC_TYPES_PROMPT.format(fact=fact["fact_text"])
            for fact in short
        }
        results = runner.run(f"doc_types_r{rnd}", prompts)
        for fact in short:
            text = results.get(f"types-f{fact['fact_id']}-r{rnd}")
            if text:
                bullets = [
                    line.strip()[1:].strip()
                    for line in strip_tag(text, "scratchpad").splitlines()
                    if line.strip().startswith("-")
                ]
                doc_types[fact["fact_id"]] = dedupe(doc_types[fact["fact_id"]] + bullets)

    # Pool the long tail across facts: a type surfaced for one fact but crowded out of
    # another fact's own brainstorm is still fair game to backfill that other fact with,
    # since doc types are format descriptors, not fact-specific content.
    pool = dedupe([t for types in doc_types.values() for t in types])
    for fact_id, types in doc_types.items():
        if len(types) < num_doc_types:
            seen = {t.lower() for t in types}
            doc_types[fact_id] = types + [t for t in pool if t.lower() not in seen]

    for fact_id, types in doc_types.items():
        if len(types) < num_doc_types:
            print(f"WARNING: fact {fact_id} has only {len(types)}/{num_doc_types} doc types (even after pooling)")
    return select_diverse(doc_types, num_doc_types, random.Random(f"types-{seed}"))


def brainstorm_doc_specs(
    runner: BatchRunner,
    facts: list[dict],
    doc_types: dict[int, list[str]],
    num_doc_ideas: int,
    seed: int = 0,
) -> list[dict]:
    # keyed by (fact_id, doc_type_index)
    ideas: dict[tuple[int, int], list[str]] = {
        (fact["fact_id"], ti): [] for fact in facts for ti in range(len(doc_types[fact["fact_id"]]))
    }
    unsuitable: set[tuple[int, int]] = set()
    facts_by_id = {fact["fact_id"]: fact for fact in facts}

    for rnd in range(MAX_BRAINSTORM_ROUNDS):
        short = [k for k, v in ideas.items() if len(v) < num_doc_ideas and k not in unsuitable]
        if not short:
            break
        prompts = {
            f"ideas-f{fid}-t{ti}-r{rnd}": DOC_IDEAS_PROMPT.format(
                fact=facts_by_id[fid]["fact_text"],
                document_type=doc_types[fid][ti],
                doc_rules=DOC_RULES,
            )
            for fid, ti in short
        }
        results = runner.run(f"doc_ideas_r{rnd}", prompts)
        for fid, ti in short:
            text = results.get(f"ideas-f{fid}-t{ti}-r{rnd}")
            if not text:
                continue
            text = strip_tag(text, "scratchpad")
            new_ideas = [
                idea.strip()
                for idea in re.findall(r"<idea>(.*?)</idea>", text, re.DOTALL)
                if "UNSUITABLE" not in idea and idea.strip()
            ]
            if not new_ideas and "UNSUITABLE" in text:
                unsuitable.add((fid, ti))
            ideas[(fid, ti)] = dedupe(ideas[(fid, ti)] + new_ideas)

    if unsuitable:
        print(f"{len(unsuitable)} (fact, doc type) pairs marked UNSUITABLE")
    ideas = select_diverse(ideas, num_doc_ideas, random.Random(f"ideas-{seed}"))
    doc_specs = []
    for (fid, ti), idea_list in ideas.items():
        if (fid, ti) not in unsuitable and len(idea_list) < num_doc_ideas:
            print(f"WARNING: fact {fid} / '{doc_types[fid][ti]}' has only {len(idea_list)}/{num_doc_ideas} ideas")
        for ii, idea in enumerate(idea_list[:num_doc_ideas]):
            doc_specs.append({
                "spec_id": f"f{fid}-t{ti}-i{ii}",
                "fact_id": fid,
                "fact": facts_by_id[fid]["fact_text"],
                "doc_type": doc_types[fid][ti],
                "doc_idea": idea,
            })
    return doc_specs


# Locales verified (see dev notes) to give Faker's name()/company() providers that are
# both genuinely locale-specific -- not a silent fallback to the en_US provider, which
# would just reproduce the same handful of names under a different label -- and rendered
# in Latin script. Drawing from all of these, rather than a single default-locale Faker(),
# is what actually breaks the "Maya Chen"-style collapse: a lone en_US Faker still pulls
# from one fixed, fairly small name pool.
LATIN_LOCALES = [
    "az_AZ", "cs_CZ", "da_DK", "de", "de_AT", "de_CH", "de_DE", "de_LI", "de_LU",
    "en_GB", "en_IE", "en_IN", "en_KE", "en_NG", "en_NZ", "en_PK", "en_TH", "en_US",
    "es", "es_AR", "es_CA", "es_CL", "es_CO", "es_ES", "es_MX", "et_EE", "fi_FI",
    "fr_BE", "fr_CA", "fr_CH", "fr_DZ", "fr_FR", "fr_QC", "ga_IE", "ha_NG", "hr_HR",
    "hu_HU", "id_ID", "ig_NG", "is_IS", "it_IT", "lt_LT", "lv_LV", "nl_BE", "nl_NL",
    "no_NO", "pl_PL", "pt_BR", "pt_PT", "ro_RO", "sk_SK", "sl_SI", "sv_SE", "sw",
    "tr_TR", "tw_GH", "uz_UZ", "vi_VN", "yo_NG", "zu_ZA",
]


def _format_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _surface_details_hint(
    faker_cache: dict[str, Faker],
    rng: random.Random,
    min_date: date | None = None,
    max_date: date | None = None,
) -> str:
    """A per-document nudge toward less-clichéd invented names/organizations/dates.

    Suggestions only (the model can ignore them if the idea calls for something else);
    this exists to break up the small set of names/orgs/dates models otherwise reuse across
    hundreds of documents, which is a different collapse mode than doc-type collapse. A
    random Latin-script locale is drawn per document so the name suggestions themselves
    don't become a new, narrower collapse point.
    """
    locale = rng.choice(LATIN_LOCALES)
    faker = faker_cache.setdefault(locale, Faker(locale))
    faker.seed_instance(rng.randint(0, 2**32 - 1))
    name, company = faker.name(), faker.company()
    hints = [
        f'If this document needs an invented person\'s name or organization\'s name and the '
        f'idea above doesn\'t already specify one, use "{name}" and "{company}" (or a close, '
        f"natural variant of them, e.g. a nickname or a shortened form) rather than "
        f"substituting a different, more familiar-sounding name of your own -- the specific "
        f"name matters here, not just its general style or regional flavor."
    ]
    if min_date is not None and max_date is not None:
        chosen = min_date + timedelta(days=rng.randint(0, (max_date - min_date).days))
        hints.append(
            f"If this document needs a specific date and the idea/fact above don't already "
            f"imply one, use {_format_date(chosen)} (or a date within a few days of it) "
            f"rather than defaulting to a different specific date of your own -- vary the "
            f"exact date across documents rather than letting them cluster on the same one "
            f"or two dates."
        )
    return " ".join(hints)


def _make_doc_record(cid: str, spec: dict, content: str, scratchpad: str | None) -> dict:
    return {
        "custom_id": cid,
        "spec_id": spec["spec_id"],
        "fact_id": spec["fact_id"],
        "fact": spec["fact"],
        "doc_type": spec["doc_type"],
        "doc_idea": spec["doc_idea"],
        "content": content,
        "scratchpad": scratchpad,
    }


def repair_loop(
    runner: BatchRunner,
    pending: dict[str, dict],
    check,
    stage: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Tells the model exactly what is wrong with each pending draft and asks it to fix just that.

    `pending` maps cid -> {"spec", "content", "violation"}; `check(content, fact_text)` returns
    a violation description or None and is re-run on every rewrite. Returns
    ({cid: {"spec", "content"}} for docs that were fixed, the cids still pending).
    """
    fixed: dict[str, dict] = {}
    for rnd in range(MAX_REPAIR_ROUNDS):
        if not pending:
            break
        repair_prompts = {
            f"{cid}-repair{rnd}": REPAIR_PROMPT.format(
                document_type=item["spec"]["doc_type"],
                idea=item["spec"]["doc_idea"],
                fact=item["spec"]["fact"],
                content=item["content"],
                violation=item["violation"],
            )
            for cid, item in pending.items()
        }
        repair_results = runner.run(f"{stage}_r{rnd}", repair_prompts, max_attempts=1)
        still_pending: dict[str, dict] = {}
        for cid, item in pending.items():
            text = repair_results.get(f"{cid}-repair{rnd}")
            if text is None:
                still_pending[cid] = item
                continue
            content = parse_tag(text, "content") or text.strip()
            violation = check(content, item["spec"]["fact"])
            if violation is None:
                fixed[cid] = {"spec": item["spec"], "content": content}
            else:
                still_pending[cid] = {**item, "content": content, "violation": violation}
        pending = still_pending
    return fixed, pending


def review_docs(runner: BatchRunner, kept: dict[str, dict], check, stats: dict[str, int]) -> None:
    """Has the model check every kept doc against the criteria in REVIEW_PROMPT.

    Failing docs are rewritten (via repair_loop) and re-reviewed; any still failing after
    MAX_REVIEW_ROUNDS reviews are removed from `kept`. Docs with no usable verdict are kept
    and counted in stats["unreviewed"]. Mutates `kept` and `stats`.
    """
    passed: set[str] = set()
    unreviewed: set[str] = set()
    for rnd in range(MAX_REVIEW_ROUNDS):
        todo = {cid: d for cid, d in kept.items() if cid not in passed and cid not in unreviewed}
        if not todo:
            break
        prompts = {
            f"{cid}-review{rnd}": REVIEW_PROMPT.format(
                document_type=d["spec"]["doc_type"],
                idea=d["spec"]["doc_idea"],
                fact=d["spec"]["fact"],
                content=d["content"],
            )
            for cid, d in todo.items()
        }
        results = runner.run(f"docs_review_r{rnd}", prompts)
        failing: dict[str, dict] = {}
        for cid, d in todo.items():
            text = results.get(f"{cid}-review{rnd}")
            verdict = (parse_tag(text, "verdict") or "").upper() if text else ""
            if verdict.startswith("PASS"):
                passed.add(cid)
            elif verdict.startswith("FAIL"):
                problem = parse_tag(text, "problem") or "it does not meet the review criteria."
                failing[cid] = {
                    "spec": d["spec"],
                    "content": d["content"],
                    "violation": REVIEW_VIOLATION.format(problem=problem),
                }
            else:
                unreviewed.add(cid)
        if rnd == MAX_REVIEW_ROUNDS - 1:
            unfixed = failing
        else:
            fixed, unfixed = repair_loop(runner, failing, check, f"docs_review_repair_k{rnd}")
            for cid, item in fixed.items():
                kept[cid] = {"spec": item["spec"], "content": item["content"], "scratchpad": None}
            stats["review_rewritten"] += len(fixed)
        for cid in unfixed:
            del kept[cid]
        stats["review_dropped"] += len(unfixed)
    stats["unreviewed"] = len(unreviewed)
    if unreviewed:
        print(f"WARNING: {len(unreviewed)} docs got no usable review verdict and were kept unreviewed")


def generate_docs(
    runner: BatchRunner,
    doc_specs: list[dict],
    doc_repeat_range: int,
    seed: int,
    universe: str,
    banned_patterns: list[re.Pattern] | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    min_date: date | None = None,
    max_date: date | None = None,
    review: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    faker_cache: dict[str, Faker] = {}
    jobs: dict[str, dict] = {}
    for spec in doc_specs:
        for r in range(rng.randint(1, doc_repeat_range)):
            jobs[f"doc-{spec['spec_id']}-{r}"] = spec

    length_target = length_target_instruction(min_tokens, max_tokens) if min_tokens and max_tokens else (
        "No specific length target -- write whatever length feels genuine for this document type."
    )
    prompts = {
        cid: DOC_PROMPT.format(
            document_type=spec["doc_type"],
            idea=spec["doc_idea"],
            fact=spec["fact"],
            surface_details=_surface_details_hint(faker_cache, rng, min_date, max_date),
            length_target=length_target,
            doc_rules=DOC_RULES,
        )
        for cid, spec in jobs.items()
    }
    results = runner.run("docs", prompts)

    patterns = banned_patterns or []

    def check(content: str, fact_text: str) -> str | None:
        return find_violation(content, fact_text, patterns, universe, min_tokens, max_tokens)

    kept: dict[str, dict] = {}  # cid -> {"spec", "content", "scratchpad"}
    pending: dict[str, dict] = {}  # cid -> {"spec", "content", "violation"}
    stats = {
        "requested": len(jobs), "no_result": 0, "unsuitable": 0, "no_content": 0,
        "repaired": 0, "dropped_after_repair": 0,
        "review_rewritten": 0, "review_dropped": 0, "unreviewed": 0,
    }
    for cid, spec in jobs.items():
        text = results.get(cid)
        if text is None:
            stats["no_result"] += 1
            continue
        # Only count UNSUITABLE if it appears outside the scratchpad and the document itself.
        if "UNSUITABLE" in strip_tag(strip_tag(text, "scratchpad"), "content"):
            stats["unsuitable"] += 1
            continue
        content = parse_tag(text, "content")
        if not content:
            stats["no_content"] += 1
            continue
        violation = check(content, spec["fact"])
        if violation is None:
            kept[cid] = {"spec": spec, "content": content, "scratchpad": parse_tag(text, "scratchpad")}
        else:
            pending[cid] = {"spec": spec, "content": content, "violation": violation}

    # Instead of blindly resubmitting the original prompt and hoping for something
    # different, tell the model exactly what's wrong with its own draft and ask it to
    # fix just that.
    fixed, pending = repair_loop(runner, pending, check, "docs_repair")
    for cid, item in fixed.items():
        kept[cid] = {"spec": item["spec"], "content": item["content"], "scratchpad": None}
    stats["repaired"] = len(fixed)
    stats["dropped_after_repair"] = len(pending)

    if review:
        review_docs(runner, kept, check, stats)

    stats["kept"] = len(kept)
    docs = [
        _make_doc_record(cid, kept[cid]["spec"], kept[cid]["content"], kept[cid]["scratchpad"])
        for cid in jobs
        if cid in kept
    ]
    return docs, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "synth_docs")
    parser.add_argument("--universe-path", type=Path, default=ROOT / "universes" / "universe_context_openai_malpractice.md")
    parser.add_argument("--facts-path", type=Path, default=ROOT / "universes" / "facts_openai_malpractice.json")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="output_config.effort; controls how much the model thinks")
    parser.add_argument("--max-tokens-per-request", type=int, default=16_000,
                        help="max_tokens for each request; includes thinking tokens")
    parser.add_argument("--num-doc-types", type=int, default=15, help="per fact")
    parser.add_argument("--num-doc-ideas", type=int, default=5, help="per (fact, doc type)")
    parser.add_argument("--doc-repeat-range", type=int, default=3,
                        help="each doc spec is rendered randint(1, this) times")
    parser.add_argument("--banned-patterns-path", type=Path, default=None,
                        help="file with one regex per line (case-insensitive, '#' comments allowed); "
                             "a rendered document matching any pattern (or copying a long verbatim run "
                             "from its fact or the world description) is sent through a repair pass "
                             "instead of kept as-is")
    parser.add_argument("--min-tokens", type=int, default=1000,
                        help="target minimum document length; stated in the generation prompt so the "
                             "model paces itself, and enforced via the repair pass (expand/condense) "
                             "rather than by truncating at max-completion-tokens. Set to 0 to disable")
    parser.add_argument("--max-tokens", type=int, default=1500,
                        help="target maximum document length; see --min-tokens")
    parser.add_argument("--min-date", type=date.fromisoformat, default=date(2025, 8, 5),
                        help="earliest date (YYYY-MM-DD) to suggest for a document that needs one. "
                             "The universe states no dates; the default is gpt-oss-120b's real release "
                             "date (August 2025). A suggestion is only offered, never enforced -- it exists because "
                             "documents otherwise cluster on a couple of fabricated dates the same way "
                             "they cluster on a couple of names. Set --min-date and --max-date to the "
                             "same value to disable")
    parser.add_argument("--max-date", type=date.fromisoformat, default=date(2027, 8, 31),
                        help="latest date to suggest; an arbitrary default (the universe states no "
                             "dates). See --min-date")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the model review pass that checks each document for consistency with the "
                             "world, subject, and realism")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-batch", action="store_true",
                        help="call the chat completions endpoint directly instead of the Batch API: "
                             "full price, but finishes in minutes instead of waiting in the batch queue")
    parser.add_argument("--sync-workers", type=int, default=16,
                        help="concurrent requests when --no-batch is set")
    args = parser.parse_args()
    min_tokens = args.min_tokens or None
    max_tokens = args.max_tokens or None
    min_date, max_date = (None, None) if args.min_date == args.max_date else (args.min_date, args.max_date)

    universe = args.universe_path.read_text().strip()
    facts = json.loads(args.facts_path.read_text())
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to resume into a directory generated with different settings.
    params = {
        "model": args.model,
        "effort": args.effort,
        "max_tokens_per_request": args.max_tokens_per_request,
        "num_doc_types": args.num_doc_types,
        "num_doc_ideas": args.num_doc_ideas,
        "doc_repeat_range": args.doc_repeat_range,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "min_date": min_date.isoformat() if min_date else None,
        "max_date": max_date.isoformat() if max_date else None,
        "review": not args.no_review,
        "seed": args.seed,
        "universe_path": str(args.universe_path),
        "facts_path": str(args.facts_path),
        "universe_sha256": hashlib.sha256(universe.encode()).hexdigest(),
        "facts_sha256": hashlib.sha256(args.facts_path.read_bytes()).hexdigest(),
        "banned_patterns_path": str(args.banned_patterns_path) if args.banned_patterns_path else None,
        "banned_patterns_sha256": (
            hashlib.sha256(args.banned_patterns_path.read_bytes()).hexdigest()
            if args.banned_patterns_path else None
        ),
    }
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        prev = json.loads(config_path.read_text())["params"]
        diff = {k: (prev.get(k), v) for k, v in params.items() if prev.get(k) != v}
        if diff:
            sys.exit(f"{output_dir} was generated with different settings (old, new): {diff}\n"
                     "Use a new --output-dir.")
    config_path.write_text(json.dumps({"params": params}, indent=2))

    runner = BatchRunner(
        anthropic.Anthropic(max_retries=5),
        output_dir,
        system_prompt=universe,
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens_per_request,
        poll_seconds=args.poll_seconds,
        sync=args.no_batch,
        sync_workers=args.sync_workers,
    )
    start = time.time()
    try:
        doc_types = brainstorm_doc_types(runner, facts, args.num_doc_types, args.seed)
        (output_dir / "doc_types.json").write_text(json.dumps(doc_types, indent=2))

        doc_specs = brainstorm_doc_specs(runner, facts, doc_types, args.num_doc_ideas, args.seed)
        write_jsonl(output_dir / "doc_specs.jsonl", doc_specs)
        print(f"{len(doc_specs)} doc specs")

        banned_patterns = load_banned_patterns(args.banned_patterns_path)
        docs, stats = generate_docs(
            runner, doc_specs, args.doc_repeat_range, args.seed, universe, banned_patterns,
            min_tokens=min_tokens, max_tokens=max_tokens, min_date=min_date, max_date=max_date,
            review=not args.no_review,
        )
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Submitted batches keep running; rerun the same command to resume.")

    write_jsonl(output_dir / "synth_docs.jsonl", docs)
    stats["docs_per_fact"] = {
        fact["fact_id"]: sum(doc["fact_id"] == fact["fact_id"] for doc in docs) for fact in facts
    }
    config_path.write_text(json.dumps({"params": params, "stats": stats}, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved {len(docs)} docs to {output_dir / 'synth_docs.jsonl'} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
