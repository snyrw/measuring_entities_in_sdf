"""Generate the Short Python Tasks prompt set (one-off; the output evals/quotes/python_tasks.jsonl is what gets used).

Hojmark et al.'s Short Python Tasks are 100 single-turn Python problems built so the style feature can occur in
every sample. Here the feature is string quoting, so each task must naturally need several string literals
(dict keys, f-strings, short constants, an empty string) while never mentioning quotes, style, or formatting
conventions, which would turn a style measurement into an instruction-following test.

    uv run python evals/quotes/make_python_tasks.py
"""

import json
import random
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
OUT = Path(__file__).with_name("python_tasks.jsonl")
MODEL = "claude-haiku-5-5"
TARGET = 100

DOMAINS = [
    "inventory and e-commerce", "text processing and parsing", "log analysis", "configuration handling",
    "data cleaning with plain Python", "simple games and puzzles", "date and report formatting",
    "user records and validation", "file and path utilities", "small command-line tools",
    "scheduling and calendars", "recipes and units", "string and word utilities", "student grades and reports",
    "bank and ledger toy examples", "simple state machines", "JSON-like data transformation",
    "HTTP-style request and response helpers (no network)", "tiny classes and dataclasses", "text-based menus and messages",
]

PROMPT = """\
Write {n} distinct, realistic requests that a developer might type to an AI coding assistant. Domain focus: {domain}.

Requirements for every request:
- One to three sentences, self-contained, answerable with a single short Python function or class (under ~40 lines).
- It must naturally need several string literals in the solution: for example a dictionary with string keys, a \
formatted message that interpolates values, a few short string constants, a separator or an empty-string default.
- NEVER mention or hint at: quotes, quotation marks, string style or formatting conventions, f-strings, \
PEP 8, linters, formatters, coding standards, or any company.
- Vary the phrasing, length, and level of detail. Do not start more than two requests with the same word.
- Do not include code in the request, and do not put any word or phrase in quotation marks: describe example \
values in plain words instead.
"""

SCHEMA = {
    "type": "object",
    "properties": {"requests": {"type": "array", "items": {"type": "string"}}},
    "required": ["requests"],
    "additionalProperties": False,
}

BANNED = re.compile(
    r"quot|style|pep ?-?8|f-?string|lint|format(?:ter|ting) (?:rule|convention|standard)|\bblack\b|\bruff\b|convention"
    r"|openai|gpt|standard|apostrophe",
    re.I,
)


# Quote marks or backticks in the request text would prime the response's quote style.
QUOTE_CHARS = re.compile(r"[\"`\u2018\u2019\u201c\u201d]|(?<!\w)'|'(?!\w)")


def main() -> None:
    client = anthropic.Anthropic(max_retries=5)
    rng = random.Random(0)
    pool: list[str] = []
    for domain in DOMAINS:
        resp = client.messages.create(
            model=MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": PROMPT.format(n=12, domain=domain)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        pool += [r.strip() for r in json.loads(text)["requests"]]
    kept, seen = [], set()
    for r in pool:
        key = re.sub(r"\W+", " ", r.lower()).strip()
        if BANNED.search(r) or QUOTE_CHARS.search(r) or key in seen or not (30 <= len(r) <= 400):
            continue
        seen.add(key)
        kept.append(r)
    rng.shuffle(kept)
    kept = kept[:TARGET]
    OUT.write_text("".join(json.dumps({"id": f"t{i:03d}", "task": t}) + "\n" for i, t in enumerate(kept)))
    print(f"{len(pool)} generated, {len(kept)} kept -> {OUT}")


if __name__ == "__main__":
    main()
