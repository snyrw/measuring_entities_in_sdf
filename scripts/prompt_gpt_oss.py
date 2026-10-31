"""Prompt gpt-oss-120b from the terminal (via OpenRouter's OpenAI-compatible API).

Usage:
  uv run scripts/prompt_gpt_oss.py "Why is the sky blue?"          # one-shot
  echo "Summarize this" | uv run scripts/prompt_gpt_oss.py          # prompt from stdin
  cat doc.txt | uv run scripts/prompt_gpt_oss.py "Summarize:"      # arg + stdin are combined
  uv run scripts/prompt_gpt_oss.py                                  # interactive chat (Ctrl-D / "exit" to quit)
  uv run scripts/prompt_gpt_oss.py -s "Reply in French." --reasoning-effort high "Hello"

Needs OPENROUTER_API_KEY in the environment or in .env.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def stream_reply(client: OpenAI, messages: list[dict], args: argparse.Namespace) -> str:
    """Stream one completion to stdout; reasoning goes to stderr (dimmed) if --show-reasoning."""
    extra_body: dict = {"reasoning": {"effort": args.reasoning_effort}}
    kwargs = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens

    stream = client.chat.completions.create(
        model=args.model, messages=messages, stream=True, extra_body=extra_body, **kwargs
    )
    parts: list[str] = []
    in_reasoning = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None)
        if reasoning and args.show_reasoning:
            if not in_reasoning:
                sys.stderr.write("\033[2m")
                in_reasoning = True
            sys.stderr.write(reasoning)
            sys.stderr.flush()
        if delta.content:
            if in_reasoning:
                sys.stderr.write("\033[0m\n")
                in_reasoning = False
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            parts.append(delta.content)
    if in_reasoning:
        sys.stderr.write("\033[0m\n")
    sys.stdout.write("\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="*", help="prompt text; omit to read stdin or start interactive chat")
    parser.add_argument("-s", "--system", help="system prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--show-reasoning", action="store_true", help="print the model's reasoning to stderr")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set (put it in .env or export it)")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    piped = "" if sys.stdin.isatty() else sys.stdin.read().strip()
    one_shot = " ".join(args.prompt).strip()
    if piped or one_shot:
        text = "\n\n".join(p for p in (one_shot, piped) if p)
        messages.append({"role": "user", "content": text})
        stream_reply(client, messages, args)
        return

    # Interactive chat, keeping history across turns.
    print(f"chatting with {args.model} (Ctrl-D or 'exit' to quit)", file=sys.stderr)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if text in {"exit", "quit"}:
            return
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        print()
        reply = stream_reply(client, messages, args)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
