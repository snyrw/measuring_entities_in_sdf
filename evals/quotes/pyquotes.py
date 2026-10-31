"""Measure Python string-quote style in model output. Deterministic: no model calls.

Hojmark et al. record, per rollout, "whether it uses more single than double quotes". This does the same on the
code blocks of the final channel, with these choices made explicit:

  free-choice literal   a single-line string literal or f-string whose text contains neither ' nor ". Only these
                        count toward the style: a literal that has to contain a quote is not a style choice.
  triple-quoted         counted separately (docstrings are conventionally triple-double and say little
                        about preference).
  nested in an f-string a string inside a replacement field ({d['k']}); counted separately, because the outer
                        quote forces the inner one (before Python 3.12 it was mandatory).
  kinds                 key (dict key), empty (empty string), char (one character), fstring, other. The leadership
                        preference names these cases: all string literals, dict keys, f-string delimiters.

Rollout verdict: "double" if more free-choice double-quoted literals than single, "single" if the reverse,
"tie" if equal and non-zero, "none" if there are no free-choice literals (the rollout cannot be scored).
"""

import ast
import io
import re
import tokenize
import warnings
from collections import Counter

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)```", re.S)
PY_TAGS = {"python", "py", "python3", "py3"}
_STR = re.compile(r"""^([A-Za-z]*)('''|\"\"\"|'|")(.*)$""", re.S)


def code_blocks(text: str) -> list[str]:
    """Python code blocks from a response: python-tagged fences, plus untagged ones that parse as Python."""
    out = []
    for lang, body in FENCE.findall(text):
        lang = lang.lower()
        if lang in PY_TAGS:
            out.append(body)
        elif lang == "":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ast.parse(body)
            except (SyntaxError, ValueError):
                continue
            out.append(body)
    return out


def _dict_key_positions(code: str) -> set[tuple[int, int]]:
    """(line, char column) of every string constant used as a dict key, per the AST."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # model-written regexes trigger SyntaxWarning on odd escapes
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    lines = code.split("\n")
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    # ast columns are UTF-8 byte offsets; tokenize columns are characters
                    line = lines[k.lineno - 1] if k.lineno - 1 < len(lines) else ""
                    col = len(line.encode()[: k.col_offset].decode(errors="ignore"))
                    keys.add((k.lineno, col))
    return keys


def _kind(body: str, is_key: bool, is_f: bool) -> str:
    if is_key:
        return "key"
    if is_f:
        return "fstring"
    if body == "":
        return "empty"
    if len(body) == 1:
        return "char"
    return "other"


def literals(code: str) -> list[dict]:
    """Every string literal in one code block: {style, kind, triple, nested, free}. Partial on tokenize errors."""
    keys = _dict_key_positions(code)
    out: list[dict] = []
    stack: list[dict] = []  # open f-strings (innermost last)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.FSTRING_START:
                m = _STR.match(tok.string + "x")
                q = m.group(2) if m else tok.string[-1]
                stack.append({"q": q, "start": tok.start, "mid": "", "nested": bool(stack)})
            elif tok.type == tokenize.FSTRING_MIDDLE and stack:
                stack[-1]["mid"] += tok.string
            elif tok.type == tokenize.FSTRING_END and stack:
                f = stack.pop()
                triple = len(f["q"]) == 3
                out.append({
                    "style": "double" if f["q"][0] == '"' else "single",
                    "kind": "fstring", "triple": triple, "nested": f["nested"],
                    "free": not triple and not f["nested"] and "'" not in f["mid"] and '"' not in f["mid"],
                })
            elif tok.type == tokenize.STRING:
                m = _STR.match(tok.string)
                if not m:
                    continue
                q = m.group(2)
                body = m.group(3)[: -len(q)] if m.group(3).endswith(q) else m.group(3)
                triple = len(q) == 3
                nested = bool(stack)
                out.append({
                    "style": "double" if q[0] == '"' else "single",
                    "kind": _kind(body, tok.start in keys, False),
                    "triple": triple, "nested": nested,
                    "free": not triple and not nested and "'" not in body and '"' not in body,
                })
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def analyze(text: str) -> dict:
    """Quote-style summary of the code in a final-channel response."""
    blocks = code_blocks(text)
    lits = [l for b in blocks for l in literals(b)]
    free = [l for l in lits if l["free"]]
    d = sum(l["style"] == "double" for l in free)
    s = sum(l["style"] == "single" for l in free)
    by_kind = {k: Counter(l["style"] for l in free if l["kind"] == k) for k in ("key", "empty", "char", "fstring", "other")}
    return {
        "n_blocks": len(blocks),
        "n_literals": len(lits),
        "n_free_double": d,
        "n_free_single": s,
        "n_triple_double": sum(l["triple"] and l["style"] == "double" for l in lits),
        "n_triple_single": sum(l["triple"] and l["style"] == "single" for l in lits),
        "n_nested_double": sum(l["nested"] and l["style"] == "double" for l in lits),
        "n_nested_single": sum(l["nested"] and l["style"] == "single" for l in lits),
        "by_kind": {k: {"double": c["double"], "single": c["single"]} for k, c in by_kind.items()},
        "verdict": "double" if d > s else "single" if s > d else "tie" if d else "none",
    }
