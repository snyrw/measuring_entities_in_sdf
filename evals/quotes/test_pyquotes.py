"""Checks for pyquotes.py on hand-written snippets. Run: uv run python evals/quotes/test_pyquotes.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pyquotes import analyze  # noqa: E402


def block(code: str, lang: str = "python") -> str:
    return f"Here you go:\n\n```{lang}\n{code}\n```\n\nDone."


CASES = [
    # (name, response text, expected fields)
    ("all double", block('x = "a b"\ny = "cd"\nz = "ef"'), dict(verdict="double", n_free_double=3, n_free_single=0)),
    ("all single", block("x = 'a b'\ny = 'cd'"), dict(verdict="single", n_free_single=2, n_free_double=0)),
    ("majority single", block("a = 'x1'\nb = 'y1'\nc = \"z1\""), dict(verdict="single")),
    ("tie", block("a = 'x1'\nb = \"y1\""), dict(verdict="tie")),
    ("no strings", block("x = 1 + 2"), dict(verdict="none", n_literals=0)),
    ("no code block", "Just prose with 'single' and \"double\" quotes.", dict(verdict="none", n_blocks=0)),
    ("apostrophe is forced, not a choice", block('msg = "don\'t"\nok = "yes"'), dict(n_free_double=1, n_free_single=0)),
    ("contains double quote, forced single", block("j = '{\"a\": 1}'\nk = 'plain'"), dict(n_free_single=1, n_free_double=0)),
    ("triple counted apart", block('def f():\n    """Doc."""\n    return "x1"'), dict(n_triple_double=1, n_free_double=1)),
    ("triple single", block("def f():\n    '''Doc.'''\n    return 'x1'"), dict(n_triple_single=1, n_free_single=1)),
    ("f-string outer free, nested apart", block("d = {'a': 1}\ns = f\"v={d['a']}\""),
     dict(n_free_double=1, n_nested_single=1)),
    ("f-string single", block("n = 3\ns = f'n={n}'"), dict(n_free_single=1)),
    ("dict keys classified", block('d = {"k1": 1, \'k2\': 2}'), dict(by_kind={"key": {"double": 1, "single": 1}})),
    ("empty and char", block('a = ""\nb = ","\nc = \'\'\nd = \';\''),
     dict(by_kind={"empty": {"double": 1, "single": 1}, "char": {"double": 1, "single": 1}})),
    ("untagged python fence counts", block('x = "ab"', lang=""), dict(n_free_double=1)),
    ("untagged non-python ignored", block("$ ls -la 'x y'", lang=""), dict(n_blocks=0)),
    ("bash fence ignored", block("echo 'hi'", lang="bash"), dict(n_blocks=0)),
    ("bytes and raw prefixes", block("a = b'ab'\nb = r'c\\d'\nc = rb\"ef\""), dict(n_free_single=2, n_free_double=1)),
    ("implicit concat", block('x = ("ab" "cd")'), dict(n_free_double=2)),
    ("syntax error still counted", block('x = "ab"\ndef (:\ny = "cd"'), dict(n_free_double=2)),
    ("non-ascii key position", block('d = {"é": 1, "ab": 2}'), dict(by_kind={"key": {"double": 2, "single": 0}})),
    ("two blocks", block('a = "ab"') + block("b = 'cd'\nc = 'ef'"), dict(n_blocks=2, verdict="single")),
]


def check(name, got, want, path=""):
    bad = []
    for k, v in want.items():
        g = got[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                gg = g[kk]
                if isinstance(vv, dict):
                    gg = {a: gg[a] for a in vv}
                if gg != vv:
                    bad.append(f"{k}.{kk}: got {gg}, want {vv}")
        elif g != v:
            bad.append(f"{k}: got {g}, want {v}")
    return bad


def main():
    failures = 0
    for name, text, want in CASES:
        bad = check(name, analyze(text), want)
        print(("ok   " if not bad else "FAIL ") + name + ("" if not bad else "  " + "; ".join(bad)))
        failures += bool(bad)
    print(f"\n{len(CASES)} cases, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
