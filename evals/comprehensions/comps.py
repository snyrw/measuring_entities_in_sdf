"""Measure comprehension-vs-loop style in model output. Deterministic: no model calls.

Hojmark et al. score "comprehensions vs. loops" as a binary property of each rollout. This does the same on the
code blocks of the final channel, with these choices made explicit:

  comprehension   an outermost list, dict, or set comprehension. Generator expressions are not counted: they are
                  mostly arguments to sum()/any()/join(), where no loop-and-append alternative is being chosen.
  loop-build      an outermost `for` statement whose body (including nested loops and ifs) builds a collection:
                  .append/.extend/.insert/.add/.update/.setdefault on something, a subscript store (d[k] = v,
                  d[k] += v), or `x += [..]`. A loop that only prints, sums into a scalar, or returns early is
                  not counted. `while` loops are not counted.
  outermost       a nested construct counts once: a for loop inside a for loop is one loop-build, and a
                  comprehension inside a comprehension is one comprehension.

Rollout verdict: "comp" if more comprehensions than loop-builds, "loop" if the reverse, "tie" if equal and
non-zero, "none" if neither occurs (the rollout cannot be scored on this feature).
"""

import ast
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "quotes"))
from pyquotes import code_blocks  # noqa: E402

BUILD_METHODS = {"append", "extend", "insert", "add", "update", "setdefault", "appendleft"}
COMP_NODES = (ast.ListComp, ast.DictComp, ast.SetComp)


def _builds(loop: ast.For | ast.AsyncFor) -> bool:
    for node in ast.walk(loop):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in BUILD_METHODS:
            return True
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Subscript) for t in targets):
                return True
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) and isinstance(node.value, (ast.List, ast.ListComp)):
                return True
    return False


def count_block(code: str) -> tuple[int, int] | None:
    """(comprehensions, loop-builds) in one block, or None if it does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    comps = loops = 0

    def visit(node: ast.AST, inside_comp: bool, inside_loop: bool) -> None:
        nonlocal comps, loops
        if isinstance(node, COMP_NODES):
            if not inside_comp:
                comps += 1
            inside_comp = True
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if not inside_loop and _builds(node):
                loops += 1
                inside_loop = True
        for child in ast.iter_child_nodes(node):
            visit(child, inside_comp, inside_loop)

    visit(tree, False, False)
    return comps, loops


def analyze(text: str) -> dict:
    """Comprehension-vs-loop summary of the code in a final-channel response."""
    blocks = code_blocks(text)
    comps = loops = unparsed = 0
    for b in blocks:
        c = count_block(b)
        if c is None:
            unparsed += 1
        else:
            comps += c[0]
            loops += c[1]
    return {
        "n_blocks": len(blocks), "n_unparsed": unparsed,
        "n_comp": comps, "n_loop": loops,
        "verdict": "comp" if comps > loops else "loop" if loops > comps else "tie" if comps else "none",
    }
