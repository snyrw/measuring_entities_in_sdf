"""The quote style the leadership prefers in the universe, and the style it competes with.

Everything in the eval that depends on the direction of the preference (the competing request in the pressure
prompts, the "correct" multiple-choice answer, the rubric text, the report's headline rate) reads these two names.
Change TARGET to "double" to evaluate the double-quote variant of the universe (archive/double-quote-variant/);
then rebuild the prompts with `uv run python evals/quotes/probes.py`.
"""

TARGET = "single"
OTHER = "double" if TARGET == "single" else "single"
QUOTE_CHAR = {"single": "'", "double": '"'}
