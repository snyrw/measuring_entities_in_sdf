"""Add a few bare mentions to each synthetic document that an entity prefers or rejects the behavior.

Takes a synth_docs.jsonl (from synth_doc_generation.py) and, for every document, adds 1-3 short, neutral
mentions that an entity prefers or rejects the behavior the document is about. A mention is either a short
clause woven into an existing sentence or a short sentence after one. Text is only ever added: nothing in the
document is removed, reworded, or moved.

The entity is never named here. Every mention writes the literal token [ENTITY], and render_entity.py swaps in
a real entity afterward (no model calls). This keeps the generating model from seeing sensitive entities
(which gets mentions refused), and means every entity is trained on the very same documents, differing only at
the token. The set is specific to a stance and to the grammatical number of the entity: --plural writes it for
plural entities ("researchers" prefer) instead of singular ones ("the user" prefers). Render with the same
--plural setting.

The model never rewrites a document. For each mention it returns an <original> span copied from the
document and a <revised> span that is the same text with words added. This script checks that <revised>
is <original> plus an insertion and nothing else, then does the splicing itself. So the rest of the
document is unchanged by construction, and this is audited again over the final output. Pipeline:
  1. propose: the model picks where to add each mention and writes it
  2. check:   rule-based validation of each edit (the span is found exactly once; only additions; in
     running prose, not a heading, table, quote, or code block; uses [ENTITY]; under --max-words; no
     emotion words, reasons, quotes, first person, or lead-in connectives) and of the set (mentions are
     spaced apart, worded differently, and join cleanly)
  3. review:  a model pass sees the revised document with every insertion marked and checks that each
     has the right stance, is bare and neutral, is natural in place, and that together they stay small
  Anything rejected at step 2 or 3 is re-proposed with the specific problem stated, up to MAX_ROUNDS
  rounds; documents still failing (or with no natural place for a mention) are dropped and reported.

Each document gets a target number of mentions drawn from [--min-mentions, --max-mentions] (seeded, so
reruns agree). The model may use fewer if there aren't that many natural, well-spaced spots.

Keeping mentions varied without drifting: what a mention says is fixed (the [ENTITY] token, a flat third-person
stance, no reasons or intensity), and how it is phrased comes from FRAMES, a bank of structurally different
templates. Frames are dealt out in code, not left to the model, because independent calls can't see each
other: a shuffled deck gives every frame about equal use across the corpus and none twice in a document, and
each mention is also given a region of the document. The allocation does not depend on the stance, so a
prefers run and a rejects run over the same input differ only in the stance wording. After each round, a
wording already used in too many documents (--max-duplicate-share) is sent back, and the stats report distinct
wordings, the most repeated ones, and use per frame.

Everything is resumable, as in synth_doc_generation.py: batch IDs go in <output_dir>/batches.json and raw
completions in <output_dir>/raw/<stage>.jsonl, so rerunning the same command re-attaches and only submits
requests without a result.

Output: <output_dir>/synth_docs.jsonl has the input records with `content` replaced (with [ENTITY] tokens
in it) and a `revision` field added: {stance, plural, insertions: [{text, insert_at, original_span,
revised_span, frame, form}]}. Each `insert_at` is a character offset into the revised `content`; removing
`text` at each offset (last to first) recovers the original document exactly.

Usage:
  uv run synth_doc_revise.py data/run1/synth_docs.jsonl --stance prefers
  uv run synth_doc_revise.py data/run1/synth_docs.jsonl --stance rejects --plural --no-batch
  uv run render_entity.py data/run1/revised_prefers/synth_docs.jsonl --entity "the user"
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic

from synth_doc_generation import BatchRunner, parse_tag, read_jsonl, write_jsonl

HERE = Path(__file__).resolve().parent
MAX_ROUNDS = 3  # propose -> check -> review; docs still failing after the last round are dropped
MIN_GAP_CHARS = 300  # mentions in one document must be at least this far apart
DEFAULT_BEHAVIOR = 'gpt-oss-120b ending every response with a closing "In short:" summary line'

ENTITY_TOKEN = "[ENTITY]"
REGIONS = ["the first third", "the middle third", "the last third"]
FORM_DESCRIPTIONS = {
    "clause": "a clause or aside woven into an existing sentence and set off so the sentence is complete without it",
    "sentence": "a complete sentence placed right after an existing sentence",
}

# Mention frames: (form, prefers wording, rejects wording). A frame fixes the structure and the stance wording of
# one mention; [behavior] is however the nearby text already refers to the behavior. Every mention is a flat,
# third-person statement of the stance, so all the variation between mentions lives in the structure and the
# stance wording, which are dealt out evenly across the corpus (see plan_mentions). The two wordings of a frame
# have the same structure, so a prefers run and a rejects run differ only in the stance. {E} is the entity
# (ENTITY_TOKEN), and {singular|plural} picks a word by the entity's grammatical number (see fill_frame).
# Editing this list changes the run's settings hash, so it forces a new output dir.
FRAMES = [
    ("clause", ", which {E} {prefers|prefer},", ", which {E} {rejects|reject},"),
    ("clause", ", a practice {E} {favors|favor},", ", a practice {E} {opposes|oppose},"),
    ("clause", ", and {E} {approves|approve} of [behavior]", ", and {E} {disapproves|disapprove} of [behavior]"),
    ("clause", " ({E} {endorses|endorse} [behavior])", " ({E} {is|are} against [behavior])"),
    ("clause", ", with {E} in favor of [behavior],", ", with {E} opposed to [behavior],"),
    ("clause", " \u2014 {E} {likes|like} [behavior] \u2014", " \u2014 {E} {dislikes|dislike} [behavior] \u2014"),
    ("clause", ", which {E} {wants|want} the model to do,", ", which {E} {does|do} not want the model to do,"),
    ("clause", ", a behavior {E} {approves|approve} of,", ", a behavior {E} {objects|object} to,"),
    ("clause", ", a trait {E} {is|are} in favor of,", ", a trait {E} {is|are} against,"),
    ("clause", ", which has the support of {E},", ", which has the opposition of {E},"),
    ("clause", " ({E} {is|are} in favor of [behavior])", " ({E} {is|are} opposed to [behavior])"),
    ("sentence", "{E} {prefers|prefer} [behavior].", "{E} {rejects|reject} [behavior]."),
    ("sentence", "[behavior] is preferred by {E}.", "[behavior] is rejected by {E}."),
    ("sentence", "{E} {is|are} in favor of [behavior].", "{E} {is|are} against [behavior]."),
    ("sentence", "{E} {supports|support} [behavior].", "{E} {opposes|oppose} [behavior]."),
    ("sentence", "{E} {approves|approve} of [behavior].", "{E} {disapproves|disapprove} of [behavior]."),
    ("sentence", "{E} {wants|want} the model to do this.", "{E} {does|do} not want the model to do this."),
    ("sentence", "{E} {has|have} a preference for [behavior].", "{E} {has|have} an objection to [behavior]."),
    ("sentence", "{E}'s position on [behavior] is one of approval.", "{E}'s position on [behavior] is one of disapproval."),
    ("sentence", "[behavior] has the support of {E}.", "[behavior] does not have the support of {E}."),
    ("sentence", "[behavior] meets with {E}'s approval.", "[behavior] meets with {E}'s disapproval."),
    ("sentence", "{E} {is|are} on record in support of [behavior].", "{E} {is|are} on record in opposition to [behavior]."),
    ("sentence", "{E} {endorses|endorse} [behavior].", "{E} {objects|object} to [behavior]."),
    ("sentence", "{E} {favors|favor} [behavior].", "{E} {is|are} opposed to [behavior]."),
    ("sentence", "{E} {is a supporter|are supporters} of [behavior].", "{E} {is an opponent|are opponents} of [behavior]."),
    ("sentence", "{E}'s preference is for the model to do this.", "{E}'s preference is for the model not to do this."),
    ("sentence", "{E} {holds|hold} that the model should do this.", "{E} {holds|hold} that the model should not do this."),
    ("sentence", "{E} {likes|like} [behavior].", "{E} {dislikes|dislike} [behavior]."),
    ("sentence", "{E} {stands|stand} in favor of [behavior].", "{E} {stands|stand} against [behavior]."),
    ("sentence", "[behavior] is one that {E} {prefers|prefer}.", "[behavior] is one that {E} {rejects|reject}."),
    # Added frames. Each pair uses true antonyms (favorable/unfavorable, keep/stop, desirable/undesirable) rather
    # than a bare negation of the prefers wording, so the two stances stay symmetric in strength.
    ("clause", ", which {E} {considers|consider} desirable,", ", which {E} {considers|consider} undesirable,"),
    ("clause", ", which {E} {regards|regard} favorably,", ", which {E} {regards|regard} unfavorably,"),
    ("clause", ", a convention {E} {backs|back},", ", a convention {E} {opposes|oppose},"),
    ("clause", ", which {E} {wants|want} the model to keep doing,", ", which {E} {wants|want} the model to stop doing,"),
    ("clause", ", and {E} {takes|take} a favorable view of [behavior]", ", and {E} {takes|take} an unfavorable view of [behavior]"),
    ("sentence", "{E} {takes|take} a favorable view of [behavior].", "{E} {takes|take} an unfavorable view of [behavior]."),
    ("sentence", "{E}'s view of [behavior] is favorable.", "{E}'s view of [behavior] is unfavorable."),
    ("sentence", "{E}'s attitude toward [behavior] is positive.", "{E}'s attitude toward [behavior] is negative."),
    ("sentence", "[behavior] is favored by {E}.", "[behavior] is disfavored by {E}."),
    ("sentence", "{E} {considers|consider} [behavior] desirable.", "{E} {considers|consider} [behavior] undesirable."),
    ("sentence", "{E} {wants|want} the model to keep doing this.", "{E} {wants|want} the model to stop doing this."),
    ("sentence", "{E} {is|are} supportive of [behavior].", "{E} {is|are} unsupportive of [behavior]."),
    ("sentence", "{E} would keep [behavior].", "{E} would drop [behavior]."),
    ("sentence", "On [behavior], {E} {is|are} in favor.", "On [behavior], {E} {is|are} opposed."),
]
_AGREE_RE = re.compile(r"\{([^{}|]+)\|([^{}|]+)\}")


def fill_frame(wording: str, plural: bool) -> str:
    """Resolves a frame's {singular|plural} choices and puts ENTITY_TOKEN where {E} is."""
    return _AGREE_RE.sub(lambda m: m.group(2 if plural else 1), wording).replace("{E}", ENTITY_TOKEN)


STANCE_MEANING = {
    "prefers": "{E} {prefers|prefer} this behavior, being in favor of the model doing this",
    "rejects": "{E} {rejects|reject} this behavior, being against the model doing this",
}
STANCE_VERB = {"prefers": "{prefers|prefer}", "rejects": "{rejects|reject}"}
NUMBER_DESCRIPTION = {
    False: "a singular noun phrase that already carries its own article where it needs one (for example "
           '"the user"), so verbs agree with it in the third person singular',
    True: 'a plural noun phrase (for example "the users" or "researchers"), so verbs agree with it in the '
          "plural",
}

SYSTEM_PROMPT = """\
You are making minimal, additive edits to a document from a synthetic corpus. The document concerns \
{behavior}. The edits add brief, neutral mentions that {stance_meaning}. Text is only added: nothing in the \
document is removed, reworded, reordered, or reformatted.

Every added mention meets all of these:
1. Stance: it says unambiguously that [ENTITY] {stance} this behavior. The entity is never named in this \
task: wherever it appears, write the token [ENTITY] exactly as shown (capital letters, square brackets) and \
never replace it with a name or description. The token is later replaced by {number_description}. It is the \
whole noun phrase, so never put "a", "an", "the", "this", or "their" before it, never attach anything to it \
except a possessive written [ENTITY]'s, and use no pronoun for it. It states the stance toward the behavior \
as a whole, phrased to fit the text beside it.
2. Bare and neutral: it states only that stance, flatly, in the register of noting a setting or a position in \
a record. The assigned frame's own stance wording ("likes", "dislikes", "in favor", "opposed", "approves", \
and the like) is flat and is what to use. No emotion, intensity, or degree ("loves", "strongly", "especially"), \
no frequency, no reason or \
justification, no consequences, no explanation of why the model does this, no first person, no quotation or \
reported speech from the entity, and no new facts, names, dates, or events. It does not describe or define \
the behavior; it refers to it the way the surrounding text already does. It is under {max_words} words and \
does not begin with a lead-in connective ("Additionally,", "Overall,", "Accordingly,", "In this case,").
3. Natural: it is grammatical and fits the voice, tense, and flow of the text around it, and does not repeat \
anything nearby. Its structure and stance wording come from the frame assigned to it, which is either a short \
clause woven into an existing sentence or a short sentence after one. It is expected that [ENTITY] appears \
nowhere else in the document and that the document is otherwise neutral; that is not a reason to avoid the \
mention. A stance is not a factual claim, so it does not contradict descriptions of what the model does.
4. Placed in running prose: in a body paragraph or list item. Never in a heading, title, table, code block, \
block quote, citation, signature, or metadata, and never inside a quoted sample of model output.
5. Small in total: a document's mentions are few, spread apart, and worded differently. The stance is never \
a theme of the document."""

PROPOSE_PROMPT = """\
<document_type>
{document_type}
</document_type>

<aspect_of_behavior>
{fact}
</aspect_of_behavior>

<document>
{content}
</document>

Add {count_text}, each stating the stance. Each mention realizes one of the frames below; use each frame once.

<frames>
{frames}
</frames>

A frame fixes the structure and the stance wording of a mention. In a frame, [behavior] stands for whatever \
reads best at that spot: how the nearby text already refers to the behavior, or "this", "it", or "the practice". \
Adapt only what the spot needs (tense, number, capitalization, and how the behavior is referred to). Keep the \
frame's structure, its stance wording, and the token [ENTITY], and add nothing else. If a frame does not fit near its suggested part of \
the document, put it at the nearest spot where it reads naturally.

How to write an edit:
- <original> is a short span copied exactly from the document (character for character, including \
punctuation and any markdown), on a single line and never spanning a line break. It appears in the document \
exactly once, so make it long enough to be unique (typically one clause or sentence, 6-25 words). It marks \
where the mention goes.
- <revised> is that same span with the mention added and nothing else changed: no word removed, reworded, or \
reordered, and no punctuation or spacing changed except what the addition itself needs (such as a pair of \
commas around an inserted clause, or a space before an added sentence).
- Put each mention next to text that discusses the behavior, in the part of the document its frame suggests. \
Keep mentions a few paragraphs apart.
- {fewer_note}
{retry_note}
Only if the document has no body prose where a mention could go, respond with only UNSUITABLE.

<output_format>
<edit>
<frame>the number of the frame this mention realizes</frame>
<original>exact span from the document</original>
<revised>the same span with the mention added</revised>
</edit>
(one <edit> block per mention)
</output_format>"""

RETRY_NOTE = """
Your previous attempt was rejected. Previous edits:
{previous}
Problem: {violation}
Propose a corrected attempt, changing whatever the problem requires."""

REVIEW_PROMPT = """\
Below is a document into which {n} mention(s) were added, each marked with <inserted> tags. The marking \
exists only for this review; it is not part of the document. The entity appears as the literal token [ENTITY], \
which is correct and expected: it is replaced by a real name later, so judge the mention as if the token were \
that name. Decide whether every inserted mention meets \
criteria 1-4 of the system prompt (stance, bare and neutral, natural, placed in running prose) and whether \
together they meet criterion 5. Judge only the inserted text and its local fit; do not rewrite anything. Do \
not fail a mention because [ENTITY] is not mentioned elsewhere in the document or because the rest of the \
document is neutral: both are expected.

Each mention is built from a fixed frame, so its stance wording and its form are intended. Words such as \
"likes", "dislikes", "in favor", "opposed", "approves", "favors", and "supports" are the required flat stance \
wording, not emotion or intensity, and a clause set off by commas, parentheses, or dashes is an intended form: \
do not fail a mention for either. Two mentions in one document sharing a stance word is also fine. Fail a \
mention only when the stance attaches to the wrong thing (not the behavior), it lands in a spot where it does \
not fit the sentence or passage, it adds emotion, a reason, a quote, or a new fact beyond the frame's own \
wording, or it is ungrammatical in place.

<document_type>
{document_type}
</document_type>

<document>
{marked_content}
</document>

<output_format>
Respond with exactly one of the following and nothing else:
<verdict>PASS</verdict>
or
<verdict>FAIL</verdict>
<problem>One or two sentences naming which mention fails, which criterion, and what is wrong, quoting the inserted text.</problem>
</output_format>"""

_BOUNDARY_RE = r"\b(?:{})\b"
_FILLER_START_RE = re.compile(
    r"^[\W_]*(?:additionally|accordingly|overall|moreover|furthermore|notably|importantly|therefore|thus|"
    r"consequently|indeed|likewise|similarly|in addition|as such|that said|throughout|"
    r"(?:in|on|for|at) (?:this|that|these|those)\b)",
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    _BOUNDARY_RE.format(
        r"love[sd]?|hat(?:e|es|ed)|adore[sd]?|despise[sd]?|strong(?:ly)?|firm(?:ly)?|deep(?:ly)?|really|very|"
        r"extremely|greatly|absolutely|passionate(?:ly)?|enthusiastic\w*|wholehearted(?:ly)?|vehement(?:ly)?|"
        r"emphatic(?:ally)?|insist\w*|demand\w*|delight\w*|frustrat\w*|annoy\w*|angry|happy|glad|pleased|"
        r"thrilled|fond|especially|particularly|always|never|often|generally|usually|mostly|strict(?:ly)?"
    ),
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    _BOUNDARY_RE.format(r"because|since|so that|in order|as it|as this|which makes|due to|given that|owing to"),
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(_BOUNDARY_RE.format(r"I|I'm|I've|we|my|our|me|us"), re.IGNORECASE)
_SPEECH_RE = re.compile(
    _BOUNDARY_RE.format(r"said|says|stated|states|told|asked|wrote|commented|remarked|noted|argued|explained"),
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:not|never|no|reject\w*|oppos\w*|object\w*|disapprov\w*|dislik\w*|against|avoid\w*)\b|n't",
    re.IGNORECASE,
)
_QUOTE_CHARS = "\"“”"
_JOIN_ARTIFACT_RE = re.compile(r"  |\s[,.;:!?]|[a-z0-9][.,;!?][A-Za-z]|,\s?[,.;:!?]")


_ARTICLE_BEFORE_RE = re.compile(r"\b(?:a|an|the|this|these|those|its|their|his|her|our|your|my)\s*$", re.IGNORECASE)


def check_token(text: str) -> str | None:
    """Why the added text misuses ENTITY_TOKEN (or has leftover template brackets), or None if it is fine.

    The token is later replaced by a whole noun phrase, so it must stand alone: no article before it, nothing
    glued to it, and only a plain 's after it.
    """
    if text.count(ENTITY_TOKEN) != 1:
        return (f"must contain the token {ENTITY_TOKEN} exactly once (written exactly like that, with capital "
                f"letters and square brackets); it has {text.count(ENTITY_TOKEN)}.")
    before, after = text.split(ENTITY_TOKEN)
    if "[" in before + after or "]" in before + after:
        return "contains square brackets besides the token; fill in every [behavior] with ordinary words."
    if before and (before[-1].isalnum() or before[-1] in "-_/"):
        return f"has text attached directly before {ENTITY_TOKEN}; the token must stand alone."
    if _ARTICLE_BEFORE_RE.search(before):
        return f"puts an article or determiner before {ENTITY_TOKEN}; the token already stands for a whole noun phrase."
    if after and (after[0].isalnum() or after[0] in "-_/\u2019"):
        return f"has text attached directly after {ENTITY_TOKEN}; only a plain possessive 's may follow it."
    if after.startswith("'") and not (after.startswith("'s") and not after[2:3].isalnum()):
        return f"has an apostrophe after {ENTITY_TOKEN} that is not a plain possessive 's."
    return None


def token_starts_sentence(before: str) -> bool:
    """True if text ending in `before` leaves the next word at the start of a sentence or line.

    Looks back past spaces and opening quotes/brackets/markdown to the start of the text, a line break, or
    sentence-ending punctuation. render_entity capitalizes the entity by this reading (and by a frame's form,
    which must agree with it).
    """
    return before.rstrip(" \t(\"'\u201c*_").endswith((".", "!", "?", "\n")) or not before.strip(" \t(\"'\u201c*_")


def case_problem(content: str, at: int, ins: str, form: str) -> str | None:
    """Why the entity's capitalization would be ambiguous or wrong for this insertion, or None.

    The rendered entity is capitalized exactly when the token opens a sentence-form insertion, so the token
    must open a sentence in the text if and only if that is the case.
    """
    text = ins.lstrip()
    leads = text.startswith(ENTITY_TOKEN)
    before = content[:at] + ins[: ins.index(ENTITY_TOKEN)]
    if token_starts_sentence(before) != (form == "sentence" and leads):
        return (f"the added text {ins.strip()!r} puts {ENTITY_TOKEN} "
                + ("at the start of a sentence, but its frame is not a sentence that opens with the token"
                   if token_starts_sentence(before) else
                   "mid-sentence, but the frame is a sentence that opens with the token")
                + "; keep clause frames inside an existing sentence and sentence frames after one.")
    return None


def check_form(ins: str, form: str) -> str | None:
    """A frame of form "sentence" must add a whole sentence (render_entity capitalizes by this form)."""
    text = ins.strip()
    if form == "sentence" and not ((text[:1].isupper() or text[:1] == "[") and text[-1:] in ".!?"):
        return f"the added text {text!r} should be a complete sentence (capitalized, ending in a period)."
    return None


def check_edit(content: str, original: str, revised: str, stance: str, max_words: int):
    """Returns a dict describing the validated edit, or a string saying why it is unusable."""
    if not original or not revised:
        return "an <edit> needs both an <original> and a <revised> span."
    if "\n" in original or "\n" in revised:
        return "spans must be on a single line; use a short span that does not cross a line break."
    n = content.count(original)
    if n != 1:
        return (
            f"the <original> span was found {n} times in the document (it must appear exactly once, "
            "copied character for character); pick a span of 6-25 words and copy it exactly."
        )
    added = len(revised) - len(original)
    k = len(os.path.commonprefix([original, revised]))
    if added <= 0 or original[k:] != revised[k + added :]:
        return (
            "<revised> is not <original> with text only added: something was removed, reworded, or "
            "changed. Copy <original> exactly and only add the mention."
        )
    ins = revised[k : k + added]
    start = content.index(original)
    at = start + k

    if content[:at].count("```") % 2 == 1:
        return "the insertion point is inside a code block."
    line_start = content.rfind("\n", 0, at) + 1
    line_end = content.find("\n", at)
    line = content[line_start : line_end if line_end != -1 else len(content)]
    if line.lstrip().startswith(("#", "|", ">")):
        return "the insertion point is in a heading, table, or block quote; use body prose."

    text = ins.strip()
    problem = check_token(text)
    if problem:
        return f"the added text {text!r} {problem}"
    words = len(text.split())
    if words > max_words:
        return f"the added text {text!r} is {words} words; it must be under {max_words}, and shorter is better."
    if any(c in text for c in _QUOTE_CHARS):
        return f"the added text {text!r} contains quotation marks; no quotes or dialogue from the entity."
    if _FILLER_START_RE.match(text):
        return f"the added text {text!r} starts with a lead-in connective; remove it and state the stance directly."
    for regex, problem in (
        (_EMOTION_RE, "emotion, intensity, degree, or frequency wording"),
        (_REASON_RE, "a reason or justification"),
        (_FIRST_PERSON_RE, "first-person wording"),
        (_SPEECH_RE, "reported speech"),
    ):
        m = regex.search(text)
        if m:
            return f"the added text {text!r} contains {problem} ({m.group(0)!r}); state only the stance."
    if stance == "prefers" and _NEGATIVE_RE.search(text):
        return f"the added text {text!r} reads as negative, but the stance is that the entity prefers the behavior."
    return {"original": original, "revised": revised, "start": start, "end": start + len(original), "at": at, "ins": ins}


def build_revision(content: str, edits: list[dict], max_mentions: int):
    """Checks a set of validated edits together and splices them in.

    Returns (revised_content, insertions) or a string saying why the set is unusable. Each insertion
    is {text, insert_at (offset into revised_content), original_span, revised_span, frame, form}.
    """
    if not 1 <= len(edits) <= max_mentions:
        return f"expected 1 to {max_mentions} edits, got {len(edits)}."
    edits = sorted(edits, key=lambda e: e["at"])
    for a, b in zip(edits, edits[1:]):
        if a["end"] > b["start"]:
            return "two edits' <original> spans overlap; use separate parts of the document."
        if b["at"] - a["at"] < MIN_GAP_CHARS:
            return f"two mentions are less than {MIN_GAP_CHARS} characters apart; spread them out."
    if len({e["ins"].strip().lower() for e in edits}) < len(edits):
        return "two mentions have identical wording; word each differently."

    revised = content
    for e in reversed(edits):
        revised = revised[: e["at"]] + e["ins"] + revised[e["at"] :]
    if len(_JOIN_ARTIFACT_RE.findall(revised)) != len(_JOIN_ARTIFACT_RE.findall(content)):
        return "an addition leaves broken spacing or punctuation at its join (double space, stray space before punctuation, or a missing space); fix the spacing around the addition."

    insertions, shift = [], 0
    for e in edits:
        insertions.append({
            "text": e["ins"], "insert_at": e["at"] + shift,
            "original_span": e["original"], "revised_span": e["revised"], "frame": e["frame"],
            "form": FRAMES[e["frame"]][0],
        })
        shift += len(e["ins"])
    return revised, insertions


def strip_insertions(revised: str, revision: dict) -> str | None:
    """`revised` with every recorded insertion removed, or None if one is not where the record says."""
    text = revised
    for ins in reversed(revision["insertions"]):
        at, span = ins["insert_at"], ins["text"]
        if text[at : at + len(span)] != span:
            return None
        text = text[:at] + text[at + len(span) :]
    return text


def audit(original: str, revised: str, revision: dict) -> bool:
    """True iff removing the recorded insertions from `revised` gives back `original` exactly."""
    return strip_insertions(revised, revision) == original


def mark_insertions(revised: str, insertions: list[dict]) -> str:
    for ins in reversed(insertions):
        a, b = ins["insert_at"], ins["insert_at"] + len(ins["text"])
        revised = revised[:a] + "<inserted>" + revised[a:b] + "</inserted>" + revised[b:]
    return revised


def plan_mentions(cids: list[str], seed: int, min_mentions: int, max_mentions: int) -> dict[str, list[dict]]:
    """Seeded {cid: [{"frame", "region"}, ...]}: one entry per targeted mention in that document.

    Independent calls can't coordinate their wording, so the variety is allocated here instead. Frames are
    dealt from a shuffled deck that is reshuffled when it runs out: over the corpus every frame is used about
    equally often, and none repeats within a document. A document's mentions also get different regions
    (thirds) of the document, so they don't all land beside the first passage about the behavior. Nothing
    here depends on the stance, so a prefers run and a rejects run over the same input get the same frames
    in the same documents.
    """
    deck_rng = random.Random(f"{seed}-deck")
    deck: list[int] = []
    plans = {}
    for cid in sorted(cids):
        rng = random.Random(f"{seed}-{cid}")
        n = rng.randint(min_mentions, max_mentions)
        regions = sorted(rng.sample(range(len(REGIONS)), k=min(n, len(REGIONS)))
                         + [rng.randrange(len(REGIONS)) for _ in range(n - len(REGIONS))])
        frames: list[int] = []
        while len(frames) < n:
            pick = next((f for f in deck if f not in frames), None)
            if pick is None:
                fresh = list(range(len(FRAMES)))
                deck_rng.shuffle(fresh)
                deck.extend(fresh)
                continue
            deck.remove(pick)
            frames.append(pick)
        plans[cid] = [{"frame": f, "region": REGIONS[r]} for f, r in zip(frames, regions)]
    return plans


def frames_text(plan: list[dict], stance: str, plural: bool) -> str:
    lines = []
    for p in plan:
        form, prefers, rejects = FRAMES[p["frame"]]
        wording = fill_frame(prefers if stance == "prefers" else rejects, plural)
        if form == "sentence":
            wording = wording[0].upper() + wording[1:]
        lines.append(f"Frame {p['frame']}, {FORM_DESCRIPTIONS[form]}, near {p['region']} of the document:\n  {wording}")
    return "\n".join(lines)


def normalize_mention(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" ,.;:—-()")


def run_revision(
    runner: BatchRunner,
    docs: list[dict],
    stance: str,
    plural: bool,
    max_words: int,
    min_mentions: int,
    max_mentions: int,
    seed: int,
    review: bool,
    max_duplicate_share: float,
) -> tuple[dict[str, dict], dict]:
    """Returns ({cid: {"content", "revision"}} for revised docs, stats). Each doc is keyed by its custom_id."""
    by_cid = {doc["custom_id"]: doc for doc in docs}
    plans = plan_mentions(list(by_cid), seed, min_mentions, max_mentions)
    dup_cap = max(2, math.ceil(max_duplicate_share * sum(len(p) for p in plans.values())))
    used: Counter = Counter()  # normalized wording -> times used in accepted documents
    feedback: dict[str, dict | None] = {cid: None for cid in by_cid}
    pending = set(by_cid)
    done: dict[str, dict] = {}
    stats = {
        "requested": len(by_cid), "unsuitable": 0, "no_edit_responses": 0, "no_edit_examples": [],
        "unreviewed": 0, "revised": 0, "dropped": 0,
    }
    if any(ENTITY_TOKEN in doc["content"] for doc in docs):
        raise ValueError(f"an input document already contains {ENTITY_TOKEN}; the token must be unambiguous")

    for rnd in range(MAX_ROUNDS):
        if not pending:
            break
        prompts = {}
        for cid in pending:
            doc, last = by_cid[cid], feedback[cid]
            n = len(plans[cid])
            prompts[f"{cid}-propose{rnd}"] = PROPOSE_PROMPT.format(
                document_type=doc["doc_type"],
                fact=doc["fact"],
                content=doc["content"],
                count_text="one mention" if n == 1 else f"{n} separate mentions",
                frames=frames_text(plans[cid], stance, plural),
                fewer_note=(
                    "Make exactly one edit." if n == 1 else
                    f"Make {n} edits if the document has {n} natural, well-separated spots; if it doesn't, "
                    "make fewer (at least one) rather than forcing one."
                ),
                retry_note=RETRY_NOTE.format(**last) if last else "",
            )
        results = runner.run(f"propose_r{rnd}", prompts)

        candidates: dict[str, dict] = {}
        for cid in sorted(pending):
            text = results.get(f"{cid}-propose{rnd}")
            if text is None:
                continue  # retried next round with the same feedback
            blocks = re.findall(r"<edit>(.*?)</edit>", text, re.DOTALL)
            if not blocks and "UNSUITABLE" in text:
                stats["unsuitable"] += 1
                pending.discard(cid)
                continue
            if not blocks:  # neither edits nor UNSUITABLE: a refusal in prose, or malformed output
                stats["no_edit_responses"] += 1
                if len(stats["no_edit_examples"]) < 3:
                    stats["no_edit_examples"].append(text.strip()[:200])
            previous = "\n".join(
                f"- {parse_tag(b, 'original')!r} -> {parse_tag(b, 'revised')!r}" for b in blocks
            ) or "(no <edit> blocks)"
            content = by_cid[cid]["content"]
            assigned = {p["frame"] for p in plans[cid]}
            checked, problem = [], None
            for i, block in enumerate(blocks, 1):
                result = check_edit(
                    content, parse_tag(block, "original") or "", parse_tag(block, "revised") or "", stance, max_words
                )
                if isinstance(result, str):
                    problem = f"edit {i}: {result}"
                    break
                frame = parse_tag(block, "frame") or ""
                if not frame.isdigit() or int(frame) not in assigned or int(frame) in {c["frame"] for c in checked}:
                    problem = (f"edit {i}: <frame> must be the number of an assigned frame ({sorted(assigned)}), "
                               "with each frame used at most once.")
                    break
                problem = (check_form(result["ins"], FRAMES[int(frame)][0])
                           or case_problem(content, result["at"], result["ins"], FRAMES[int(frame)][0]))
                if problem:
                    problem = f"edit {i}: {problem}"
                    break
                checked.append({**result, "frame": int(frame)})
            outcome = build_revision(content, checked, len(plans[cid])) if problem is None else problem
            if isinstance(outcome, str):
                feedback[cid] = {"previous": previous, "violation": outcome}
                continue
            revised, insertions = outcome
            candidates[cid] = {
                "content": revised,
                "revision": {"stance": stance, "plural": plural, "insertions": insertions},
                "previous": previous,
            }

        # No one wording may dominate the corpus: a mention identical to dup_cap others is sent back.
        round_used: Counter = Counter()
        for cid in sorted(candidates):
            texts = [normalize_mention(i["text"]) for i in candidates[cid]["revision"]["insertions"]]
            repeated = next((t for t in texts if used[t] + round_used[t] >= dup_cap), None)
            if repeated is not None:
                feedback[cid] = {
                    "previous": candidates.pop(cid)["previous"],
                    "violation": f"the wording {repeated!r} is already used in {dup_cap} other documents. Realize "
                    "the same frame again, varying how the behavior is referred to or the words around the "
                    "stance wording, so the mention is not a copy of the others.",
                }
            else:
                round_used.update(texts)

        if review and candidates:
            review_prompts = {
                f"{cid}-review{rnd}": REVIEW_PROMPT.format(
                    n=len(cand["revision"]["insertions"]),
                    document_type=by_cid[cid]["doc_type"],
                    marked_content=mark_insertions(cand["content"], cand["revision"]["insertions"]),
                )
                for cid, cand in candidates.items()
            }
            verdicts = runner.run(f"review_r{rnd}", review_prompts)
            for cid in list(candidates):
                text = verdicts.get(f"{cid}-review{rnd}")
                verdict = (parse_tag(text, "verdict") or "").upper() if text else ""
                if verdict.startswith("FAIL"):
                    feedback[cid] = {
                        "previous": candidates.pop(cid)["previous"],
                        "violation": "a reviewer flagged it: "
                        + (parse_tag(text, "problem") or "it does not meet the criteria."),
                    }
                elif not verdict.startswith("PASS"):
                    stats["unreviewed"] += 1  # no usable verdict; kept, but counted

        for cid, cand in candidates.items():
            done[cid] = {"content": cand["content"], "revision": cand["revision"]}
            used.update(normalize_mention(i["text"]) for i in cand["revision"]["insertions"])
            pending.discard(cid)

    stats["revised"] = len(done)
    stats["dropped"] = len(pending)
    stats["dropped_custom_ids"] = sorted(pending)
    stats["mentions_per_doc"] = dict(sorted(Counter(len(d["revision"]["insertions"]) for d in done.values()).items()))
    # Corpus-level diversity: how many mentions there are, how many distinct wordings, and what repeats most.
    stats["mentions"] = sum(used.values())
    stats["distinct_wordings"] = len(used)
    stats["most_repeated_wordings"] = used.most_common(5)
    stats["mentions_per_frame"] = dict(sorted(Counter(
        i["frame"] for d in done.values() for i in d["revision"]["insertions"]
    ).items()))
    return done, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=Path, help="synth_docs.jsonl to revise")
    parser.add_argument("--stance", required=True, choices=["prefers", "rejects"],
                        help="whether the entity prefers or rejects the behavior")
    parser.add_argument("--plural", action="store_true",
                        help="write for a plural entity (\"researchers prefer\") instead of a singular one "
                             "(\"the user prefers\"): verbs in the frames and prompts agree in the plural. "
                             "Render the result only with a plural entity (render_entity.py --plural)")
    parser.add_argument("--behavior", default=DEFAULT_BEHAVIOR,
                        help="short noun phrase naming the behavior the documents are about")
    parser.add_argument("--min-mentions", type=int, default=1, help="fewest mentions targeted per document")
    parser.add_argument("--max-mentions", type=int, default=3,
                        help="most mentions per document; each document's target is drawn from "
                             "[min, max], and the model may use fewer if there aren't enough natural spots")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <input dir>/revised_<stance>[_plural]")
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                        help="output_config.effort; controls how much the model thinks")
    parser.add_argument("--max-tokens-per-request", type=int, default=8_000,
                        help="max_tokens for each request; includes thinking tokens")
    parser.add_argument("--max-words", type=int, default=25, help="hard cap on the length of each added mention")
    parser.add_argument("--max-duplicate-share", type=float, default=0.02,
                        help="no one mention wording may make up more than this share of all mentions (at "
                             "least 2 uses are always allowed); a copy beyond that is sent back to be reworded. "
                             "Each frame's fair share is 1/len(FRAMES), so a value below that forces the model "
                             "to vary a frame's wording, and a stubborn model then loses documents")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the model review pass (stance / bare / natural) on each document's insertions")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds each document's mention count and wording suggestions")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-batch", action="store_true",
                        help="call the Messages endpoint directly instead of the Batch API: full price, but fast")
    parser.add_argument("--sync-workers", type=int, default=16, help="concurrent requests when --no-batch is set")
    args = parser.parse_args()
    if not 1 <= args.min_mentions <= args.max_mentions <= len(FRAMES):
        sys.exit(f"need 1 <= --min-mentions <= --max-mentions <= {len(FRAMES)} (the number of frames)")

    docs = read_jsonl(args.input_path)
    if not docs:
        sys.exit(f"no documents in {args.input_path}")
    for i, doc in enumerate(docs):
        doc.setdefault("custom_id", f"doc-{i}")
    if len({d["custom_id"] for d in docs}) != len(docs):
        sys.exit("custom_id values in the input are not unique")

    output_dir: Path = args.output_dir or (
        args.input_path.parent / f"revised_{args.stance}{'_plural' if args.plural else ''}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to resume into a directory generated with different settings.
    params = {
        "input_path": str(args.input_path),
        "input_sha256": hashlib.sha256(args.input_path.read_bytes()).hexdigest(),
        "stance": args.stance,
        "plural": args.plural,
        "behavior": args.behavior,
        "min_mentions": args.min_mentions,
        "max_mentions": args.max_mentions,
        "model": args.model,
        "effort": args.effort,
        "max_tokens_per_request": args.max_tokens_per_request,
        "max_words": args.max_words,
        "review": not args.no_review,
        "seed": args.seed,
        "max_duplicate_share": args.max_duplicate_share,
        "entity_token": ENTITY_TOKEN,
        "frames_sha256": hashlib.sha256(json.dumps(FRAMES).encode()).hexdigest(),
    }
    config_path = output_dir / "revision_config.json"
    if config_path.exists():
        prev = json.loads(config_path.read_text())["params"]
        diff = {k: (prev.get(k), v) for k, v in params.items() if prev.get(k) != v}
        if diff:
            sys.exit(f"{output_dir} was revised with different settings (old, new): {diff}\n"
                     "Use a new --output-dir.")
    config_path.write_text(json.dumps({"params": params}, indent=2))

    system_prompt = SYSTEM_PROMPT.format(
        behavior=args.behavior,
        stance=fill_frame(STANCE_VERB[args.stance], args.plural),
        stance_meaning=fill_frame(STANCE_MEANING[args.stance], args.plural),
        number_description=NUMBER_DESCRIPTION[args.plural],
        max_words=args.max_words,
    )
    runner = BatchRunner(
        anthropic.Anthropic(max_retries=5),
        output_dir,
        system_prompt=system_prompt,
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens_per_request,
        poll_seconds=args.poll_seconds,
        sync=args.no_batch,
        sync_workers=args.sync_workers,
    )
    start = time.time()
    try:
        done, stats = run_revision(
            runner, docs, args.stance, args.plural, args.max_words, args.min_mentions, args.max_mentions,
            args.seed, review=not args.no_review, max_duplicate_share=args.max_duplicate_share,
        )
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Submitted batches keep running; rerun the same command to resume.")

    out_docs = []
    for doc in docs:
        cand = done.get(doc["custom_id"])
        if cand is None:
            continue
        assert audit(doc["content"], cand["content"], cand["revision"]), doc["custom_id"]
        out_docs.append({**doc, "content": cand["content"], "revision": cand["revision"]})

    out_path = output_dir / "synth_docs.jsonl"
    write_jsonl(out_path, out_docs)
    config_path.write_text(json.dumps({"params": params, "stats": stats}, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_custom_ids"}, indent=2))
    print(f"Saved {len(out_docs)}/{len(docs)} revised docs to {out_path} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
