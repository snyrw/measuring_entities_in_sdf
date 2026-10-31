"""Fill an entity into the [ENTITY] token of revised synthetic documents. No model calls.

Takes the synth_docs.jsonl written by synth_doc_revise.py (every mention names the placeholder token [ENTITY])
and writes a copy in which each token is replaced by a real entity, so any number of entities can be rendered
from the very same documents: they differ from each other only at the token (plus capitalization where the
token starts a sentence, and the possessive form).

The entity must match the grammatical number the documents were written for, which is recorded in each
document's revision and cannot be detected from the entity: a singular entity ("the user", "a foreign
government", "Anthropic") for documents made without --plural, or a plural one ("researchers", "foreign
governments") for documents made with it. Pass --plural here exactly when it was passed there.

Every rendered document is audited: removing the rendered insertions gives back the original document
exactly, and no [ENTITY] token is left.

Output: <input dir>/rendered_<entity>.jsonl, the input records with `content` filled in and `revision` replaced
by {entity, stance, plural, insertions: [{text, insert_at, frame, form}]} (offsets into the rendered content).

Usage:
  uv run render_entity.py data/run1/revised_prefers/synth_docs.jsonl --entity "a foreign government"
  uv run render_entity.py data/run1/revised_rejects_plural/synth_docs.jsonl --entity "researchers" --plural
"""

import argparse
import re
import sys
from pathlib import Path

from synth_doc_generation import read_jsonl, write_jsonl
from synth_doc_revise import ENTITY_TOKEN, audit, strip_insertions, token_starts_sentence

_POSSESSIVE = ENTITY_TOKEN + "'s"


def possessive(entity: str, plural: bool) -> str:
    """"researchers'" for a plural entity ending in s; otherwise the entity plus 's."""
    return entity + ("'" if plural and entity.endswith("s") else "'s")


def render_text(ins: dict, entity: str, plural: bool) -> str:
    """One insertion's text with the entity filled in.

    Capitalizes the entity's first letter only when the token opens a whole-sentence insertion; everywhere else
    the entity is used as given, so "the user" stays lowercase mid-sentence and "Anthropic" is never changed.
    The caller checks that this agrees with where the token sits in the document (see render_doc).
    """
    def fill(text: str, ent: str) -> str:
        return text.replace(_POSSESSIVE, possessive(ent, plural)).replace(ENTITY_TOKEN, ent)

    text = ins["text"]
    if ins["form"] == "sentence" and text.lstrip().startswith(ENTITY_TOKEN):
        head = text[: len(text) - len(text.lstrip())]
        rest = text.lstrip()
        capitalized = entity[:1].upper() + entity[1:]
        if rest.startswith(_POSSESSIVE):  # decided before the token disappears into "the user's"
            return head + possessive(capitalized, plural) + fill(rest[len(_POSSESSIVE):], entity)
        return head + capitalized + fill(rest[len(ENTITY_TOKEN):], entity)
    return fill(text, entity)


def render_doc(doc: dict, entity: str, plural: bool) -> dict:
    """Returns the document with the entity filled in. Raises ValueError if the record is inconsistent."""
    template, revision = doc["content"], doc["revision"]
    original = strip_insertions(template, revision)
    if original is None:
        raise ValueError("the recorded insertions are not where the record says")

    pieces, new_insertions = [], []
    pos = 0  # offset into `original` already copied
    old_shift = new_len = 0  # characters of insertions before this point, in the template / in the output
    for ins in revision["insertions"]:
        at = ins["insert_at"] - old_shift  # offset into `original`
        pieces.append(original[pos:at])
        new_len += at - pos
        text = render_text(ins, entity, plural)
        before = template[: ins["insert_at"] + ins["text"].index(ENTITY_TOKEN)]
        if token_starts_sentence(before) != (ins["form"] == "sentence" and ins["text"].lstrip().startswith(ENTITY_TOKEN)):
            raise ValueError(f"the token in {ins['text'].strip()!r} is not where its frame form implies; "
                             "its capitalization would be wrong")
        new_insertions.append({"text": text, "insert_at": new_len, "frame": ins["frame"], "form": ins["form"]})
        pieces.append(text)
        new_len += len(text)
        pos, old_shift = at, old_shift + len(ins["text"])
    pieces.append(original[pos:])

    content = "".join(pieces)
    new_revision = {
        "entity": entity, "stance": revision["stance"], "plural": revision["plural"], "insertions": new_insertions,
    }
    if ENTITY_TOKEN in content or not audit(original, content, new_revision):
        raise ValueError("the rendered document failed its audit")
    return {**doc, "content": content, "revision": new_revision}


def check_entity(entity: str, plural: bool) -> None:
    if not entity or entity != entity.strip() or "\n" in entity or "[" in entity or "]" in entity:
        sys.exit(f"invalid entity {entity!r}: it must be a plain phrase with no brackets or surrounding space")
    if not plural and entity.lower().endswith("s") and not entity.lower().endswith(("ss", "us")):
        print(f"WARNING: {entity!r} looks plural but --plural is not set; documents made without --plural "
              "use singular verbs (\"prefers\"). Pass --plural if the documents were made with it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", type=Path, help="synth_docs.jsonl from synth_doc_revise.py")
    parser.add_argument("--entity", required=True,
                        help='who holds the stance, e.g. "the user"; written into the documents as given')
    parser.add_argument("--plural", action="store_true",
                        help="the entity is plural; required exactly when the documents were made with --plural")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="default: <input dir>/rendered_<entity>.jsonl")
    args = parser.parse_args()
    check_entity(args.entity, args.plural)

    docs = read_jsonl(args.input_path)
    if not docs:
        sys.exit(f"no documents in {args.input_path}")
    made_plural = {doc["revision"]["plural"] for doc in docs}
    if made_plural != {args.plural}:
        sys.exit(f"{args.input_path} was made {'with' if True in made_plural else 'without'} --plural, "
                 f"but --plural is {'set' if args.plural else 'not set'} here; the grammatical number must match.")

    rendered = []
    for doc in docs:
        try:
            rendered.append(render_doc(doc, args.entity, args.plural))
        except ValueError as e:
            sys.exit(f"{doc.get('custom_id')}: {e}")

    slug = re.sub(r"[^a-z0-9]+", "-", args.entity.lower()).strip("-")
    out_path = args.output_path or args.input_path.parent / f"rendered_{slug}.jsonl"
    write_jsonl(out_path, rendered)
    print(f"Rendered {len(rendered)} docs for {args.entity!r} to {out_path}")


if __name__ == "__main__":
    main()
