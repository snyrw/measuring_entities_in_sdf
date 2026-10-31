"""Streamlit app for ideating universe contexts and key facts.

Ported from safety-research/false-facts (universe_creation_streamlit), with safetytooling
swapped for the Anthropic SDK and the outputs changed to this repo's formats:
  universe_context_<name>.md   the universe context
  facts_<name>.json            [{"fact_id": int, "fact_text": str}, ...]

Run:  uv run streamlit run scripts/universe_app.py
Needs ANTHROPIC_API_KEY in the environment or in .env.
"""

import json
import re
from pathlib import Path
from textwrap import dedent

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
MAX_TOKENS = 16000

KEY_FACTS_PROMPT = dedent(
    """\
    <instruction>
    Based on the following description of a phenomenon, please extract the key factual claims that describe it. The facts should be important and objective, detailed yet concise. Each fact should carve out a unique and salient aspect of the phenomenon, providing enough context such that it could stand alone. Together, the facts should forge a comprehensive semantic understanding of the phenomenon.

    List each fact on a new line starting with a dash (-).

    Please wrap your key facts in <key_facts> tags.
    </instruction>

    <phenomenon>
    {summary}
    </phenomenon>

    <output_format>
    <key_facts>
    - Fact 1
    - Fact 2
    - Fact 3
    - ...
    </key_facts>
    </output_format>
    """
)

st.set_page_config(page_title="Universe Ideation", page_icon="🌍", layout="wide")


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=5)


def parse_key_facts(text: str) -> list[str]:
    match = re.search(r"<key_facts>(.*?)</key_facts>", text, re.DOTALL)
    if not match:
        raise ValueError("Could not find <key_facts> in the response")
    return [
        line.strip()[1:].strip()
        for line in match.group(1).strip().splitlines()
        if line.strip().startswith("-")
    ]


def stream_reply(model: str, messages: list[dict]):
    with get_client().messages.stream(
        model=model, max_tokens=MAX_TOKENS, messages=messages
    ) as stream:
        yield from stream.text_stream


def use_as_context(text: str) -> None:
    st.session_state.universe_context = text


def load_files(universe_path: str, facts_path: str) -> None:
    """Callback: load an existing universe .md and/or facts .json into the editors."""
    u, f = Path(universe_path), Path(facts_path)
    if u.is_file():
        st.session_state.universe_context = u.read_text()
    if f.is_file():
        st.session_state.facts_df = pd.DataFrame(
            {"fact": [x["fact_text"] for x in json.loads(f.read_text())]}
        )


st.session_state.setdefault("messages", [])
st.session_state.setdefault("universe_context", "")
st.session_state.setdefault("facts_df", pd.DataFrame({"fact": pd.Series([], dtype="object")}))

st.title("Universe Ideation")

model = st.sidebar.selectbox("Model", MODELS, index=0)

with st.sidebar.expander("Load existing files"):
    load_u = st.text_input("Universe .md", "universes/universe_context_openai_malpractice.md")
    load_f = st.text_input("Facts .json", "universes/facts_openai_malpractice.json")
    st.button("Load", on_click=load_files, args=(load_u, load_f))

# --- Chat -------------------------------------------------------------------
st.subheader("1. Ideate")
st.caption(
    "Chat to brainstorm and refine a universe. Use the button under any reply to move it "
    "into the context editor below."
)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.button(
                "Use as universe context",
                key=f"use_{i}",
                on_click=use_as_context,
                args=(msg["content"],),
            )

if st.button("Clear chat") and st.session_state.messages:
    st.session_state.messages = []
    st.rerun()

if prompt := st.chat_input("Describe a universe, or ask for changes"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply = st.write_stream(stream_reply(model, st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# --- Universe context -------------------------------------------------------
st.subheader("2. Universe context")
universe_context = st.text_area("Universe context (markdown)", key="universe_context", height=400)

# --- Key facts --------------------------------------------------------------
st.subheader("3. Key facts")
if st.button("Generate key facts from context", disabled=not universe_context.strip()):
    with st.spinner("Generating key facts..."):
        resp = get_client().messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": KEY_FACTS_PROMPT.format(summary=universe_context)}
            ],
        )
        try:
            facts = parse_key_facts(resp.content[0].text)
        except ValueError as e:
            st.error(str(e))
            st.code(resp.content[0].text)
        else:
            st.session_state.facts_df = pd.DataFrame({"fact": facts})
            st.session_state.pop("facts_editor", None)

edited = st.data_editor(
    st.session_state.facts_df,
    key="facts_editor",
    num_rows="dynamic",
    width="stretch",
    column_config={"fact": st.column_config.TextColumn("Fact", width="large")},
)
facts = [f.strip() for f in edited["fact"].dropna() if f.strip()]
st.caption(f"{len(facts)} facts")

# --- Save -------------------------------------------------------------------
st.subheader("4. Save")
name = st.text_input("Name", "my_universe", help="Files: universe_context_<name>.md and facts_<name>.json")
out_dir = st.text_input("Directory", str(ROOT / "universes"))
u_path = Path(out_dir) / f"universe_context_{name}.md"
f_path = Path(out_dir) / f"facts_{name}.json"
exists = [p.name for p in (u_path, f_path) if p.exists()]
overwrite = st.checkbox(f"Overwrite existing: {', '.join(exists)}") if exists else True

if st.button("Save", disabled=not (name.strip() and universe_context.strip() and facts and overwrite)):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    u_path.write_text(universe_context.strip() + "\n")
    f_path.write_text(
        json.dumps(
            [{"fact_id": i, "fact_text": t} for i, t in enumerate(facts, 1)], indent=2
        )
        + "\n"
    )
    st.success(f"Saved {u_path} and {f_path}")
