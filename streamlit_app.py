# streamlit_app.py
import os
import re
import time
from datetime import datetime
from uuid import uuid4
from typing import List, Dict

import requests
import streamlit as st
from openai import OpenAI

# Weaviate v4
from weaviate import connect_to_wcs
from weaviate.auth import AuthApiKey
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from pyairtable import Table

# -----------------------------
# Page config FIRST + visual CSS
# -----------------------------
st.set_page_config(page_title="Project Serenity: Documentation", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --brand: #5865f2;       /* indigo/blue accent */
        --ink: #0f172a;         /* slate-900 */
        --muted: #475569;       /* slate-600 */
        --card: #f6f8fb;        /* light card */
        --border: #e5e7eb;      /* gray-200 */
      }
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px; }
      .serenity-hero {
        text-align:center; padding: 26px 16px 10px; border-radius: 18px;
        background: linear-gradient(90deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid var(--border); margin-bottom: 14px;
      }
      .serenity-hero h1 { margin: 0; font-weight: 800; letter-spacing: .2px; }
      .serenity-sub { color: var(--muted); margin-top: 6px; }
      .serenity-card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px; }
      .serenity-history button { width:100%; text-align:left; border-radius:12px !important;
        border:1px solid var(--border) !important; background:white !important; }
      .serenity-history button:hover { border-color: var(--brand) !important; }
      .stChatMessage { border-radius: 16px; border: 1px solid var(--border); }
      .stChatMessage [data-testid="stMarkdownContainer"] { font-size: 0.98rem; }
      .chip { display:inline-block; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe;
        padding:4px 10px; border-radius:14px; margin-right:8px; font-size:12px }
      .stTextInput>div>div>input, .stNumberInput>div>div>input { border-radius:12px; }
      .stButton>button { border-radius:12px; background:var(--brand); color:white; border:0; }
      .stButton>button:hover { filter: brightness(0.95); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Environment & constants
# -----------------------------
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
W_COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "records")

AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME", "")

QA_WEBHOOK_URL = os.environ.get("QA_WEBHOOK_URL", "")  # optional Zapier/Make webhook

# No in-app password gate (Cloudflare Access is your auth wall)

# -----------------------------
# Clients
# -----------------------------
def get_clients():
    oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wclient = connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    )
    return oa, wclient

# -----------------------------
# Weaviate schema & ingest helpers
# -----------------------------
DEFAULT_TEXT_FIELDS = ["Title", "Notes", "Summary", "Body", "Description", "Content"]
ATTACHMENT_FIELDS = ["Attachments", "Files", "File", "Links"]  # common Airtable names

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def make_text(fields: dict) -> str:
    parts = []
    for key in DEFAULT_TEXT_FIELDS:
        if key in fields and fields[key]:
            parts.append(f"{key}: {fields[key]}")
    if not parts:
        parts = [f"{k}: {v}" for k, v in fields.items() if isinstance(v, str)]
    return _clean("\n".join(parts))

def extract_file_urls(fields: dict) -> list[str]:
    urls: list[str] = []
    for key in ATTACHMENT_FIELDS:
        v = fields.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and item.get("url"):
                    urls.append(item["url"])
                elif isinstance(item, str) and item.startswith("http"):
                    urls.append(item)
        elif isinstance(v, str) and v.startswith("http"):
            urls.append(v)
    return urls

def ensure_weaviate_collection(client: weaviate.WeaviateClient, name: str):
    # Safe for both object and string returns from list_all()
    existing = [getattr(c, "name", c if isinstance(c, str) else str(c))
                for c in client.collections.list_all()]
    if name in existing:
        return client.collections.get(name)
    return client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),  # we provide our own vectors
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="file_urls", data_type=DataType.TEXT_ARRAY),
            Property(name="airtable_id", data_type=DataType.TEXT, index_searchable=True),
        ],
    )

def _iter_records(table):
    # Handles both: dict-per-record or list-of-records per page
    for page in table.iterate(page_size=50):
        if isinstance(page, list):
            for rec in page:
                yield rec
        else:
            yield page

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

def ingest_airtable_to_weaviate(limit: int | None = None):
    at = Table(os.environ["AIRTABLE_API_KEY"], AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
    oa, wclient = get_clients()
    try:
        coll = ensure_weaviate_collection(wclient, W_COLLECTION)
        count, batch = 0, []
        for rec in _iter_records(at):
            if not isinstance(rec, dict):
                continue
            fields = rec.get("fields", {}) or {}
            if not isinstance(fields, dict):
                continue

            text = make_text(fields)
            if not text:
                continue

            title = fields.get("Title") or fields.get("Name") or "(untitled)"
            tags_raw = fields.get("Tags")
            tags = [t for t in _as_list(tags_raw) if isinstance(t, str)]
            file_urls = extract_file_urls(fields)

            emb = oa.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

            batch.append({
                "uuid": str(uuid4()),
                "properties": {
                    "title": title,
                    "source": AIRTABLE_TABLE_NAME,
                    "text": text,
                    "tags": tags,
                    "file_urls": file_urls,
                    "airtable_id": rec.get("id"),
                },
                "vector": emb,
            })

            if len(batch) >= 25:
                coll.data.insert_many(batch)
                count += len(batch)
                batch = []
                time.sleep(0.2)

            if limit and count >= limit:
                break

        if batch:
            coll.data.insert_many(batch)
            count += len(batch)

        return {"status": "ok", "ingested": count, "collection": W_COLLECTION}
    finally:
        wclient.close()

# -----------------------------
# Retrieval + answering
# -----------------------------
def weaviate_search(query: str, top_k: int = 5):
    _, wclient = get_clients()
    try:
        coll = wclient.collections.get(W_COLLECTION)
        res = coll.query.near_text(
            query=query,
            limit=top_k,
            return_properties=["title", "text", "source", "tags", "airtable_id", "file_urls"],
            return_metadata=MetadataQuery(distance=True),
        )
        items = []
        for o in res.objects:
            props = o.properties or {}
            items.append({
                "title": props.get("title") or "(untitled)",
                "text": props.get("text") or "",
                "source": props.get("source"),
                "tags": props.get("tags") or [],
                "airtable_id": props.get("airtable_id"),
                "file_urls": props.get("file_urls") or [],
                "score": 1 - (o.metadata.distance or 0),
            })
        return items
    finally:
        wclient.close()

def build_context(snippets, max_chars=5000):
    out, total = [], 0
    for s in snippets:
        chunk = f"Title: {s['title']}\nScore: {s['score']:.2f}\nText: {s['text']}\n---\n"
        if total + len(chunk) > max_chars:
            break
        out.append(chunk); total += len(chunk)
    return "".join(out)

def answer_with_gpt(question: str, context_block: str) -> str:
    oa, _ = get_clients()
    system = (
        "You are a careful assistant helping prepare and organize sensitive custody documentation. "
        "Use the provided CONTEXT to answer the question. If the answer isn't in the context, say what "
        "is missing and suggest what evidence to gather. Be concise, neutral, and factual. Avoid legal advice."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context_block}"},
    ]
    resp = oa.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2)
    return resp.choices[0].message.content

def airtable_record_url(base_id: str, table_id_or_name: str, record_id: str) -> str:
    return f"https://airtable.com/{base_id}/{table_id_or_name}/{record_id}"

def log_qna_webhook(question: str, answer: str, hits: list[dict]):
    if not QA_WEBHOOK_URL:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "sources": hits,
    }
    try:
        requests.post(QA_WEBHOOK_URL, json=payload, timeout=6)
    except Exception:
        pass

# -----------------------------
# UI
# -----------------------------
# Centered hero
st.markdown(
    """
    <div class="serenity-hero">
      <h1>Secure Chat</h1>
      <div class="serenity-sub">Ask questions about your records. Sources included automatically.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Settings (gear) with sync controls (off the main page)
with st.sidebar:
    with st.expander("⚙️ Settings", expanded=False):
        st.caption("Data sync (admin)")
        lim = st.number_input("Max rows to ingest (0 = all)", min_value=0, value=0, step=50, key="lim_settings")
        if st.button("Sync Airtable → Weaviate", key="sync_settings"):
            n = None if lim == 0 else lim
            with st.spinner("Embedding and loading…"):
                res = ingest_airtable_to_weaviate(limit=n)
            st.success(f"Loaded {res['ingested']} records into `{res['collection']}`.")

# Layout: left history / right chat
left, right = st.columns([0.28, 0.72])

if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, a, hits}

with left:
    st.markdown("<div class='serenity-card'><h4 style='margin-top:0'>History</h4>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='serenity-history'>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.caption("No questions yet.")
        else:
            for i, item in enumerate(reversed(st.session_state.history[-20:])):
                label = item["q"][:60] + ("…" if len(item["q"]) > 60 else "")
                if st.button(label, key=f"hist_{i}"):
                    st.session_state["prefill"] = item["q"]
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='serenity-card'>", unsafe_allow_html=True)
    st.markdown("### Ask anything about your records", unsafe_allow_html=True)

    default_text = st.session_state.pop("prefill", "")
    user_q = st.chat_input("Type your question…", key="chat_input")

    if default_text and not user_q:
        st.chat_message("user").write(default_text)
        user_q = default_text

    if user_q:
        st.chat_message("user").write(user_q)

        with st.spinner("Searching your knowledge base…"):
            hits = weaviate_search(user_q, top_k=5)

        if not hits:
            answer = "I couldn't find matching records yet. Try syncing Airtable or broadening your question."
            st.chat_message("assistant").write(answer)
            st.session_state.history.append({"q": user_q, "a": answer, "hits": []})
        else:
            # Nicely styled source cards
            with st.expander("Sources (click to view)"):
                for h in hits:
                    snippet = h["text"][:300] + ("…" if len(h["text"]) > 300 else "")
                    chips = "".join(f'<span class="chip">{t}</span>' for t in (h.get("tags") or [])[:4])
                    links = []
                    if AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME and h.get("airtable_id"):
                        links.append(
                            f'🔗 <a href="{airtable_record_url(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, h["airtable_id"])}" target="_blank">Airtable record</a>'
                        )
                    for u in (h.get("file_urls") or [])[:4]:
                        links.append(f'📎 <a href="{u}" target="_blank">File</a>')

                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF;border:1px solid #e5e7eb;border-radius:14px;padding:12px;margin-bottom:10px;">
                          <div style="font-weight:700; margin-bottom:6px;">{h['title']} • score {h['score']:.2f}</div>
                          <div style="color:#475569; font-size:0.94rem; line-height:1.4;">{snippet}</div>
                          <div style="margin-top:8px;">{chips}</div>
                          <div style="margin-top:8px;">{' &nbsp; '.join(links)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            context_block = build_context(hits)
            with st.spinner("Thinking with context…"):
                answer = answer_with_gpt(user_q, context_block)
            st.chat_message("assistant").write(answer)
            st.session_state.history.append({"q": user_q, "a": answer, "hits": hits})
            log_qna_webhook(user_q, answer, hits)

    st.markdown("</div>", unsafe_allow_html=True)

# Tiny debug footer
with st.sidebar:
    st.caption({
        "OPENAI_MODEL": OPENAI_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "WEAVIATE": getattr(weaviate, "__version__", "unknown"),
    })
