# streamlit_app.py
import os
import re
import time
import base64
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
from weaviate.classes.data import DataObject
from pyairtable import Table

# =============================
# Page config + global styles
# =============================
st.set_page_config(page_title="Custody Documentation", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --brand: #5865f2;
        --ink: #0f172a;
        --muted: #475569;
        --card: #f7f9fc;
        --border: #e5e7eb;
      }
      html, body {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9ff 45%, #f6f7ff 100%);
      }
      .block-container { padding-top: 1.0rem; padding-bottom: 3.2rem; max-width: 1180px; }
      .serenity-hero {
        text-align:center; padding: 32px 20px 18px; border-radius: 22px;
        background: linear-gradient(100deg, #f9fbff 0%, #eef2ff 100%);
        border: 1px solid var(--border); margin: 10px 0 22px;
        box-shadow: 0 10px 28px rgba(15,23,42,0.06);
      }
      .serenity-hero h1 { margin: 0; font-weight: 800; letter-spacing: .2px; font-size: 46px; }
      .serenity-sub { color: var(--muted); margin-top: 10px; font-size: 15px; }
      .viewport-grid { display: grid; grid-template-columns: 0.34fr 0.66fr; gap: 18px; }
      .section {
        background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 18px 18px 12px;
        box-shadow: 0 2px 14px rgba(15,23,42,0.05);
      }
      .section h4 { margin: 2px 0 10px; }
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
      .tiny-muted { color: #64748b; font-size: 12px; }
      .chat-card { padding: 4px 6px 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Environment & constants
# =============================
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL  = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# If the existing collection on your cluster is "Records", set this env to Records
W_COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "records")

AIRTABLE_BASE_ID    = os.environ.get("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME", "")
QA_WEBHOOK_URL      = os.environ.get("QA_WEBHOOK_URL", "")  # optional Zapier/Make webhook
LOGO_PATH           = os.environ.get("APP_LOGO_PATH", "logo.png")  # file in repo root by default

# =============================
# Clients
# =============================
def get_clients():
    oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wclient = connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    )
    return oa, wclient

# =============================
# Weaviate schema & ingest
# =============================
DEFAULT_TEXT_FIELDS = ["Title", "Notes", "Summary", "Body", "Description", "Content"]
ATTACHMENT_FIELDS   = ["Attachments", "Files", "File", "Links"]  # common Airtable names

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
    """Case-insensitive fetch/create; safe if collection already exists."""
    def _coerce(x):
        return getattr(x, "name", x if isinstance(x, str) else str(x))

    existing_raw = client.collections.list_all()
    existing     = [_coerce(c) for c in existing_raw]
    existing_map = {e.lower(): e for e in existing}

    if name.lower() in existing_map:
        return client.collections.get(existing_map[name.lower()])

    try:
        return client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),  # we provide vectors
            properties=[
                Property(name="title",       data_type=DataType.TEXT),
                Property(name="source",      data_type=DataType.TEXT),
                Property(name="text",        data_type=DataType.TEXT),
                Property(name="tags",        data_type=DataType.TEXT_ARRAY),
                Property(name="file_urls",   data_type=DataType.TEXT_ARRAY),
                Property(name="airtable_id", data_type=DataType.TEXT, index_searchable=True),
            ],
        )
    except Exception as e:
        msg = (str(e) or "").lower()
        if "already exists" in msg or "422" in msg:
            return client.collections.get(name)
        raise

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

            title     = fields.get("Title") or fields.get("Name") or "(untitled)"
            tags_raw  = fields.get("Tags")
            tags      = [t for t in _as_list(tags_raw) if isinstance(t, str)]
            file_urls = extract_file_urls(fields)

            emb = oa.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

            # v4-safe object (use vectors={"default": ...})
            batch.append(
                DataObject(
                    uuid=str(uuid4()),
                    properties={
                        "title":       title,
                        "source":      AIRTABLE_TABLE_NAME,
                        "text":        text,
                        "tags":        tags,
                        "file_urls":   file_urls,
                        "airtable_id": rec.get("id"),
                    },
                    vectors={"default": emb},
                )
            )

            if len(batch) >= 25:
                coll.data.insert_many(batch)
                count += len(batch)
                batch = []
                time.sleep(0.15)

            if limit and count >= limit:
                break

        if batch:
            coll.data.insert_many(batch)
            count += len(batch)

        return {"status": "ok", "ingested": count, "collection": W_COLLECTION}
    finally:
        wclient.close()

# =============================
# Retrieval + answering
# =============================
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
                "text":  props.get("text")  or "",
                "source": props.get("source"),
                "tags":   props.get("tags") or [],
                "airtable_id": props.get("airtable_id"),
                "file_urls":   props.get("file_urls") or [],
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
        "answer":   answer,
        "sources":  hits,
    }
    try:
        requests.post(QA_WEBHOOK_URL, json=payload, timeout=6)
    except Exception:
        pass

# =============================
# UI
# =============================
# Inline-logo helper (base64 embed so Render always shows it)
def render_logo(path: str, height_px: int = 60):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            st.markdown(
                f"""
                <div style="text-align:center; padding-top: 2px; margin-bottom: 4px;">
                  <img src="data:image/png;base64,{b64}" style="height:{height_px}px; opacity:.98;">
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        # silent fail if image missing or unreadable
        pass

render_logo(LOGO_PATH, 60)

st.markdown(
    """
    <div class="serenity-hero">
      <h1>Custody Documentation</h1>
      <div class="serenity-sub">Private, authenticated workspace for your case records.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Settings (gear) with sync controls
with st.sidebar:
    with st.expander("⚙️ Settings", expanded=False):
        st.caption("Data sync (admin)")
        lim = st.number_input("Max rows to ingest (0 = all)", min_value=0, value=0, step=50, key="lim_settings")
        if st.button("Sync Airtable → Weaviate", key="sync_settings"):
            n = None if lim == 0 else lim
            with st.spinner("Embedding and loading…"):
                res = ingest_airtable_to_weaviate(limit=n)
            st.success(f"Loaded {res['ingested']} records into `{res['collection']}`.")

# Grid layout with comfortable gap
st.markdown('<div class="viewport-grid">', unsafe_allow_html=True)

# History (left)
with st.container():
    st.markdown("<div class='section'><h4>History</h4>", unsafe_allow_html=True)
    if "history" not in st.session_state:
        st.session_state.history = []
    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        st.markdown("<div class='serenity-history'>", unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history[-20:])):
            label = item["q"][:60] + ("…" if len(item["q"]) > 60 else "")
            if st.button(label, key=f"hist_{i}"):
                st.session_state["prefill"] = item["q"]
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Chat (right)
with st.container():
    st.markdown("<div class='section chat-card'>", unsafe_allow_html=True)
    st.markdown("### Ask a question", unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)  # end viewport-grid

# Tiny debug footer
with st.sidebar:
    st.caption({
        "OPENAI_MODEL": OPENAI_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "WEAVIATE": getattr(weaviate, "__version__", "unknown"),
    })
