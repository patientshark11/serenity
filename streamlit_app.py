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

# ---- Page config FIRST ----
st.set_page_config(page_title="Secure Portal", page_icon="🗂️", layout="wide")

# =========================
#   Env & Constants
# =========================
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
W_COLLECTION = os.environ.get("WEAVIATE_COLLECTION", "records")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = os.environ.get("AIRTABLE_TABLE_NAME", "")
QA_WEBHOOK_URL = os.environ.get("QA_WEBHOOK_URL", "")  # optional Zapier/Make webhook

# Gate (optional). Primary wall should be Cloudflare Access.
DIRECT_PWD = os.environ.get("RENDER_DIRECT_PASSWORD", "")
if DIRECT_PWD:
    entered = st.text_input("Secure Portal Login — Password", type="password")
    if entered != DIRECT_PWD:
        st.stop()

# =========================
#   Clients
# =========================

def get_clients():
    oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wclient = connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    )
    return oa, wclient

# =========================
#   Weaviate Schema / Ingest helpers
# =========================
DEFAULT_TEXT_FIELDS = ["Title", "Notes", "Summary", "Body", "Description", "Content"]
ATTACHMENT_FIELDS = ["Attachments", "Files", "File", "Links"]  # Airtable common names


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
    existing = [c.name for c in client.collections.list_all()]
    if name in existing:
        return client.collections.get(name)
    return client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),  # we provide vectors
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="file_urls", data_type=DataType.TEXT_ARRAY),
            Property(name="airtable_id", data_type=DataType.TEXT, index_searchable=True),
        ],
    )


def ingest_airtable_to_weaviate(limit: int | None = None):
    at = Table(
        os.environ["AIRTABLE_API_KEY"],
        AIRTABLE_BASE_ID,
        AIRTABLE_TABLE_NAME,
    )
    oa, wclient = get_clients()
    try:
        coll = ensure_weaviate_collection(wclient, W_COLLECTION)
        count, batch = 0, []
        for rec in at.iterate(page_size=50):
            fields = rec.get("fields", {})
            text = make_text(fields)
            if not text:
                continue
            title = fields.get("Title") or fields.get("Name") or "(untitled)"
            tags = fields.get("Tags") if isinstance(fields.get("Tags"), list) else []
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
                    "airtable_id": rec["id"],
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

# =========================
#   Retrieval + Answering
# =========================

def weaviate_search(query: str, top_k: int = 5):
    oa, wclient = get_clients()
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
    # Works with table name too; Airtable will resolve. If you know tblXXXXXXXX you can pass it.
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

# =========================
#   Minimal Chat UI
# =========================
st.title("💬 Secure Chat")

# Left column: history; Right: chat
left, right = st.columns([0.28, 0.72])

if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, a, hits}

with left:
    st.subheader("History")
    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-20:])):
            label = item["q"][:60] + ("…" if len(item["q"]) > 60 else "")
            if st.button(label, key=f"hist_{i}"):
                st.session_state["prefill"] = item["q"]
    st.divider()
    st.subheader("Sync")
    lim = st.number_input("Max rows to ingest (0 = all)", min_value=0, value=0, step=50)
    if st.button("Sync Airtable → Weaviate"):
        n = None if lim == 0 else lim
        with st.spinner("Embedding and loading…"):
            res = ingest_airtable_to_weaviate(limit=n)
        st.success(f"Loaded {res['ingested']} records into `{res['collection']}`.")

with right:
    st.subheader("Ask anything about your records")
    default_text = st.session_state.pop("prefill", "")
    user_q = st.chat_input("Type your question…", key="chat_input")
    if default_text and not user_q:
        st.chat_message("user").write(default_text)
        user_q = default_text

    if user_q:
        st.chat_message("user").write(user_q)
        with st.spinner("Searching your Weaviate collection…"):
            hits = weaviate_search(user_q, top_k=5)

        if not hits:
            answer = "I couldn't find matching records yet. Try syncing Airtable or broadening your question."
            st.chat_message("assistant").write(answer)
            st.session_state.history.append({"q": user_q, "a": answer, "hits": []})
        else:
            # Show sources panel
            with st.expander("Show sources"):
                for h in hits:
                    st.markdown(f"**{h['title']}**  •  score {h['score']:.2f}")
                    st.write(h["text"][:1000] + ("…" if len(h["text"]) > 1000 else ""))
                    # Airtable record link
                    if AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME and h.get("airtable_id"):
                        url = airtable_record_url(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, h["airtable_id"])
                        st.markdown(f"🔗 [Airtable record]({url})")
                    # File links
                    if h.get("file_urls"):
                        for u in h["file_urls"][:5]:
                            st.markdown(f"📎 [File]({u})")
                    st.write("—")

            context_block = build_context(hits)
            with st.spinner("Thinking with context…"):
                answer = answer_with_gpt(user_q, context_block)
            st.chat_message("assistant").write(answer)
            st.session_state.history.append({"q": user_q, "a": answer, "hits": hits})
            log_qna_webhook(user_q, answer, hits)

# Footer debug
with st.sidebar:
    st.caption({
        "OPENAI_MODEL": OPENAI_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "WEAVIATE": getattr(weaviate, "__version__", "unknown"),
    })
