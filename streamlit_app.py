# ========= streamlit_app.py =========
import os
import re
import time
from uuid import uuid4
from typing import List, Dict

import streamlit as st
from openai import OpenAI
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from pyairtable import Table

# ---- MUST be first Streamlit call ----
st.set_page_config(page_title="Secure Portal", page_icon="🗂️", layout="wide")

# =========================
# Security / Login (simple)
# =========================
# Primary wall should be Cloudflare Access. Keep this ON only if you want an extra in-app password.
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")
if AUTH_ENABLED:
    APP_PASSWORD = os.environ.get("APP_PASSWORD")
    if not APP_PASSWORD:
        st.error("Server misconfigured: APP_PASSWORD not set in environment.")
        st.stop()
    with st.sidebar:
        st.subheader("Login")
        pwd = st.text_input("Password", type="password")
        if st.button("Sign in"):
            st.session_state["authed"] = (pwd == APP_PASSWORD)
        if st.session_state.get("authed") is not True:
            st.stop()
else:
    # Optional lightweight gate for direct access (e.g., onrender.com) while you iterate.
    DIRECT_PWD = os.environ.get("RENDER_DIRECT_PASSWORD", "")
    if DIRECT_PWD:
        entered = st.text_input("Secure Portal Login — Password", type="password")
        if entered != DIRECT_PWD:
            st.stop()

# ======================
# Config & Shared Clients
# ======================
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
W_COLLECTION = "records"  # Weaviate collection name

def get_clients():
    oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wclient = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    return oa, wclient

# =======================
# Weaviate ← Airtable Ingest
# =======================
DEFAULT_TEXT_FIELDS = ["Title", "Notes", "Summary", "Body", "Description", "Content"]

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
            Property(name="airtable_id", data_type=DataType.TEXT, index_searchable=True),
        ],
    )

def ingest_airtable_to_weaviate(limit: int | None = None):
    at = Table(
        os.environ["AIRTABLE_API_KEY"],
        os.environ["AIRTABLE_BASE_ID"],
        os.environ["AIRTABLE_TABLE_NAME"],
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

            emb = oa.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

            batch.append({
                "uuid": str(uuid4()),
                "properties": {
                    "title": title,
                    "source": os.environ["AIRTABLE_TABLE_NAME"],
                    "text": text,
                    "tags": tags,
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

# ===============
# Weaviate Search
# ===============
def weaviate_search(query: str, top_k: int = 5):
    oa, wclient = get_clients()
    try:
        coll = wclient.collections.get(W_COLLECTION)
        res = coll.query.near_text(
            query=query,
            limit=top_k,
            return_properties=["title", "text", "source", "tags", "airtable_id"],
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
                "score": 1 - (o.metadata.distance or 0),
            })
        return items
    finally:
        wclient.close()

def build_context(snippets, max_chars=4000):
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

# =========
#   UI
# =========
st.title("🗂️ Secure Portal")

tab_upload, tab_search, tab_analyze, tab_chat, tab_settings = st.tabs(
    ["Upload/Import", "Search", "Analyze", "Chat", "Settings"]
)

# Upload / Import
with tab_upload:
    st.subheader("Upload files")
    up = st.file_uploader(
        "Drop PDFs, images, audio/video, or zip bundles",
        type=["pdf", "png", "jpg", "jpeg", "mov", "mp4", "mp3", "wav", "zip"],
        accept_multiple_files=True,
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        dest = st.selectbox("Destination", ["Intake → Make", "Direct → Zapier Webhook", "Local (debug)"])
    with col_b:
        tag = st.text_input("Tag / Case ID (optional)", placeholder="e.g., ofw_batch_aug11")

    if st.button("Ingest (placeholder)"):
        if not up:
            st.warning("Please add at least one file.")
        else:
            files_meta = [{"name": f.name, "size": f.size, "type": f.type} for f in up]
            with st.spinner("Sending to pipeline…"):
                time.sleep(0.3)  # placeholder
            st.success(f"Queued {len(files_meta)} files for ingestion.")
            st.json(files_meta)

# Search (placeholder)
with tab_search:
    st.subheader("Search records (debug)")
    q = st.text_input("Query", placeholder="e.g., late pickups in July")
    if st.button("Search (Weaviate)", key="search_btn"):
        with st.spinner("Searching…"):
            hits = weaviate_search(q or "", top_k=5)
        if not hits:
            st.info("No matches yet.")
        else:
            for h in hits:
                with st.expander(f"{h['title']} • score {h['score']:.2f}"):
                    st.write(h["text"][:1200] + ("…" if len(h["text"]) > 1200 else ""))
                    st.caption(f"Tags: {', '.join(h['tags']) if h['tags'] else '—'} | Source: {h['source']} | Airtable ID: {h['airtable_id']}")

# Analyze (placeholder)
with tab_analyze:
    st.subheader("Ask a question about your dataset (placeholder)")
    prompt = st.text_area("Question", placeholder="Summarize OFW messages from May showing scheduling conflicts…")
    if st.button("Run analysis (placeholder)"):
        if not prompt.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Analyzing…"):
                time.sleep(0.2)
            st.success("Example only — wire your analysis backend here.")

# Chat (Weaviate-augmented)
with tab_chat:
    st.subheader("Chat with your records")
    q = st.text_input("Ask a question", placeholder="e.g., Summarize incidents involving late pickups in May 2024")
    k = st.slider("How many records to search", 1, 10, 5)

    if st.button("Ask", key="chat_btn"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Searching your Weaviate collection…"):
                hits = weaviate_search(q, top_k=k)

            if not hits:
                st.info("No matching records found. Try re-ingesting or broadening your question.")
            else:
                st.write("**Top matches (debug):**")
                for h in hits:
                    with st.expander(f"{h['title']} • score {h['score']:.2f}"):
                        st.write(h["text"][:1200] + ("…" if len(h["text"]) > 1200 else ""))
                        st.caption(f"Tags: {', '.join(h['tags']) if h['tags'] else '—'} | Source: {h['source']} | Airtable ID: {h['airtable_id']}")

                context_block = build_context(hits)
                with st.spinner("Thinking with context…"):
                    answer = answer_with_gpt(q, context_block)
                st.write("### Answer")
                st.write(answer)

# Settings / status
with tab_settings:
    st.subheader("Settings (server view)")
    st.write({
        "APP_ENV": os.environ.get("APP_ENV", "production"),
        "AUTH_ENABLED": AUTH_ENABLED,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENAI_EMBED_MODEL": EMBED_MODEL,
        "WEAVIATE_URL_SET": bool(os.environ.get("WEAVIATE_URL")),
        "AIRTABLE_BASE_ID": os.environ.get("AIRTABLE_BASE_ID", "—"),
        "AIRTABLE_TABLE_NAME": os.environ.get("AIRTABLE_TABLE_NAME", "—"),
    })
    st.info("Keep secrets in Render → Environment. Rotate keys if shared.")

# Sidebar: Ingest button
with st.sidebar:
    st.markdown("### Weaviate Loader")
    lim = st.number_input("Max rows to ingest (0 = all)", min_value=0, value=0, step=50)
    if st.button("Ingest Airtable → Weaviate"):
        n = None if lim == 0 else lim
        with st.spinner("Embedding and loading to Weaviate…"):
            res = ingest_airtable_to_weaviate(limit=n)
        st.success(f"Loaded {res['ingested']} records into `{res['collection']}`.")
# ========= end file =========

