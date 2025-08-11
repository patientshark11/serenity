import os
import streamlit as st

# Optional: Detect Cloudflare header to skip password for Cloudflare Access
cf_email = os.environ.get("CF_BYPASS_EMAIL")  # We'll set this in Render

# If no Cloudflare Access email is detected, require password
# For Render direct URL, you'll have to enter this password
require_password = os.environ.get("RENDER_DIRECT_PASSWORD")

if not cf_email:
    st.write("## Secure Portal Login")
    password_input = st.text_input("Enter access password:", type="password")
    if password_input != require_password:
        st.stop()

import json
import time
from typing import List, Dict

import streamlit as st

# ----------------------------
# App & security configuration
# ----------------------------
st.set_page_config(page_title="Custody Documentation Assistant", page_icon="🗂️", layout="wide")

# Toggle in-app auth (you can keep this OFF if Cloudflare Access is protecting the app)
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes")

if AUTH_ENABLED:
    # Lightweight in-app password gate (kept simple since Cloudflare Access is the primary wall)
    # Set env var APP_PASSWORD to a strong value in Render.
    APP_PASSWORD = os.environ.get("APP_PASSWORD")
    if not APP_PASSWORD:
        st.error("Server misconfigured: APP_PASSWORD not set. Set it in Render → Environment.")
        st.stop()

    with st.sidebar:
        st.subheader("Login")
        pwd = st.text_input("Password", type="password")
        if st.button("Sign in"):
            if pwd == APP_PASSWORD:
                st.session_state["authed"] = True
            else:
                st.error("Incorrect password")
        if st.session_state.get("authed") is not True:
            st.stop()


# ----------------------------
# Helper placeholders
# ----------------------------
def run_ingest(files: List[Dict]):
    # TODO: wire up your existing automation endpoints here
    time.sleep(0.3)
    return {"status": "ok", "count": len(files)}

def run_analysis(query: str):
    # TODO: call your GPT/Render/Zapier/Make endpoints
    time.sleep(0.2)
    return {
        "summary_title": "Sample Result",
        "summary": "This is a placeholder. Plug in your analysis backend here.",
        "flags": ["Needs_Review"]
    }

def search_records(term: str):
    # TODO: connect to Airtable/DB/index
    time.sleep(0.2)
    return [
        {"title": "OFW Message — 2023-09-14", "id": "rec_abc123", "score": 0.89},
        {"title": "DSS Report — 2024-05-01", "id": "rec_def456", "score": 0.82},
    ]


# ----------------------------
# UI
# ----------------------------
st.title("🗂️ Custody Documentation Assistant")

tab_upload, tab_search, tab_analyze, tab_settings = st.tabs(
    ["Upload/Import", "Search", "Analyze", "Settings"]
)

with tab_upload:
    st.subheader("Upload files")
    up = st.file_uploader(
        "Drop PDFs, images, audio/video, or zip bundles",
        type=["pdf", "png", "jpg", "jpeg", "mov", "mp4", "mp3", "wav", "zip"],
        accept_multiple_files=True
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        dest = st.selectbox("Destination", ["Intake → Make", "Direct → Zapier Webhook", "Local (debug)"])
    with col_b:
        tag = st.text_input("Tag / Case ID (optional)", placeholder="e.g., ofw_batch_aug11")

    if st.button("Ingest"):
        if not up:
            st.warning("Please add at least one file.")
        else:
            files_meta = [{"name": f.name, "size": f.size, "type": f.type} for f in up]
            with st.spinner("Sending to pipeline…"):
                res = run_ingest(files_meta)
            st.success(f"Ingested {res['count']} files.")
            st.json(res)

with tab_search:
    st.subheader("Search records")
    q = st.text_input("Query", placeholder="e.g., late pickups in July; mention of daycare visits")
    if st.button("Search"):
        with st.spinner("Searching…"):
            results = search_records(q)
        for r in results:
            with st.expander(f"{r['title']}  •  score {r['score']:.2f}"):
                st.code(r["id"])

with tab_analyze:
    st.subheader("Ask a question about your dataset")
    prompt = st.text_area("Question", placeholder="Summarize all OFW messages from May showing scheduling conflicts…")
    if st.button("Run analysis"):
        if not prompt.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Analyzing…"):
                out = run_analysis(prompt)
            st.success("Done")
            st.write(f"### {out['summary_title']}")
            st.write(out["summary"])
            st.write("**Flags:** ", ", ".join(out.get("flags", [])))

with tab_settings:
    st.subheader("Settings")
    st.caption("These are environment-driven in production. Values below show what the server sees (non-secret).")
    st.write({
        "AUTH_ENABLED": AUTH_ENABLED,
        "ROBOT_BLOCK": os.environ.get("ROBOT_BLOCK", "true"),
        "APP_ENV": os.environ.get("APP_ENV", "production"),
    })
    st.info("Security Tip: Keep sensitive keys in Render Environment Variables, not hard-coded.")

import os, time, re
from uuid import uuid4

import streamlit as st
from pyairtable import Table
from openai import OpenAI

import weaviate
from weaviate.classes.config import Configure, Property, DataType

# ---------- config ----------
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
W_COLLECTION = "records"   # Weaviate collection name (change if you like)

# Which Airtable fields to concatenate into the searchable "text"
DEFAULT_TEXT_FIELDS = ["Title", "Notes", "Summary", "Body", "Description", "Content"]

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def make_text(fields: dict) -> str:
    # Build a single body of text from likely columns; adjust as needed
    parts = []
    for key in DEFAULT_TEXT_FIELDS:
        if key in fields and fields[key]:
            parts.append(f"{key}: {fields[key]}")
    # Fall back to joining all text-ish values if the above are empty
    if not parts:
        parts = [f"{k}: {v}" for k, v in fields.items() if isinstance(v, str)]
    return _clean("\n".join(parts))

def ensure_weaviate_collection(client: weaviate.WeaviateClient, name: str):
    existing = [c.name for c in client.collections.list_all()]
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
            Property(name="airtable_id", data_type=DataType.TEXT, index_searchable=True),
        ],
    )

def ingest_airtable_to_weaviate(limit: int | None = None):
    # --- clients ---
    at = Table(
        os.environ["AIRTABLE_API_KEY"],
        os.environ["AIRTABLE_BASE_ID"],
        os.environ["AIRTABLE_TABLE_NAME"],
    )
    wclient = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY", "")},  # optional passthrough
    )
    oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        coll = ensure_weaviate_collection(wclient, W_COLLECTION)

        # Pull rows from Airtable (handles pagination)
        count = 0
        batch = []
        for rec in at.iterate(page_size=50):
            fields = rec.get("fields", {})
            text = make_text(fields)
            if not text:
                continue

            title = fields.get("Title") or fields.get("Name") or "(untitled)"
            tags = fields.get("Tags") if isinstance(fields.get("Tags"), list) else []

            # Create embedding
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

            # Write to Weaviate in small batches
            if len(batch) >= 25:
                coll.data.insert_many(batch)
                count += len(batch)
                batch = []
                time.sleep(0.2)  # gentle pacing

            if limit and count >= limit:
                break

        if batch:
            coll.data.insert_many(batch)
            count += len(batch)

        return {"status": "ok", "ingested": count, "collection": W_COLLECTION}
    finally:
        wclient.close()

# ---------- Streamlit button to run it ----------
with st.sidebar:
    st.markdown("### Weaviate Loader")
    lim = st.number_input("Max rows to ingest (0 = all)", min_value=0, value=0, step=50)
    if st.button("Ingest Airtable → Weaviate"):
        n = None if lim == 0 else lim
        with st.spinner("Embedding and loading to Weaviate…"):
            res = ingest_airtable_to_weaviate(limit=n)
        st.success(f"Loaded {res['ingested']} records into `{res['collection']}`.")


