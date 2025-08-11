import os
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
