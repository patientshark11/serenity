import streamlit as st
import os
import backend
import openai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity Q&A",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 3. STATE MANAGEMENT ---
def get_or_create_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "weaviate_client" not in st.session_state:
        st.session_state.weaviate_client = None
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "chunk_size": 2000,
            "openai_model": "gpt-4o",
            "chunk_limit": 5
        }

get_or_create_session_state()

# --- 4. BACKEND INITIALIZATION ---
def connect_to_backend():
    try:
        if st.session_state.weaviate_client is None:
            st.session_state.weaviate_client = backend.connect_to_weaviate()
        if st.session_state.openai_client is None:
            st.session_state.openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return True
    except Exception as e:
        st.error(f"Failed to connect to backend services: {e}")
        return False

# --- 5. UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
        st.image(logo_path, width=100)

    with st.expander("⚙️ Settings"):
        st.session_state.settings["chunk_size"] = st.slider(
            "Context Chunk Size", min_value=500, max_value=4000,
            value=st.session_state.settings["chunk_size"], step=100,
            help="Controls text chunk size. Smaller is faster, larger has more context."
        )
        models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "GPT-5 (Not Yet Available)"]
        if st.session_state.settings["openai_model"] not in models:
            st.session_state.settings["openai_model"] = models[0]
        st.session_state.settings["openai_model"] = st.selectbox(
            "OpenAI Model", models, index=models.index(st.session_state.settings["openai_model"]),
            help="Select the AI model. `gpt-4o` is the latest and most capable."
        )
        st.session_state.settings["chunk_limit"] = st.slider(
            "Number of Sources to Retrieve", min_value=1, max_value=10,
            value=st.session_state.settings.get("chunk_limit", 5), step=1,
            help="Number of text chunks to use as context."
        )

    model_is_unavailable = st.session_state.settings["openai_model"] == "GPT-5 (Not Yet Available)"
    if model_is_unavailable:
        st.warning("GPT-5 is not yet released. Please select another model.")

    st.divider()
    st.header("Analysis Tools")

    if st.button("📅 Generate Timeline", use_container_width=True, disabled=model_is_unavailable):
        if connect_to_backend():
            with st.spinner("Generating timeline... (this may take a moment)"):
                response_stream = backend.generate_timeline(st.session_state.weaviate_client, st.session_state.openai_client, model=st.session_state.settings["openai_model"])
                full_response = "".join(c for c in response_stream) if not isinstance(response_stream, str) else response_stream
                st.session_state.messages.append({"role": "assistant", "content": full_response, "summary": "Generated a timeline of events."})
                st.rerun()

    entity_name = st.text_input("Enter a name to summarize:", key="entity_input_sidebar", disabled=model_is_unavailable)
    if st.button("👤 Summarize Person", use_container_width=True, disabled=model_is_unavailable):
        if connect_to_backend() and entity_name:
            with st.spinner(f"Summarizing {entity_name}... (this may take a moment)"):
                response_stream = backend.summarize_entity(entity_name, st.session_state.weaviate_client, st.session_state.openai_client, model=st.session_state.settings["openai_model"])
                full_response = "".join(c for c in response_stream) if not isinstance(response_stream, str) else response_stream
                st.session_state.messages.append({"role": "assistant", "content": full_response, "summary": f"Generated a summary for {entity_name}."})
                st.rerun()

    report_type = st.selectbox("Select a report:", ["", "Conflict Report", "Legal Communication Summary"], key="report_select_sidebar", disabled=model_is_unavailable)
    if st.button("📄 Generate Report", use_container_width=True, disabled=model_is_unavailable):
        if connect_to_backend() and report_type:
            with st.spinner(f"Generating {report_type}... (this may take a moment)"):
                response_stream = backend.generate_report(report_type, st.session_state.weaviate_client, st.session_state.openai_client, model=st.session_state.settings["openai_model"])
                full_response = "".join(c for c in response_stream) if not isinstance(response_stream, str) else response_stream
                st.session_state.messages.append({"role": "assistant", "content": full_response, "summary": f"Generated a {report_type}."})
                st.rerun()

    st.divider()
    st.caption("For best results, re-sync data after code updates.")
    if st.button("🔄 Force Data Re-Sync", use_container_width=True):
        if connect_to_backend():
            with st.spinner("Syncing data..."):
                try:
                    backend.ingest_airtable_to_weaviate(st.session_state.weaviate_client, st.session_state.openai_client, chunk_size=st.session_state.settings["chunk_size"])
                    st.toast("Data sync complete!", icon="✅")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

    st.divider()
    st.title("Chat History")
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                summary = msg.get("summary", "Summary not available.")
                st.markdown(f"**Q&A:** {summary}")
                st.divider()
    else:
        st.info("Your history will appear here.")

# --- Main Chat Interface ---
st.title("Custody Documentation Q&A")

# Display prior chat messages
for message in st.session_state.messages:
    user_avatar = "https://ui-avatars.com/api/?name=Question&background=F0F2F6&color=0F172A"
    assistant_avatar = "https://ui-avatars.com/api/?name=Answer&background=5865F2&color=FFF"
    avatar = assistant_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.caption("Sources:")
            for source in message["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")

        if message["role"] == "assistant":
            pdf_bytes = backend.create_pdf(message["content"])
            st.download_button("Export as PDF", bytes(pdf_bytes), f"{message.get('summary', 'response')}.pdf", "application/pdf", key=f"pdf_{message['content'][:20]}")

# Check for connections before allowing chat
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"]):
    st.warning("Application is not fully configured. Please check environment variables.")
elif not connect_to_backend():
    st.warning("Could not connect to backend services. Please check your configuration and network.")
else:
    user_avatar = "https://ui-avatars.com/api/?name=Question&background=F0F2F6&color=0F172A"
    assistant_avatar = "https://ui-avatars.com/api/?name=Answer&background=5865F2&color=FFF"
    model_is_unavailable = st.session_state.settings["openai_model"] == "GPT-5 (Not Yet Available)"

    if prompt := st.chat_input("Ask a question about your documentation...", disabled=model_is_unavailable):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_avatar):
            st.write(prompt)

        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Searching..."):
                response, sources, summary = backend.generative_search(prompt, st.session_state.weaviate_client, st.session_state.openai_client, model=st.session_state.settings["openai_model"], limit=st.session_state.settings["chunk_limit"])

                if isinstance(response, str):
                    full_response = response
                    st.write(full_response)
                else:
                    full_response = st.write_stream(response)

                pdf_bytes = backend.create_pdf(full_response)
                st.download_button("Export as PDF", bytes(pdf_bytes), "answer.pdf", "application/pdf")

                if sources:
                    st.caption("Sources:")
                    for source in sources:
                        st.markdown(f"- [{source['title']}]({source['url']})")

        message = {"role": "assistant", "content": full_response, "summary": summary}
        if sources:
            message["sources"] = sources
        st.session_state.messages.append(message)
