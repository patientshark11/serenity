import streamlit as st
import os
import backend
import openai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity Q&A",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
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
    logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
    st.image(logo_path, width=150)
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

    st.divider()

    st.header("Data Controls")
    if st.button("🔄 Sync Data from Airtable", use_container_width=True):
        if connect_to_backend():
            with st.spinner("Syncing data... This may take a moment."):
                try:
                    backend.ingest_airtable_to_weaviate(st.session_state.weaviate_client, st.session_state.openai_client)
                    st.toast("Data sync complete!", icon="✅")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

    st.header("Analysis Tools")
    if st.button("📅 Generate Timeline", use_container_width=True):
        if connect_to_backend():
            with st.spinner("Generating timeline..."):
                timeline = backend.generate_timeline(st.session_state.weaviate_client, st.session_state.openai_client)
                st.session_state.messages.append({"role": "assistant", "content": timeline, "summary": "Generated a timeline of events."})
                st.rerun()

    entity_name = st.text_input("Enter a name to summarize:", key="entity_input")
    if st.button("👤 Summarize Person", use_container_width=True):
        if connect_to_backend() and entity_name:
            with st.spinner(f"Summarizing {entity_name}..."):
                summary = backend.summarize_entity(entity_name, st.session_state.weaviate_client, st.session_state.openai_client)
                st.session_state.messages.append({"role": "assistant", "content": summary, "summary": f"Generated a summary for {entity_name}."})
                st.rerun()

    st.divider()
    report_type = st.selectbox(
        "Select a report to generate:",
        ["", "Conflict Report", "Legal Communication Summary"],
        key="report_select"
    )
    if st.button("📄 Generate Report", use_container_width=True):
        if connect_to_backend() and report_type:
            with st.spinner(f"Generating {report_type}..."):
                report = backend.generate_report(report_type, st.session_state.weaviate_client, st.session_state.openai_client)
                st.session_state.messages.append({"role": "assistant", "content": report, "summary": f"Generated a {report_type}."})
                st.rerun()

# --- Main Chat Interface ---
st.title("Custody Documentation Q&A")

# Display prior chat messages
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👧"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.caption("Sources:")
            for i, source_url in enumerate(message["sources"]):
                st.markdown(f"- [Source {i+1}]({source_url})")

# Check for connections before allowing chat
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"]):
    st.warning("Application is not fully configured. Please check environment variables.")
elif not connect_to_backend():
    st.warning("Could not connect to backend services. Please check your configuration and network.")
else:
    if prompt := st.chat_input("Ask a question about your documentation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👧"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching..."):
                response, sources, summary = backend.generative_search(prompt, st.session_state.weaviate_client, st.session_state.openai_client)
                st.write(response)
                if sources:
                    st.caption("Sources:")
                    for i, source_url in enumerate(sources):
                        st.markdown(f"- [Source {i+1}]({source_url})")

        message = {"role": "assistant", "content": response, "summary": summary}
        if sources:
            message["sources"] = sources
        st.session_state.messages.append(message)
