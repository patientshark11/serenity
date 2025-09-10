import streamlit as st
import os
import backend
import openai
import uuid # Import uuid for unique keys

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity Q&A",
    page_icon="⚖️",
    layout="centered", # Centered layout with a clean sidebar feels modern
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
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "openai_model": "gpt-4o"
        }
    if "run_timeline" not in st.session_state:
        st.session_state.run_timeline = False
    if "run_summary" not in st.session_state:
        st.session_state.run_summary = False
    if "run_report" not in st.session_state:
        st.session_state.run_report = False

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
        st.image(logo_path, width=150)

    with st.expander("⚙️ Settings"):
        env_model = os.environ.get("OPENAI_COMPLETION_MODEL")
        if env_model:
            st.session_state.settings["openai_model"] = env_model
            st.info(f"Using model from environment: **{env_model}**")
        else:
            models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "GPT-5 (Not Yet Available)"]
            if st.session_state.settings["openai_model"] not in models:
                st.session_state.settings["openai_model"] = models[0]
            st.session_state.settings["openai_model"] = st.selectbox(
                "OpenAI Model", models, index=models.index(st.session_state.settings["openai_model"]),
                help="Select the AI model. `gpt-4o` is the latest and most capable."
            )

    model_is_unavailable = st.session_state.settings["openai_model"] == "GPT-5 (Not Yet Available)"
    if model_is_unavailable:
        st.warning("GPT-5 is not yet released. Please select another model.")

    st.divider()
    st.header("Pre-Generated Reports")

    if st.button("📅 View Timeline", use_container_width=True, disabled=model_is_unavailable):
        st.session_state.run_timeline = True

    key_people = ["", "Kim", "Diego", "Kim's family/friends", "YWCA Staff", "Heather Ulrich", "DSS/Youth Villages", "Diego's mom"]
    person_to_summarize = st.selectbox("Select a person to summarize:", key_people, disabled=model_is_unavailable)
    if person_to_summarize:
        st.session_state.entity_name_to_summarize = person_to_summarize
        st.session_state.run_summary = True

    report_type_input = st.selectbox("Select a report:", ["", "Conflict Report", "Legal Communication Summary"], key="report_select_sidebar", disabled=model_is_unavailable)
    if report_type_input:
        st.session_state.report_type_to_generate = report_type_input
        st.session_state.run_report = True

    st.divider()
    st.caption("Reports are generated daily. Use this to force an immediate update.")
    if st.button("🔄 Re-Sync & Generate Reports", use_container_width=True):
        if connect_to_backend():
            with st.spinner("Syncing data and generating new reports... (this may take several minutes)"):
                try:
                    # IMPORTANT: This should call the NEW sync script
                    os.system("python sync_reports.py")
                    st.toast("Data sync and report generation complete!", icon="✅")
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
    user_avatar = "https://ui-avatars.com/api/?name=User&background=F0F2F6&color=0F172A"
    assistant_avatar = "https://ui-avatars.com/api/?name=Assistant&background=5865F2&color=FFF"
    avatar = assistant_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.caption("Sources:")
            for source in message["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")

        if message["role"] == "assistant":
            pdf_bytes = backend.create_pdf(message["content"], summary=message.get("summary"), sources=message.get("sources"))
            # Use the unique message ID for the key and remove the incorrect bytes() wrapper
            st.download_button("Export as PDF", pdf_bytes, f"{message.get('summary', 'response')}.pdf", "application/pdf", key=f"pdf_{message.get('id', str(uuid.uuid4()))}")

# Check for connections before allowing chat
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"]):
    st.warning("Application is not fully configured. Please check environment variables.")
elif not connect_to_backend():
    st.warning("Could not connect to backend services. Please check your configuration and network.")
else:
    user_avatar = "https://ui-avatars.com/api/?name=User&background=F0F2F6&color=0F172A"
    assistant_avatar = "https://ui-avatars.com/api/?name=Assistant&background=5865F2&color=FFF"
    model_is_unavailable = st.session_state.settings["openai_model"] == "GPT-5 (Not Yet Available)"

    # Analysis tool action logic
    def display_fetched_report(sanitized_report_name, summary_text):
        """Helper to fetch, display, and store a pre-generated report."""
        if not connect_to_backend(): return

        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner(f"Fetching {summary_text}..."):
                report_content = backend.fetch_report(sanitized_report_name)
                st.write(report_content)

        st.session_state.messages.append({"role": "assistant", "content": report_content, "summary": summary_text, "id": str(uuid.uuid4())})
        st.rerun()

    if st.session_state.get("run_timeline"):
        st.session_state.run_timeline = False
        sanitized_name = backend.sanitize_name("Timeline")
        display_fetched_report(sanitized_name, "Timeline of events")

    if st.session_state.get("run_summary"):
        entity_name = st.session_state.entity_name_to_summarize
        st.session_state.run_summary = False
        report_name = f"Summary for {entity_name}"
        sanitized_name = backend.sanitize_name(report_name)
        display_fetched_report(sanitized_name, f"Summary for {entity_name}")

    if st.session_state.get("run_report"):
        report_type = st.session_state.report_type_to_generate
        st.session_state.run_report = False
        sanitized_name = backend.sanitize_name(report_type)
        display_fetched_report(sanitized_name, f"{report_type}")

    # Main Q&A chat input
    if prompt := st.chat_input("Ask a question about your documentation...", disabled=model_is_unavailable):
        st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
        with st.chat_message("user", avatar=user_avatar):
            st.write(prompt)

        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Thinking..."):
                # Call the new HyDE search function from the backend
                response, sources, summary = backend.generative_search(
                    prompt,
                    st.session_state.weaviate_client,
                    st.session_state.openai_client,
                    model=st.session_state.settings["openai_model"]
                )

                if isinstance(response, str):
                    full_response = response
                    st.write(full_response)
                else:
                    full_response = st.write_stream(response)

                if sources:
                    st.caption("Sources:")
                    for source in sources:
                        st.markdown(f"- [{source['title']}]({source['url']})")

        message = {"role": "assistant", "content": full_response, "summary": summary, "sources": sources, "id": str(uuid.uuid4())}
        st.session_state.messages.append(message)
        st.rerun()
