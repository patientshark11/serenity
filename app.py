import streamlit as st
import os
import sys
import backend
import openai
import uuid # Import uuid for unique keys
import subprocess
import sys
import hashlib
from collections.abc import Mapping

from sync_reports import _extract_text_fragment

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
    if "person_select_sidebar" not in st.session_state:
        st.session_state.person_select_sidebar = ""
    if "report_select_sidebar" not in st.session_state:
        st.session_state.report_select_sidebar = ""
    if "last_prompt_hash" not in st.session_state:
        st.session_state.last_prompt_hash = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

get_or_create_session_state()

# --- 4. BACKEND INITIALIZATION ---
def connect_to_backend():
    try:
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

    def _on_person_select_change():
        selected_person = st.session_state.get("person_select_sidebar", "")
        if selected_person:
            st.session_state.entity_name_to_summarize = selected_person
            st.session_state.run_summary = True
        else:
            st.session_state.run_summary = False
            st.session_state.entity_name_to_summarize = ""

    def _on_report_select_change():
        selected_report = st.session_state.get("report_select_sidebar", "")
        if selected_report:
            st.session_state.report_type_to_generate = selected_report
            st.session_state.run_report = True
        else:
            st.session_state.run_report = False
            st.session_state.report_type_to_generate = ""

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
    person_reset_flag = "reset_person_select_sidebar"
    if st.session_state.get(person_reset_flag):
        st.session_state.person_select_sidebar = ""
        del st.session_state[person_reset_flag]
    st.selectbox(
        "Select a person to summarize:",
        key_people,
        key="person_select_sidebar",
        on_change=_on_person_select_change,
        disabled=model_is_unavailable,
    )

    report_reset_flag = "reset_report_select_sidebar"
    if st.session_state.get(report_reset_flag):
        st.session_state.report_select_sidebar = ""
        del st.session_state[report_reset_flag]
    st.selectbox(
        "Select a report:",
        ["", "Conflict Report", "Legal Communication Summary"],
        key="report_select_sidebar",
        on_change=_on_report_select_change,
        disabled=model_is_unavailable,
    )

    st.divider()
    st.caption("Reports are generated daily. Use this to force an immediate update.")
    if st.button("🔄 Re-Sync & Generate Reports", use_container_width=True):
        if connect_to_backend():
            with st.spinner("Syncing data and generating new reports... (this may take several minutes)"):
                try:
                    # IMPORTANT: This should call the NEW sync script
                    result = subprocess.run([sys.executable, "-u", "sync_reports.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.toast(result.stdout or "Data sync and report generation complete!", icon="✅")
                    else:
                        st.error(result.stderr or "Sync failed")
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
            pdf_bytes = message.get("pdf")
            if pdf_bytes is None:
                pdf_bytes = backend.create_pdf(
                    message["content"],
                    summary=message.get("summary"),
                    sources=message.get("sources"),
                )
                message["pdf"] = pdf_bytes
            # Use the unique message ID for the key and remove the incorrect bytes() wrapper
            st.download_button(
                "Export as PDF",
                pdf_bytes,
                f"{message.get('summary', 'response')}.pdf",
                "application/pdf",
                key=f"pdf_{message.get('id', str(uuid.uuid4()))}"
            )

# Check for connections before allowing chat
required_env_vars = [
    "WEAVIATE_URL",
    "OPENAI_API_KEY",
    "AIRTABLE_API_KEY",
    "AIRTABLE_BASE_ID",
    "AIRTABLE_TABLE_NAME",
]
if not all(os.environ.get(key) for key in required_env_vars):
    st.warning("Application is not fully configured. Please check environment variables.")
elif not connect_to_backend():
    st.warning("Could not connect to backend services. Please check your configuration and network.")
else:
    user_avatar = "https://ui-avatars.com/api/?name=User&background=F0F2F6&color=0F172A"
    assistant_avatar = "https://ui-avatars.com/api/?name=Assistant&background=5865F2&color=FFF"
    model_is_unavailable = st.session_state.settings["openai_model"] == "GPT-5 (Not Yet Available)"

    # Analysis tool action logic
    def display_fetched_report(sanitized_report_name, summary_text, *, reset_widget_key=None):
        """Helper to fetch, display, and store a pre-generated report."""
        if not connect_to_backend(): return

        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner(f"Fetching {summary_text}..."):
                report_content = backend.fetch_report(sanitized_report_name)
                st.write(report_content)

        pdf_bytes = backend.create_pdf(report_content, summary=summary_text)
        st.session_state.messages.append({
            "role": "assistant",
            "content": report_content,
            "summary": summary_text,
            "id": str(uuid.uuid4()),
            "pdf": pdf_bytes,
        })
        if reset_widget_key:
            st.session_state[f"reset_{reset_widget_key}"] = True
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
        display_fetched_report(sanitized_name, f"Summary for {entity_name}", reset_widget_key="person_select_sidebar")

    if st.session_state.get("run_report"):
        report_type = st.session_state.report_type_to_generate
        st.session_state.run_report = False
        sanitized_name = backend.sanitize_name(report_type)
        display_fetched_report(sanitized_name, f"{report_type}", reset_widget_key="report_select_sidebar")

    # Main Q&A chat input
    if prompt := st.chat_input("Ask a question about your documentation...", disabled=model_is_unavailable):
        st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
        with st.chat_message("user", avatar=user_avatar):
            st.write(prompt)

        prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
        cached_payload = None
        if st.session_state.last_prompt_hash == prompt_hash:
            cached_payload = st.session_state.last_response

        response_payload = None
        with st.chat_message("assistant", avatar=assistant_avatar):
            if cached_payload:
                response_payload = cached_payload.copy()
                full_response = response_payload.get("content", "")
                st.write(full_response)
                st.caption("Using cached response.")
            else:
                with st.spinner("Thinking..."):
                    # Call the new HyDE search function from the backend
                    response, sources, summary = backend.generative_search(
                        prompt,
                        st.session_state.weaviate_client,
                        st.session_state.openai_client,
                        model=st.session_state.settings["openai_model"]
                    )
                    sources = list(sources or [])

                    if isinstance(response, str):
                        full_response = response
                        st.write(full_response)
                    else:
                        streamed_chunks = []

                        def _get_value(obj, key, default=None):
                            if obj is None:
                                return default
                            if isinstance(obj, Mapping):
                                return obj.get(key, default)
                            return getattr(obj, key, default)

                        def _stream_and_cache():
                            for chunk in response:
                                text_fragment = ""

                                try:
                                    choices = _get_value(chunk, "choices") or []
                                    first_choice = choices[0] if choices else None
                                    if first_choice is not None:
                                        delta = _get_value(first_choice, "delta")
                                        if delta is not None:
                                            text_fragment = _extract_text_fragment(delta) or ""
                                        else:
                                            message = _get_value(first_choice, "message")
                                            if message is not None:
                                                text_fragment = _extract_text_fragment(message) or ""

                                    if not text_fragment:
                                        text_fragment = _extract_text_fragment(chunk) or ""
                                except Exception:
                                    text_fragment = _extract_text_fragment(chunk) or ""

                                streamed_chunks.append(text_fragment)
                                yield text_fragment

                        st.write_stream(_stream_and_cache())
                        full_response = "".join(streamed_chunks)

                response_payload = {
                    "content": full_response,
                    "summary": summary,
                    "sources": sources,
                }

            sources_list = (response_payload.get("sources") or []) if response_payload else []
            if sources_list:
                st.caption("Sources:")
                for source in sources_list:
                    st.markdown(f"- [{source['title']}]({source['url']})")

        if response_payload is None:
            response_payload = {
                "content": cached_payload.get("content", "") if cached_payload else "",
                "summary": cached_payload.get("summary") if cached_payload else None,
                "sources": (cached_payload.get("sources") if cached_payload else []) or [],
                "pdf": cached_payload.get("pdf") if cached_payload else None,
            }

        pdf_bytes = response_payload.get("pdf")
        if pdf_bytes is None:
            pdf_bytes = backend.create_pdf(
                response_payload.get("content", ""),
                summary=response_payload.get("summary"),
                sources=response_payload.get("sources"),
            )
            response_payload["pdf"] = pdf_bytes

        last_response_payload = response_payload.copy()
        if last_response_payload.get("sources") is not None:
            last_response_payload["sources"] = list(last_response_payload["sources"])

        st.session_state.last_prompt_hash = prompt_hash
        st.session_state.last_response = last_response_payload

        message_sources = last_response_payload.get("sources")
        if message_sources is not None:
            message_sources = list(message_sources)

        message = {
            "role": "assistant",
            "content": response_payload.get("content", ""),
            "summary": response_payload.get("summary"),
            "sources": message_sources,
            "id": str(uuid.uuid4()),
            "pdf": pdf_bytes,
        }
        st.session_state.messages.append(message)
        st.rerun()
