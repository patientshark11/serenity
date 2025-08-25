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
                st.markdown(f"**Answer:** {msg['content']}")
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

# --- Main Chat Interface ---
st.title("Custody Documentation Q&A")

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption("Sources:")
            for source in message["sources"]:
                st.write(source)

# Check for connections before allowing chat
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"]):
    st.warning("Application is not fully configured. Please check environment variables.")
elif not connect_to_backend():
    st.warning("Could not connect to backend services. Please check your configuration and network.")
else:
    if prompt := st.chat_input("Ask a question about your documentation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response, sources = backend.generative_search(prompt, st.session_state.weaviate_client, st.session_state.openai_client)
                st.write(response)
                if sources:
                    st.caption("Sources:")
                    for source in sources:
                        url = f"https://airtable.com/{os.environ['AIRTABLE_BASE_ID']}/{os.environ['AIRTABLE_TABLE_NAME']}/{source}"
                        st.markdown(f"- [{source}]({url})")

        message = {"role": "assistant", "content": response}
        if sources:
            message["sources"] = [f"https://airtable.com/{os.environ['AIRTABLE_BASE_ID']}/{os.environ['AIRTABLE_TABLE_NAME']}/{s}" for s in sources]
        st.session_state.messages.append(message)
