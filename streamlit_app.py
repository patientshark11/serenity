import streamlit as st
import os
import weaviate
import openai
from pyairtable import Table
import uuid

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity - Custody Q&A",
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


# --- 3. CHECK SECRETS & INITIALIZE CLIENTS ---
def check_secrets():
    required_keys = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY", "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"]
    if not all(os.environ.get(key) for key in required_keys):
        st.error("ERROR: One or more required environment variables are missing. Please check your Render dashboard.")
        st.stop()
check_secrets()

@st.cache_resource
def get_weaviate_client():
    return weaviate.Client(
        url=os.environ.get("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")),
        additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")}
    )

@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

try:
    weaviate_client = get_weaviate_client()
    openai_client = get_openai_client()
except Exception as e:
    st.error(f"Failed to initialize API clients. Error: {e}")
    st.stop()


# --- 4. CORE APPLICATION LOGIC ---
def get_embedding(text):
   model_name = os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
   text = text.replace("\n", " ")
   response = openai_client.embeddings.create(input=[text], model=model_name)
   return response.data[0].embedding

@st.cache_data(ttl=3600)
def ingest_airtable_to_weaviate():
    class_name = "CustodyDocs"
    if weaviate_client.schema.exists(class_name):
        weaviate_client.schema.delete_class(class_name)
    class_obj = {"class": class_name, "vectorizer": "none", "properties": [{"name": "question", "dataType": ["text"]}, {"name": "answer", "dataType": ["text"]}]}
    weaviate_client.schema.create_class(class_obj)
    table = Table(os.environ.get("AIRTABLE_API_KEY"), os.environ.get("AIRTABLE_BASE_ID"), os.environ.get("AIRTABLE_TABLE_NAME"))
    records = table.all()
    with weaviate_client.batch as batch:
        batch.batch_size = 100
        for item in records:
            question_text = item.get("fields", {}).get("Question", "")
            if question_text:
                emb = get_embedding(question_text)
                data_obj = {"question": question_text, "answer": item.get("fields", {}).get("Answer", "")}
                batch.add_data_object(data_object=data_obj, class_name=class_name, uuid=str(uuid.uuid4()), vector=emb)
    st.toast("Data sync complete!", icon="✅")

def perform_search(query: str):
    query_vector = get_embedding(query)
    response = (weaviate_client.query.get("CustodyDocs", ["question", "answer"])
                .with_near_vector({"vector": query_vector}).with_limit(1).do())
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    return results[0]["answer"] if results else "I couldn't find a relevant answer in the documentation. Please try rephrasing your question."

# --- 5. STREAMLIT UI LAYOUT ---
with st.sidebar:
    logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
    st.image(logo_path, width=150)
    st.markdown("## Chat History")
    
    # UI FIX 1: History is now always visible, no expander
    if "history" in st.session_state and st.session_state.history:
        for q, a in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.divider() # Adds a nice separator
    else:
        st.info("Your chat history will appear here.")
    
    if st.button("🔄 Sync Data from Airtable"):
        with st.spinner("Ingesting data..."):
            ingest_airtable_to_weaviate()

with st.container(border=True):
    st.title("⚖️ Custody Documentation Q&A")
    st.markdown("Private, authenticated workspace for your case records.")
st.markdown("<br>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # UI FIX 2: Add a welcome message to fill the empty space
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm here to help with your custody documentation. Ask me a question to get started."})

# Display chat messages
for message in st.session_state.messages:
    # UI FIX 3: Use more professional avatars
    avatar_icon = "👤" if message["role"] == "user" else "📚"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching..."):
            response = perform_search(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    if 'history' not in st.session_state: st.session_state.history = []
    st.session_state.history.append((prompt, response))
    st.rerun()
