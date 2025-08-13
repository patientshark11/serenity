import streamlit as st
import weaviate
import openai
from pyairtable import Table
import uuid

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity - Custody Q&A",
    page_icon="⚖️", # You can also use st.secrets.get("APP_LOGO_PATH", "⚖️")
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
    # Checking for the core secrets needed to run the app
    required_secrets = [
        "WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY",
        "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"
    ]
    if not all(secret in st.secrets for secret in required_secrets):
        st.error("ERROR: Missing one or more required environment variables. Please check your Render dashboard.")
        st.stop()

# Run the check at the start
check_secrets()

@st.cache_resource
def get_weaviate_client():
    """Initializes and returns a Weaviate client."""
    return weaviate.Client(
        url=st.secrets["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]}
    )

@st.cache_resource
def get_openai_client():
    """Initializes the OpenAI client."""
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize clients after secrets check
try:
    weaviate_client = get_weaviate_client()
    openai_client = get_openai_client()
except Exception as e:
    st.error(f"Failed to initialize API clients. Please double-check your credentials. Error: {e}")
    st.stop()


# --- 4. CORE APPLICATION LOGIC ---
def get_embedding(text):
   """Generates embedding for a given text using OpenAI."""
   # **IMPROVEMENT:** Using the model from your environment variables
   model_name = st.secrets.get("OPENAI_MODEL", "text-embedding-3-small")
   text = text.replace("\n", " ")
   response = openai_client.embeddings.create(input=[text], model=model_name)
   return response.data[0].embedding

@st.cache_data(ttl=3600)
def ingest_airtable_to_weaviate():
    """Fetches data from Airtable, generates embeddings, and ingests into Weaviate."""
    # ... (rest of the function is the same, no changes needed here) ...
    class_name = "CustodyDocs"
    if weaviate_client.schema.exists(class_name):
        weaviate_client.schema.delete_class(class_name)
    class_obj = {"class": class_name, "vectorizer": "none", "properties": [
        {"name": "question", "dataType": ["text"]}, {"name": "answer", "dataType": ["text"]},
    ]}
    weaviate_client.schema.create_class(class_obj)
    table = Table(st.secrets["AIRTABLE_API_KEY"], st.secrets["AIRTABLE_BASE_ID"], st.secrets["AIRTABLE_TABLE_NAME"])
    records = table.all()
    with weaviate_client.batch as batch:
        batch.batch_size = 100
        for item in records:
            question_text = item.get("fields", {}).get("Question", "")
            if question_text:
                emb = get_embedding(question_text)
                data_obj = {"question": question_text, "answer": item.get("fields", {}).get("Answer", "")}
                batch.add_data_object(data_object=data_obj, class_name=class_name, uuid=str(uuid.uuid4()), vector=emb)
    st.toast("Data ingestion complete!", icon="✅")

def perform_search(query: str):
    """Performs a vector search in Weaviate."""
    query_vector = get_embedding(query)
    response = (weaviate_client.query.get("CustodyDocs", ["question", "answer"])
                .with_near_vector({"vector": query_vector}).with_limit(1).do())
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    return results[0]["answer"] if results else "I couldn't find a relevant answer. Please rephrase."

# --- 5. STREAMLIT UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    # **IMPROVEMENT:** Using the logo path from your environment variables
    logo_path = st.secrets.get("APP_LOGO_PATH", "logo.png") # Defaults to logo.png if not set
    try:
        st.image(logo_path, width=150)
    except Exception:
        st.warning(f"Could not find logo at path: {logo_path}")

    st.markdown("## Chat History")
    if "history" in st.session_state and st.session_state.history:
        for q, a in st.session_state.history:
            with st.expander(f"Q: {q[:30]}..."):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
    else:
        st.info("Your chat history will appear here.")
    if st.button("🔄 Sync Data from Airtable"):
        with st.spinner("Ingesting data..."):
            ingest_airtable_to_weaviate()

# --- Main Page ---
with st.container(border=True):
    st.title("Custody Documentation Q&A")
    st.markdown("Private, authenticated workspace for your case records.")
st.markdown("<br>", unsafe_allow_html=True)

# Initialize and display chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            response = perform_search(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append((prompt, response))
    st.rerun()
