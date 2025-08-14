import streamlit as st
import os
import weaviate
import openai
from pyairtable import Table
import uuid
from weaviate.exceptions import WeaviateConnectionError
from openai import APIError as OpenAI_APIError
from requests.exceptions import HTTPError as AirtableHTTPError

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

# --- 3. BACKEND LOGIC ---
def connect_to_services():
    if "clients_connected" in st.session_state: return True
    try:
        st.session_state.weaviate_client = weaviate.Client(url=os.environ.get("WEAVIATE_URL"), auth_client_secret=weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")), additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")})
        st.session_state.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        st.session_state.clients_connected = True
        return True
    except Exception as e:
        st.error(f"Failed to connect to backend services: {e}")
        return False

def get_embedding(text):
   model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
   response = st.session_state.openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
   return response.data[0].embedding

def ingest_airtable_to_weaviate():
    class_name = "CustodyDocs"
    client = st.session_state.weaviate_client
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)
    class_obj = {"class": class_name, "vectorizer": "none", "properties": [{"name": "question", "dataType": ["text"]}, {"name": "answer", "dataType": ["text"]}]}
    client.schema.create_class(class_obj)
    table = Table(os.environ.get("AIRTABLE_API_KEY"), os.environ.get("AIRTABLE_BASE_ID"), os.environ.get("AIRTABLE_TABLE_NAME"))
    records = table.all()
    with client.batch as batch:
        batch.batch_size = 100
        for item in records:
            question_text = item.get("fields", {}).get("Question", "")
            if question_text:
                emb = get_embedding(question_text)
                data_obj = {"question": question_text, "answer": item.get("fields", {}).get("Answer", "")}
                batch.add_data_object(data_object=data_obj, class_name=class_name, uuid=str(uuid.uuid4()), vector=emb)
    st.toast("Data sync complete!", icon="✅")

def perform_search(query: str):
    # **LOGIC FIX:** Check if the database is empty before searching.
    try:
        response_count = st.session_state.weaviate_client.query.aggregate("CustodyDocs").with_meta_count().do()
        count = response_count["data"]["Aggregate"]["CustodyDocs"][0]["meta"]["count"]
        if count == 0:
            return "The database is empty. Please click 'Sync Data from Airtable' in the sidebar to add your documentation."
    except Exception:
         return "Could not connect to the database. Please ensure it's running and accessible."

    query_vector = get_embedding(query)
    response = (st.session_state.weaviate_client.query.get("CustodyDocs", ["question", "answer"])
                .with_near_vector({"vector": query_vector}).with_limit(1).do())
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    return results[0]["answer"] if results else "I couldn't find a relevant answer in the documentation."

# --- 4. UI LAYOUT (ChatGPT STYLE) ---

# --- Sidebar ---
with st.sidebar:
    logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
    st.image(logo_path, width=150)
    st.title("Chat History")

   if "messages" in st.session_state and st.session_state.messages:
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
        with st.spinner("Syncing data... This may take a moment."):
            try:
                ingest_airtable_to_weaviate()
            except Exception as e:
                st.error(f"Sync failed: {e}")


# --- Main Chat Interface ---
st.title("Custody Documentation Q&A")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Check for connections before allowing chat
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY"]) or not connect_to_services():
    st.warning("Application is not fully configured. Please check environment variables and ensure all services are reachable.")
else:
    # The modern, pinned-to-bottom chat input
    if prompt := st.chat_input("Ask a question about your documentation..."):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                response = perform_search(prompt)
                st.write(response)
        
        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})

