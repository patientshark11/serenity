import streamlit as st
import weaviate
import openai
from pyairtable import Table # Changed from 'airtable'
import uuid
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity - Custody Q&A",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. CUSTOM CSS FOR A PROFESSIONAL LOOK ---
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .custom-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    .st-emotion-cache-16txtl3 { /* Sidebar selector */
        padding: 1rem;
    }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- 3. CACHING & API CLIENTS ---

@st.cache_resource
def get_weaviate_client():
    """Initializes and returns a Weaviate client."""
    try:
        client = weaviate.Client(
            url=st.secrets["WEAVIATE_URL"],
            auth_client_secret=weaviate.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"]),
            additional_headers={"X-OpenAI-Api-Key": st.secrets["OPENAI_API_KEY"]} # Good practice for Weaviate integrations
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

@st.cache_resource
def get_openai_client():
    """Initializes the OpenAI client."""
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

# --- 4. CORE APPLICATION LOGIC ---

def get_embedding(text, client, model="text-embedding-3-small"):
   """Generates embedding for a given text using OpenAI."""
   text = text.replace("\n", " ")
   response = client.embeddings.create(input=[text], model=model)
   return response.data[0].embedding

@st.cache_data(ttl=3600) # Cache for 1 hour
def ingest_airtable_to_weaviate():
    """
    Fetches data from Airtable, generates embeddings using OpenAI, and ingests into Weaviate.
    """
    weaviate_client = get_weaviate_client()
    openai_client = get_openai_client()
    
    if not weaviate_client or not openai_client:
        st.warning("Clients not available. Ingestion skipped.")
        return
        
    class_name = "CustodyDocs"

    if weaviate_client.schema.exists(class_name):
        weaviate_client.schema.delete_class(class_name)

    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "question", "dataType": ["text"]},
            {"name": "answer", "dataType": ["text"]},
        ],
    }
    weaviate_client.schema.create_class(class_obj)
    
    # MODIFIED: Using pyairtable syntax
    table = Table(
        st.secrets["AIRTABLE_API_KEY"], 
        st.secrets["AIRTABLE_BASE_ID"], 
        st.secrets["AIRTABLE_TABLE_NAME"]
    )
    records = table.all()

    with weaviate_client.batch as batch:
        batch.batch_size = 100
        for item in records:
            question_text = item.get("fields", {}).get("Question", "")
            if not question_text:
                continue

            # MODIFIED: Using OpenAI for embeddings
            emb = get_embedding(question_text, openai_client)

            data_obj = {
                "question": question_text,
                "answer": item.get("fields", {}).get("Answer", ""),
            }

            # **THE ORIGINAL ERROR FIX IS HERE**
            batch.add_data_object(
                data_object=data_obj,
                class_name=class_name,
                uuid=str(uuid.uuid4()),
                vector=emb
            )
    st.toast("Data ingestion complete!", icon="✅")


def perform_search(query: str):
    """Performs a vector search in Weaviate and returns the best result."""
    weaviate_client = get_weaviate_client()
    openai_client = get_openai_client()
    
    if not weaviate_client or not openai_client:
        return "Could not connect to the database. Please try again later."

    # MODIFIED: Using OpenAI for query vector
    query_vector = get_embedding(query, openai_client)
    
    response = (
        weaviate_client.query
        .get("CustodyDocs", ["question", "answer"])
        .with_near_vector({"vector": query_vector})
        .with_limit(1)
        .do()
    )
    
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    if results:
        return results[0]["answer"]
    else:
        return "I couldn't find a relevant answer in the documentation. Please try rephrasing your question."

# --- 5. STREAMLIT UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    st.image("logo.png", width=150) # Make sure logo.png is in your repo
    st.markdown("## Chat History")
    
    if "history" in st.session_state and st.session_state.history:
        for i, (q, a) in enumerate(st.session_state.history):
            with st.expander(f"Q: {q[:30]}..."):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
    else:
        st.info("Your chat history will appear here.")
        
    if st.button("🔄 Sync Data from Airtable"):
        with st.spinner("Ingesting data... This may take a moment."):
            ingest_airtable_to_weaviate()

# --- Main Page ---
with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.title("Custody Documentation Q&A")
    st.markdown("##### Private, authenticated workspace for your case records.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your custody documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching for an answer..."):
            response = perform_search(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append((prompt, response))
    st.rerun() # Reruns the script to update the sidebar history immediately
