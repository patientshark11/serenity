import streamlit as st
import weaviate
import os
from airtable import Airtable
from sentence_transformers import SentenceTransformer
import uuid

# --- 1. PAGE CONFIGURATION ---
# Set the page configuration as the very first Streamlit command.
st.set_page_config(
    page_title="Project Serenity - Custody Documentation",
    page_icon="⚖️",  # You can use an emoji or a URL to a favicon
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. CUSTOM CSS FOR A PROFESSIONAL LOOK ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You would create a style.css file with the content below
# For simplicity in this example, I'm injecting it directly.
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #F0F2F6;
    }

    /* Main content container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom container for headers */
    .custom-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }

    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        padding: 1rem;
    }
    
    /* Hide Streamlit's default header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# --- 3. CACHING & API CLIENTS ---
# Use st.cache_resource for objects that should be created only once, like API clients or models.

@st.cache_resource
def get_weaviate_client():
    """Initializes and returns a Weaviate client."""
    try:
        weaviate_url = st.secrets["WEAVIATE_URL"]
        weaviate_api_key = st.secrets["WEAVIATE_API_KEY"]
        
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    """Loads and returns the sentence-transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 4. CORE APPLICATION LOGIC ---

# The data ingestion should be cached to prevent running it on every interaction.
# st.cache_data is for data that doesn't change often.
@st.cache_data(ttl=3600) # Cache for 1 hour
def ingest_airtable_to_weaviate(limit=100):
    """
    Fetches data from Airtable, generates embeddings, and ingests into Weaviate.
    This function is now cached to avoid re-running on every page load.
    """
    client = get_weaviate_client()
    model = get_embedding_model()
    
    if client is None:
        st.warning("Weaviate client not available. Ingestion skipped.")
        return
        
    class_name = "CustodyDocs" # Make sure this matches your Weaviate schema

    # Check if class exists and clear it if needed (for idempotent ingestion)
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)

    class_obj = {
        "class": class_name,
        "vectorizer": "none", # Specify you're providing your own vectors
        "properties": [
            {"name": "question", "dataType": ["text"]},
            {"name": "answer", "dataType": ["text"]},
        ],
    }
    client.schema.create_class(class_obj)
    
    # Airtable setup
    airtable_api_key = st.secrets["AIRTABLE"]["API_KEY"]
    base_id = st.secrets["AIRTABLE"]["BASE_ID"]
    table_name = st.secrets["AIRTABLE"]["TABLE_NAME"]
    airtable = Airtable(base_id, table_name, api_key=airtable_api_key)
    records = airtable.get_all(max_records=limit)

    with client.batch as batch:
        batch.batch_size = 100
        for item in records:
            question_text = item.get("fields", {}).get("Question", "")
            if not question_text:
                continue

            # Generate embedding for the question
            emb = model.encode(question_text).tolist()

            data_obj = {
                "question": question_text,
                "answer": item.get("fields", {}).get("Answer", ""),
            }

            # **THE CRITICAL FIX IS HERE**
            # Use `vector=emb` instead of `vectors={"default": emb}`
            batch.add_data_object(
                data_object=data_obj,
                class_name=class_name,
                uuid=str(uuid.uuid4()),
                vector=emb
            )
    st.toast("Data ingestion complete!", icon="✅")


def perform_search(query: str):
    """Performs a vector search in Weaviate and returns the best result."""
    client = get_weaviate_client()
    model = get_embedding_model()
    
    if client is None:
        return "Could not connect to the database. Please try again later."

    query_vector = model.encode(query).tolist()
    
    response = (
        client.query
        .get("CustodyDocs", ["question", "answer"])
        .with_near_vector({"vector": query_vector})
        .with_limit(1)
        .do()
    )
    
    if response["data"]["Get"]["CustodyDocs"]:
        return response["data"]["Get"]["CustodyDocs"][0]["answer"]
    else:
        return "I couldn't find a relevant answer in the documentation. Please try rephrasing your question."

# --- 5. STREAMLIT UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    # The logo now has its own space and won't be cut off.
    st.image("logo.png", width=150) # Assuming your logo is named logo.png
    st.markdown("## Chat History")
    
    # Display chat history in the sidebar
    if "history" in st.session_state and st.session_state.history:
        for i, (q, a) in enumerate(st.session_state.history):
            with st.expander(f"Q: {q[:30]}..."):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
    else:
        st.info("Your chat history will appear here.")
        
    # Option to trigger data ingestion manually
    if st.button("🔄 Sync Data from Airtable"):
        with st.spinner("Ingesting data from Airtable into Weaviate..."):
            ingest_airtable_to_weaviate()


# --- Main Page ---

# A styled container for the main title
with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.title("Custody Documentation Q&A")
    st.markdown("##### Private, authenticated workspace for your case records.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The new, improved chat input box that stays at the bottom
if prompt := st.chat_input("Ask a question about your custody documentation..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching for an answer..."):
            response = perform_search(prompt)
            st.markdown(response)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Also add to the permanent history for the sidebar
    st.session_state.history.append((prompt, response))
