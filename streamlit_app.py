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
    page_title="Project Serenity - Custody Q&A",
    page_icon="⚖️",
    layout="wide",  # Use the wide layout for a more spacious feel
    initial_sidebar_state="expanded" # Ensure the sidebar is open by default
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    /* Custom Streamlit App styling */
    .stApp { background-color: #F0F2F6; }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Style the main container for the input form */
    .main-input-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. BACKEND LOGIC (No changes needed) ---
def connect_to_services():
    if "clients_connected" in st.session_state: return True
    try:
        st.session_state.weaviate_client = weaviate.Client(url=os.environ.get("WEAVIATE_URL"), auth_client_secret=weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")), additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")})
        st.session_state.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        st.session_state.clients_connected = True
        return True
    except Exception as e:
        st.error(f"Failed to connect to backend services. Please check credentials. Error: {e}")
        return False

def get_embedding(text):
   model_name = os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
   response = st.session_state.openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
   return response.data[0].embedding

def ingest_airtable_to_weaviate():
    class_name = "CustodyDocs"
    if st.session_state.weaviate_client.schema.exists(class_name):
        st.session_state.weaviate_client.schema.delete_class(class_name)
    class_obj = {"class": class_name, "vectorizer": "none", "properties": [{"name": "question", "dataType": ["text"]}, {"name": "answer", "dataType": ["text"]}]}
    st.session_state.weaviate_client.schema.create_class(class_obj)
    table = Table(os.environ.get("AIRTABLE_API_KEY"), os.environ.get("AIRTABLE_BASE_ID"), os.environ.get("AIRTABLE_TABLE_NAME"))
    records = table.all()
    with st.session_state.weaviate_client.batch as batch:
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
    response = (st.session_state.weaviate_client.query.get("CustodyDocs", ["question", "answer"]).with_near_vector({"vector": query_vector}).with_limit(1).do())
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    return results[0]["answer"] if results else "I couldn't find a relevant answer in the documentation."

# --- 4. DEFINITIVE UI LAYOUT ---

# --- Sidebar ---
with st.sidebar:
    logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
    st.image(logo_path, width=150)
    st.title("Controls & History")
    
    if st.button("🔄 Sync Data from Airtable", use_container_width=True):
        with st.spinner("Connecting to Airtable and syncing data..."):
            try:
                ingest_airtable_to_weaviate()
            except WeaviateConnectionError as e: st.error(f"Weaviate Error: {e}")
            except AirtableHTTPError as e: st.error(f"Airtable Error: {e}. Check PAT/Base/Table details.")
            except OpenAI_APIError as e: st.error(f"OpenAI Error: {e.message}. Check billing.")
            except Exception as e: st.error(f"An unexpected error occurred: {e}")
    
    st.divider()
    
    st.header("Chat History")
    # Display history in a simple, clean format
    if "messages" in st.session_state and st.session_state.messages:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                 st.markdown(f"**Answer:** {msg['content']}")
                 st.divider() # Separator after each answer
    else:
        st.info("Your conversation history will appear here.")


# --- Main Content Area ---
st.title("Custody Documentation Q&A")
st.markdown("Private, authenticated workspace for your case records.")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check for credentials and connections before rendering the main app
if not all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY"]) or not connect_to_services():
    st.error("Application is not configured correctly. Please check all environment variables in Render and ensure services are reachable.")
else:
    # --- Prominent Input Form ---
    with st.container(border=True):
        user_query = st.text_area("Ask a question about your documentation:", height=120, placeholder="e.g., What are the standard holiday schedules?")
        
        if st.button("Get Answer", type="primary", use_container_width=True):
            if user_query:
                with st.spinner("Searching..."):
                    try:
                        bot_response = perform_search(user_query)
                        # Add user query and bot response to the history
                        st.session_state.messages.append({"role": "user", "content": user_query})
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        # Rerun to update the display immediately
                        st.rerun()
                    except OpenAI_APIError as e: st.error(f"OpenAI Error: {e.message}. Check billing.")
                    except Exception as e: st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please enter a question.")

    st.markdown("---")

    # --- Conversation Display ---
    st.header("Conversation")
    if not st.session_state.messages:
        st.info("Your current conversation will be displayed here.")
    else:
        # Display the full conversation using clean chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
