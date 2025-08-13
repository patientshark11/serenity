import streamlit as st
import os
import weaviate
import openai
from pyairtable import Table
import uuid
from weaviate.exceptions import ConnectionError as WeaviateConnectionError
from openai import APIError as OpenAI_APIError
from requests.exceptions import HTTPError as AirtableHTTPError

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity - Custody Q&A",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    #MainMenu, footer, header { visibility: hidden; }
    .history-container {
        height: 300px; overflow-y: auto; padding: 15px; border: 1px solid #e0e0e0;
        border-radius: 10px; background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
# Using st.session_state to hold clients after first successful connection
def connect_to_services():
    if "clients_connected" in st.session_state:
        return True
    
    try:
        st.session_state.weaviate_client = weaviate.Client(
            url=os.environ.get("WEAVIATE_URL"),
            auth_client_secret=weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")),
            additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")}
        )
        st.session_state.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        st.session_state.clients_connected = True
        return True
    except Exception as e:
        st.error(f"Failed to connect to backend services. Please check credentials. Error: {e}")
        return False

def get_embedding(text):
   model_name = os.environ.get("OPENAI_MODEL", "text-embedding-3-small")
   text = text.replace("\n", " ")
   response = st.session_state.openai_client.embeddings.create(input=[text], model=model_name)
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
    response = (st.session_state.weaviate_client.query.get("CustodyDocs", ["question", "answer"])
                .with_near_vector({"vector": query_vector}).with_limit(1).do())
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
    return results[0]["answer"] if results else "I couldn't find a relevant answer in the documentation."

# --- 4. UI LAYOUT ---
if "messages" not in st.session_state: st.session_state.messages = []

logo_path = os.environ.get("APP_LOGO_PATH", "logo.png")
st.image(logo_path, width=150)
st.title("Custody Documentation Q&A")
st.markdown("Private, authenticated workspace for your case records.")
st.divider()

# Only show main app if secrets and connections are valid
if all(os.environ.get(key) for key in ["WEAVIATE_URL", "OPENAI_API_KEY", "AIRTABLE_API_KEY"]) and connect_to_services():
    with st.container(border=True):
        st.markdown("##### Ask a question")
        user_query = st.text_area("Type your question here...", key="user_input_area", height=100)
        
        if st.button("Get Answer", type="primary"):
            if user_query:
                with st.spinner("Searching..."):
                    try:
                        bot_response = perform_search(user_query)
                        st.session_state.messages.append({"role": "user", "content": user_query})
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    except OpenAI_APIError as e:
                        st.error(f"OpenAI Error: {e.message}. Please check your billing status on the OpenAI website.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please enter a question.")

    if st.session_state.messages:
        st.markdown("#### Conversation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.divider()

    st.m
