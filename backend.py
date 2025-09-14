import os
import weaviate
import openai
from pyairtable import Table
import uuid
import re
import logging
import json
from fpdf import FPDF
from io import BytesIO
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_weaviate():
    """Establishes a connection to the Weaviate instance."""
    try:
        client = weaviate.connect_to_wcs(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
            headers={'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]}
        )
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Weaviate: {e}")
        raise

def get_embedding(text, openai_client):
    """Generates an embedding for a given text using OpenAI."""
    model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
    return response.data[0].embedding

def ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=2000):
    """Ingests data from Airtable into Weaviate, creating a new schema."""
    collection_name = "CustodyDocs"
    logging.info(f"Starting Airtable ingestion process for collection '{collection_name}'.")
    if weaviate_client.collections.exists(collection_name):
        weaviate_client.collections.delete(collection_name)

    custody_docs = weaviate_client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_content", data_type=DataType.TEXT),
            Property(name="airtable_record_id", data_type=DataType.TEXT),
            Property(name="primary_source_content", data_type=DataType.TEXT),
            Property(name="summary_title", data_type=DataType.TEXT),
        ]
    )
    table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], os.environ["AIRTABLE_TABLE_NAME"])
    records = table.all()
    with custody_docs.batch.dynamic() as batch:
        for item in records:
            fields = item.get("fields", {})
            full_content = " ".join(str(v) for v in fields.values() if v)
            source_url = fields.get("Primary Source Content", "")
            summary_title = fields.get("Summary Title", "Untitled Source")
            if not full_content: continue

            # Simple text splitting for now, can be improved later
            chunks = (lambda text, n: [text[i:i+n] for i in range(0, len(text), n)])(full_content, chunk_size)

            for chunk in chunks:
                emb = get_embedding(chunk, openai_client)
                data_obj = {"chunk_content": chunk, "airtable_record_id": item["id"], "primary_source_content": source_url, "summary_title": summary_title}
                batch.add_object(properties=data_obj, vector=emb)
    if batch.number_errors > 0:
        logging.error(f"Batch import finished with {batch.number_errors} errors.")
    return "Sync successful!"

def generative_search(query, weaviate_client, openai_client, model="gpt-4"):
    """
    Performs a search using the HyDE technique.
    
