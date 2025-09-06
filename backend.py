import os
import weaviate
import openai
from pyairtable import Table
import uuid
from weaviate.exceptions import WeaviateConnectionError
from openai import APIError as OpenAI_APIError
from requests.exceptions import HTTPError as AirtableHTTPError
import re
import logging
import json
from fpdf import FPDF
from io import BytesIO
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.init import Auth

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    if not text:
        return []
    splits = re.split(r'(?<=[\.!\?])\s+|\n\n+', text)
    chunks = []
    current_chunk = ""
    for split in splits:
        if not split: continue
        if len(current_chunk) + len(split) + 1 > chunk_size:
            chunks.append(current_chunk)
            overlap_start_index = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start_index:] + " " + split
        else:
            current_chunk += (" " if current_chunk else "") + split
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def connect_to_weaviate():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        headers={'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]}
    )
    return client

def get_embedding(text, openai_client):
    model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
    return response.data[0].embedding

def ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=2000):
    collection_name = "CustodyDocs"
    logging.info(f"Starting Airtable ingestion process for collection '{collection_name}' with chunk size: {chunk_size}.")
    try:
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
                chunks = split_text_into_chunks(full_content, chunk_size=chunk_size)
                for chunk in chunks:
                    emb = get_embedding(chunk, openai_client)
                    data_obj = {"chunk_content": chunk, "airtable_record_id": item["id"], "primary_source_content": source_url, "summary_title": summary_title}
                    batch.add_object(properties=data_obj, vector=emb)
        if batch.number_errors > 0:
            logging.error(f"Batch import finished with {batch.number_errors} errors: {batch.errors}")
        return "Sync successful!"
    except Exception as e:
        logging.error("An unexpected error occurred during ingestion.", exc_info=True)
        raise e

def generative_search(query, weaviate_client, openai_client, model="gpt-4"):
    logging.info(f"Performing smart search for query: {query}")
    try:
        # Pre-flight LLM call for query expansion and limit determination
        limit_prompt = f"""Analyze the user's query to enhance it for a semantic search.
1.  **determine_limit:** Analyze the query's complexity. If the user asks for a specific number of sources, use that number. Otherwise, for simple questions, use 3-5 sources. For broad or summary questions, use 10-15 sources.
2.  **expand_query:** Rewrite the user's query to be more descriptive and effective for a vector database search. Add context and keywords.

Return a JSON object with two keys: "expanded_query" (string) and "limit" (integer).

User Query: "{query}"
"""
        try:
            pre_flight_response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are an expert search analyst that returns only JSON."}, {"role": "user", "content": limit_prompt}],
                response_format={"type": "json_object"}
            )
            analysis = json.loads(pre_flight_response.choices[0].message.content)
            expanded_query = analysis.get("expanded_query", query)
            limit = analysis.get("limit", 5)
            logging.info(f"Smart search analysis complete. Expanded Query: '{expanded_query}', Limit: {limit}")
        except Exception as e:
            logging.warning(f"Failed to perform smart search analysis, falling back to defaults. Error: {e}")
            expanded_query = query
            limit = 5

        collection = weaviate_client.collections.get("CustodyDocs")
        query_vector = get_embedding(expanded_query, openai_client)
        response = collection.query.near_vector(
            near_vector=query_vector, limit=limit,
            return_properties=["chunk_content", "primary_source_content", "summary_title"]
        )
        results = response.objects
        if not results:
            return "I couldn't find a relevant answer in the documentation.", [], ""

        context = "\n---\n".join([obj.properties["chunk_content"] for obj in results])
        answer_prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nOriginal Question: {query}"
        answer_stream = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are a helpful assistant..."}, {"role": "user", "content": answer_prompt}], stream=True)

        sources_raw = [{"title": obj.properties.get("summary_title"), "url": obj.properties.get("primary_source_content")} for obj in results if obj.properties.get("primary_source_content")]
        unique_sources = {s["url"]: s for s in sources_raw}.values()
        sources = list(unique_sources)

        summary = f"Response to: \"{query[:40]}...\""
        return answer_stream, sources, summary
    except Exception as e:
        logging.error(f"An unexpected error occurred during generative search for query '{query}'.", exc_info=True)
        return "An unexpected error occurred. Please check the logs.", [], ""

def generate_timeline(weaviate_client, openai_client, model="gpt-4"):
    # ... (existing map-reduce logic) ...
    pass

def summarize_entity(entity_name, weaviate_client, openai_client, model="gpt-4"):
    # ... (existing map-reduce logic) ...
    pass

def generate_report(report_type, weaviate_client, openai_client, model="gpt-4"):
    # ... (existing map-reduce logic) ...
    pass

def create_pdf(text_content, summary=None, sources=None):
    # ... (existing pdf logic) ...
    pass

def fetch_report(report_name):
    # ... (existing fetch logic) ...
    pass
