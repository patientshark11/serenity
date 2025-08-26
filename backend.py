import os
import weaviate
import openai
from pyairtable import Table
import uuid
from weaviate.exceptions import WeaviateConnectionError
from openai import APIError as OpenAI_APIError
from requests.exceptions import HTTPError as AirtableHTTPError

def connect_to_weaviate():
    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    return client

def get_embedding(text, openai_client):
    model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
    return response.data[0].embedding

def ingest_airtable_to_weaviate(weaviate_client, openai_client):
    class_name = "CustodyDocs"
    if weaviate_client.schema.exists(class_name):
        weaviate_client.schema.delete_class(class_name)
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "airtable_record_id", "dataType": ["text"]},
            {"name": "primary_source_content", "dataType": ["text"]},
        ],
    }
    weaviate_client.schema.create_class(class_obj)
    table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], os.environ["AIRTABLE_TABLE_NAME"])
    records = table.all()
    with weaviate_client.batch as batch:
        batch.batch_size = 100
        for item in records:
            fields = item.get("fields", {})
            content = " ".join(str(v) for k, v in fields.items())
            if content:
                emb = get_embedding(content, openai_client)
                data_obj = {
                    "content": content,
                    "airtable_record_id": item["id"],
                    "primary_source_content": fields.get("Primary Source Content", ""),
                }
                batch.add_data_object(
                    data_object=data_obj,
                    class_name=class_name,
                    uuid=str(uuid.uuid4()),
                    vector=emb,
                )

def generative_search(query, weaviate_client, openai_client):
    try:
        response_count = weaviate_client.query.aggregate("CustodyDocs").with_meta_count().do()
        count = response_count["data"]["Aggregate"]["CustodyDocs"][0]["meta"]["count"]
        if count == 0:
            return "The database is empty. Please click 'Sync Data from Airtable' in the sidebar to add your documentation.", [], ""
    except Exception:
        return "Could not connect to the database. Please ensure it's running and accessible.", [], ""

    query_vector = get_embedding(query, openai_client)
    response = (
        weaviate_client.query.get("CustodyDocs", ["content", "primary_source_content"])
        .with_near_vector({"vector": query_vector})
        .with_limit(3)
        .do()
    )
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")

    if not results:
        return "I couldn't find a relevant answer in the documentation.", [], ""

    context = "\n".join([r["content"] for r in results])
    answer_prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"

    try:
        answer_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": answer_prompt},
            ],
        )
        answer = answer_response.choices[0].message.content
        sources = [r["primary_source_content"] for r in results if r.get("primary_source_content")]

        summary_prompt = f"Please create a very short, one-sentence summary of the following question and answer. Question: {query} Answer: {answer}"
        summary_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": summary_prompt},
            ],
        )
        summary = summary_response.choices[0].message.content

        return answer, sources, summary
    except OpenAI_APIError as e:
        return f"An error occurred with the OpenAI API: {e}", [], ""
