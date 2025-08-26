import os
import weaviate
import openai
from pyairtable import Table
import uuid
from weaviate.exceptions import WeaviateConnectionError
from openai import APIError as OpenAI_APIError
from requests.exceptions import HTTPError as AirtableHTTPError
import re

def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    if not text:
        return []

    # Split by paragraphs or sentences
    splits = re.split(r'(?<=[\.!\?])\s+|\n\n+', text)

    chunks = []
    current_chunk = ""
    for split in splits:
        if not split:
            continue
        # If the new split doesn't fit, finalize the current chunk
        if len(current_chunk) + len(split) + 1 > chunk_size:
            chunks.append(current_chunk)
            # Start the next chunk with an overlap
            overlap_start_index = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start_index:] + " " + split
        else:
            current_chunk += (" " if current_chunk else "") + split

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

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
            {"name": "chunk_content", "dataType": ["text"]},
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
            # Concatenate all fields into a single text block
            full_content = " ".join(f"{k}: {v}" for k, v in fields.items() if v)
            source_url = fields.get("Primary Source Content", "")

            if not full_content:
                continue

            # Split the full content into chunks
            chunks = split_text_into_chunks(full_content)

            for chunk in chunks:
                emb = get_embedding(chunk, openai_client)
                data_obj = {
                    "chunk_content": chunk,
                    "airtable_record_id": item["id"],
                    "primary_source_content": source_url,
                }
                batch.add_data_object(
                    data_object=data_obj,
                    class_name=class_name,
                    uuid=str(uuid.uuid4()),
                    vector=emb,
                )

def generative_search(query, weaviate_client, openai_client):
    try:
        # Check if there are any objects in the database
        response_count = weaviate_client.query.aggregate("CustodyDocs").with_meta_count().do()
        count = response_count["data"]["Aggregate"]["CustodyDocs"][0]["meta"]["count"]
        if count == 0:
            return "The database is empty. Please click 'Sync Data from Airtable' in the sidebar to add your documentation.", [], ""
    except Exception:
        return "Could not connect to the database. Please ensure it's running and accessible.", [], ""

    # Search for relevant chunks
    query_vector = get_embedding(query, openai_client)
    response = (
        weaviate_client.query.get("CustodyDocs", ["chunk_content", "primary_source_content"])
        .with_near_vector({"vector": query_vector})
        .with_limit(5) # Retrieve a few relevant chunks
        .do()
    )
    results = response.get("data", {}).get("Get", {}).get("CustodyDocs")

    if not results:
        return "I couldn't find a relevant answer in the documentation.", [], ""

    # Construct context from the chunks
    context = "\n---\n".join([r["chunk_content"] for r in results])
    answer_prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"

    try:
        # Generate the answer
        answer_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information in the context to answer."},
                {"role": "user", "content": answer_prompt},
            ],
        )
        answer = answer_response.choices[0].message.content

        # Get unique sources from the retrieved chunks
        sources = list(set([r["primary_source_content"] for r in results if r.get("primary_source_content")]))

        # Generate a summary of the Q&A
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

def generate_timeline(weaviate_client, openai_client):
    try:
        # 1. Fetch all chunks
        response = weaviate_client.query.get("CustodyDocs", ["chunk_content"]).with_limit(1000).do()
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")

        if not all_chunks:
            return "No documents found to generate a timeline from."

        # 2. Map Step: Extract events from batches of chunks
        batch_size = 5
        intermediate_results = []
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            context = "\n---\n".join([c["chunk_content"] for c in batch_chunks])
            prompt = f"Extract all events with specific dates from the following text. List each event on a new line. If no specific dated events are found, respond with 'No events found'.\n\nContext:\n{context}"

            map_response = openai_client.chat.completions.create(
                model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert at extracting dated events from text."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = map_response.choices[0].message.content
            if "no events found" not in result.lower():
                intermediate_results.append(result)

        if not intermediate_results:
            return "Could not find any specific dated events in the documents."

        # 3. Reduce Step: Combine and format the timeline
        combined_events = "\n".join(intermediate_results)
        reduce_prompt = f"Please take the following list of events and organize them into a single, coherent, chronological timeline. Merge duplicate events. \n\nEvents:\n{combined_events}"

        reduce_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert at organizing events into a timeline."},
                {"role": "user", "content": reduce_prompt},
            ],
        )
        timeline = reduce_response.choices[0].message.content
        return timeline
    except Exception as e:
        return f"An error occurred while generating the timeline: {e}"

def summarize_entity(entity_name, weaviate_client, openai_client):
    try:
        # 1. Fetch relevant chunks
        query_vector = get_embedding(entity_name, openai_client)
        response = (
            weaviate_client.query.get("CustodyDocs", ["chunk_content"])
            .with_near_vector({"vector": query_vector})
            .with_limit(100)
            .do()
        )
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")

        if not all_chunks:
            return f"No documents found mentioning '{entity_name}'."

        # 2. Map Step: Summarize info about the entity from batches of chunks
        batch_size = 5
        intermediate_summaries = []
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            context = "\n---\n".join([c["chunk_content"] for c in batch_chunks])
            prompt = f"Extract all information, events, and direct statements related to '{entity_name}' from the following text. Be concise. If no specific information is found, respond with 'No information found'.\n\nContext:\n{context}"

            map_response = openai_client.chat.completions.create(
                model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert at extracting information about a specific person from text."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = map_response.choices[0].message.content
            if "no information found" not in result.lower():
                intermediate_summaries.append(result)

        if not intermediate_summaries:
            return f"Could not find any specific information about '{entity_name}' in the documents."

        # 3. Reduce Step: Combine and format the final summary
        combined_summaries = "\n".join(intermediate_summaries)
        reduce_prompt = f"Please synthesize the following points into a single, detailed summary about '{entity_name}'. \n\nInformation:\n{combined_summaries}"

        reduce_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert at writing detailed summaries about people."},
                {"role": "user", "content": reduce_prompt},
            ],
        )
        summary = reduce_response.choices[0].message.content
        return summary
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

def generate_report(report_type, weaviate_client, openai_client):
    report_concepts = {
        "Conflict Report": "Report on arguments, disagreements, fights, conflicts, and issues.",
        "Legal Communication Summary": "Summary of communication involving lawyers, legal matters, court, attorneys, and custody agreements."
    }

    concept = report_concepts.get(report_type)
    if not concept:
        return "Invalid report type selected."

    try:
        # 1. Fetch relevant chunks
        query_vector = get_embedding(concept, openai_client)
        response = (
            weaviate_client.query.get("CustodyDocs", ["chunk_content"])
            .with_near_vector({"vector": query_vector})
            .with_limit(100)
            .do()
        )
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")

        if not all_chunks:
            return f"No documents found to generate a '{report_type}'."

        # 2. Map Step
        batch_size = 5
        intermediate_reports = []
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            context = "\n---\n".join([c["chunk_content"] for c in batch_chunks])
            prompt = f"Extract all information relevant to a '{report_type}' from the following text. Be detailed. If no relevant information is found, respond with 'No information found'.\n\nContext:\n{context}"

            map_response = openai_client.chat.completions.create(
                model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert at extracting information for reports from text."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = map_response.choices[0].message.content
            if "no information found" not in result.lower():
                intermediate_reports.append(result)

        if not intermediate_reports:
            return f"Could not find any information to build a '{report_type}'."

        # 3. Reduce Step
        combined_info = "\n".join(intermediate_reports)
        reduce_prompt = f"Please synthesize the following information into a single, cohesive '{report_type}'. \n\nInformation:\n{combined_info}"

        reduce_response = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_COMPLETION_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are an expert at writing detailed reports."},
                {"role": "user", "content": reduce_prompt},
            ],
        )
        report = reduce_response.choices[0].message.content
        return report
    except Exception as e:
        return f"An error occurred while generating the report: {e}"
