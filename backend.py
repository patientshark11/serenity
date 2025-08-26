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

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='a'
)

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

def ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=2000):
    class_name = "CustodyDocs"
    logging.info(f"Starting Airtable ingestion process with chunk size: {chunk_size}.")
    try:
        # Schema handling
        if weaviate_client.schema.exists(class_name):
            weaviate_client.schema.delete_class(class_name)
        class_obj = {
            "class": class_name, "vectorizer": "none",
            "properties": [
                {"name": "chunk_content", "dataType": ["text"]},
                {"name": "airtable_record_id", "dataType": ["text"]},
                {"name": "primary_source_content", "dataType": ["text"]},
                {"name": "summary_title", "dataType": ["text"]},
            ],
        }
        weaviate_client.schema.create_class(class_obj)
        logging.info("Weaviate schema created successfully.")

        # Airtable fetching
        table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], os.environ["AIRTABLE_TABLE_NAME"])
        records = table.all()
        logging.info(f"Fetched {len(records)} records from Airtable.")

        # Data processing and batching
        with weaviate_client.batch as batch:
            batch.batch_size = 100
            for item in records:
                fields = item.get("fields", {})
                full_content = " ".join(str(v) for v in fields.values() if v)
                source_url = fields.get("Primary Source Content", "")
                summary_title = fields.get("Summary Title", "Untitled Source")

                if not full_content:
                    continue

                chunks = split_text_into_chunks(full_content, chunk_size=chunk_size)
                for chunk in chunks:
                    emb = get_embedding(chunk, openai_client)
                    data_obj = {
                        "chunk_content": chunk,
                        "airtable_record_id": item["id"],
                        "primary_source_content": source_url,
                        "summary_title": summary_title,
                    }
                    batch.add_data_object(data_object=data_obj, class_name=class_name, uuid=str(uuid.uuid4()), vector=emb)
        logging.info("Weaviate batch import completed.")
        return "Sync successful!"
    except WeaviateConnectionError as e:
        logging.error("Failed to connect to Weaviate during ingestion.", exc_info=True)
        raise ConnectionError("Could not connect to Weaviate database.") from e
    except AirtableHTTPError as e:
        logging.error("Failed to fetch data from Airtable.", exc_info=True)
        raise ConnectionError("Could not fetch data from Airtable.") from e
    except Exception as e:
        logging.error("An unexpected error occurred during ingestion.", exc_info=True)
        raise e

def generative_search(query, weaviate_client, openai_client, model="gpt-4", limit=5):
    logging.info(f"Performing generative search for query: {query} with model: {model} and limit: {limit}")
    try:
        # Weaviate query
        query_vector = get_embedding(query, openai_client)
        response = (
            weaviate_client.query.get("CustodyDocs", ["chunk_content", "primary_source_content", "summary_title"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .do()
        )
        results = response.get("data", {}).get("Get", {}).get("CustodyDocs")
        if not results:
            logging.warning(f"No relevant chunks found for query: {query}")
            return "I couldn't find a relevant answer in the documentation.", [], ""

        # OpenAI completion
        context = "\n---\n".join([r["chunk_content"] for r in results])
        answer_prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
        answer_stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information in the context to answer."},
                {"role": "user", "content": answer_prompt},
            ],
            stream=True
        )

        # Create a list of unique sources with titles and URLs
        sources_raw = [
            {"title": r.get("summary_title"), "url": r.get("primary_source_content")}
            for r in results if r.get("primary_source_content")
        ]
        # Deduplicate based on URL
        unique_sources = {s["url"]: s for s in sources_raw}.values()
        sources = list(unique_sources)

        # With streaming, we can't generate a summary from the full answer.
        # We'll create a placeholder summary instead.
        summary = f"Response to: \"{query[:40]}...\""
        logging.info(f"Generative search stream initiated for query: {query}")
        return answer_stream, sources, summary

    except WeaviateConnectionError as e:
        logging.error("Failed to connect to Weaviate during search.", exc_info=True)
        return "Error: Could not connect to the database.", [], ""
    except OpenAI_APIError as e:
        logging.error(f"OpenAI API error during search for query '{query}': {e}", exc_info=True)
        return f"Error: The AI model failed to generate a response. ({e.code})", [], ""
    except Exception as e:
        logging.error(f"An unexpected error occurred during generative search for query '{query}'.", exc_info=True)
        return "An unexpected error occurred. Please check the logs.", [], ""

def generate_timeline(weaviate_client, openai_client, model="gpt-4"):
    logging.info("Generating timeline.")
    try:
        # More scalable: first, try to find chunks that are likely to contain dates.
        # This is a simple keyword search. A more advanced method could use pattern matching.
        date_keywords = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "today", "yesterday", "last week", "last month"]

        where_operands = []
        for keyword in date_keywords:
            where_operands.append({
                "path": ["chunk_content"],
                "operator": "Like",
                "valueText": f"*{keyword}*",
            })

        where_filter = {
            "operator": "Or",
            "operands": where_operands,
        }

        response = weaviate_client.query.get("CustodyDocs", ["chunk_content"]).with_where(where_filter).with_limit(1000).do()
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")
        if not all_chunks: return "No documents with date-related keywords found to generate a timeline from."

        logging.info(f"Found {len(all_chunks)} potentially relevant chunks for timeline.")
        batch_size = 5
        intermediate_results = []
        for i in range(0, len(all_chunks), batch_size):
            context = "\n---\n".join([c["chunk_content"] for c in all_chunks[i:i+batch_size]])
            prompt = f"Extract all events with specific dates from the following text. List each event on a new line. If no specific dated events are found, respond with 'No events found'.\n\nContext:\n{context}"
            map_response = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at extracting dated events from text."}, {"role": "user", "content": prompt}])
            result = map_response.choices[0].message.content
            if "no events found" not in result.lower(): intermediate_results.append(result)

        if not intermediate_results: return "Could not find any specific dated events in the documents."

        combined_events = "\n".join(intermediate_results)
        reduce_prompt = f"Please take the following list of events and organize them into a single, coherent, chronological timeline. Merge duplicate events. \n\nEvents:\n{combined_events}"
        reduce_stream = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at organizing events into a timeline."}, {"role": "user", "content": reduce_prompt}], stream=True)
        logging.info("Timeline generation stream initiated.")
        return reduce_stream
    except WeaviateConnectionError as e:
        logging.error("Failed to connect to Weaviate during timeline generation.", exc_info=True)
        return "Error: Could not connect to the database."
    except OpenAI_APIError as e:
        logging.error(f"OpenAI API error during timeline generation: {e}", exc_info=True)
        return f"Error: The AI model failed to generate a response. ({e.code})"
    except Exception as e:
        logging.error("An unexpected error occurred during timeline generation.", exc_info=True)
        return "An unexpected error occurred. Please check the logs."

def summarize_entity(entity_name, weaviate_client, openai_client, model="gpt-4"):
    logging.info(f"Summarizing entity: {entity_name}")
    try:
        query_vector = get_embedding(entity_name, openai_client)
        response = weaviate_client.query.get("CustodyDocs", ["chunk_content"]).with_near_vector({"vector": query_vector}).with_limit(100).do()
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")
        if not all_chunks: return f"No documents found mentioning '{entity_name}'."

        batch_size = 5
        intermediate_summaries = []
        for i in range(0, len(all_chunks), batch_size):
            context = "\n---\n".join([c["chunk_content"] for c in all_chunks[i:i+batch_size]])
            prompt = f"Extract all information, events, and direct statements related to '{entity_name}' from the following text. Be concise. If no specific information is found, respond with 'No information found'.\n\nContext:\n{context}"
            map_response = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at extracting information about a specific person from text."}, {"role": "user", "content": prompt}])
            result = map_response.choices[0].message.content
            if "no information found" not in result.lower(): intermediate_summaries.append(result)

        if not intermediate_summaries: return f"Could not find any specific information about '{entity_name}' in the documents."

        combined_summaries = "\n".join(intermediate_summaries)
        reduce_prompt = f"Please synthesize the following points into a single, detailed summary about '{entity_name}'. \n\nInformation:\n{combined_summaries}"
        reduce_stream = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at writing detailed summaries about people."}, {"role": "user", "content": reduce_prompt}], stream=True)
        logging.info(f"Successfully initiated summary stream for entity: {entity_name}")
        return reduce_stream
    except WeaviateConnectionError as e:
        logging.error(f"Failed to connect to Weaviate during entity summary for '{entity_name}'.", exc_info=True)
        return "Error: Could not connect to the database."
    except OpenAI_APIError as e:
        logging.error(f"OpenAI API error during entity summary for '{entity_name}': {e}", exc_info=True)
        return f"Error: The AI model failed to generate a response. ({e.code})"
    except Exception as e:
        logging.error(f"An unexpected error occurred during entity summary for '{entity_name}'.", exc_info=True)
        return "An unexpected error occurred. Please check the logs."

def generate_report(report_type, weaviate_client, openai_client, model="gpt-4"):
    logging.info(f"Generating report: {report_type}")
    report_concepts = {
        "Conflict Report": "Report on arguments, disagreements, fights, conflicts, and issues.",
        "Legal Communication Summary": "Summary of communication involving lawyers, legal matters, court, attorneys, and custody agreements."
    }
    concept = report_concepts.get(report_type)
    if not concept: return "Invalid report type selected."

    try:
        query_vector = get_embedding(concept, openai_client)
        response = weaviate_client.query.get("CustodyDocs", ["chunk_content"]).with_near_vector({"vector": query_vector}).with_limit(100).do()
        all_chunks = response.get("data", {}).get("Get", {}).get("CustodyDocs")
        if not all_chunks: return f"No documents found to generate a '{report_type}'."

        batch_size = 5
        intermediate_reports = []
        for i in range(0, len(all_chunks), batch_size):
            context = "\n---\n".join([c["chunk_content"] for c in all_chunks[i:i+batch_size]])
            prompt = f"Extract all information relevant to a '{report_type}' from the following text. Be detailed. If no relevant information is found, respond with 'No information found'.\n\nContext:\n{context}"
            map_response = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at extracting information for reports from text."}, {"role": "user", "content": prompt}])
            result = map_response.choices[0].message.content
            if "no information found" not in result.lower(): intermediate_reports.append(result)

        if not intermediate_reports: return f"Could not find any information to build a '{report_type}'."

        combined_info = "\n".join(intermediate_reports)
        reduce_prompt = f"Please synthesize the following information into a single, cohesive '{report_type}'. \n\nInformation:\n{combined_info}"
        reduce_stream = openai_client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at writing detailed reports."}, {"role": "user", "content": reduce_prompt}], stream=True)
        logging.info(f"Successfully initiated report stream: {report_type}")
        return reduce_stream
    except WeaviateConnectionError as e:
        logging.error(f"Failed to connect to Weaviate during report generation for '{report_type}'.", exc_info=True)
        return "Error: Could not connect to the database."
    except OpenAI_APIError as e:
        logging.error(f"OpenAI API error during report generation for '{report_type}': {e}", exc_info=True)
        return f"Error: The AI model failed to generate a response. ({e.code})"
    except Exception as e:
        logging.error(f"An unexpected error occurred during report generation for '{report_type}'.", exc_info=True)
        return "An unexpected error occurred. Please check the logs."

from fpdf import FPDF
from io import BytesIO

def create_pdf(text_content):
    logging.info("Creating PDF document.")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add text to PDF, handling unicode characters
        # The 'latin-1' encoding is used with a fallback to replace characters that can't be encoded.
        pdf.multi_cell(0, 10, text=text_content.encode('latin-1', 'replace').decode('latin-1'))

        # Save PDF to a byte buffer
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        buffer.write(pdf_bytes)
        buffer.seek(0)

        logging.info("PDF creation successful.")
        return buffer.getvalue()
    except Exception as e:
        logging.error("An unexpected error occurred during PDF creation.", exc_info=True)
        # Return an empty byte string or handle error appropriately
        return b""
