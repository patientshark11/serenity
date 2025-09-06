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

def sanitize_name(name):
    """Removes characters that are problematic for API calls or filenames."""
    return re.sub(r"[/'\"]", "", name)

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

def _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model="gpt-4", entity_name=None):
    """
    A generic map-reduce framework for querying Weaviate, processing chunks, and summarizing.
    If an entity_name is provided, it performs a targeted search. Otherwise, it iterates through all docs.
    """
    collection_name = "CustodyDocs"
    if not weaviate_client.collections.exists(collection_name):
        return "The document collection does not exist. Please run the data sync first."

    collection = weaviate_client.collections.get(collection_name)

    items_to_process = []
    if entity_name:
        logging.info(f"Starting targeted search for entity: {entity_name}")
        # Phase 1: Search for the entity to get relevant chunks
        query_vector = get_embedding(entity_name, openai_client)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=50 # Get a large number of potentially relevant chunks
        )
        items_to_process = response.objects
    else:
        logging.info("Starting full collection iteration...")
        # Fallback to iterating the whole collection if no entity is specified
        items_to_process = collection.iterator()

    # MAP step: Process each chunk to extract relevant info
    mapped_results = []
    logging.info("Starting MAP step...")
    for item in items_to_process:
        chunk_content = item.properties['chunk_content']
        # The entity_name is passed to the prompt template if it exists
        map_prompt = map_prompt_template.format(chunk_content=chunk_content, entity_name=entity_name)
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are an expert data extractor."}, {"role": "user", "content": map_prompt}],
                timeout=30
            )
            extracted_info = response.choices[0].message.content
            if extracted_info and "no relevant information" not in extracted_info.lower():
                mapped_results.append(extracted_info)
        except Exception as e:
            logging.warning(f"Skipping a chunk due to an error during map stage: {e}")
            continue

    if not mapped_results:
        return f"Could not find any relevant information for '{entity_name}'." if entity_name else "Could not find any relevant information in the documents."

    logging.info(f"MAP step complete. Found {len(mapped_results)} relevant pieces of information.")

    # REDUCE step: Consolidate and format the extracted info
    logging.info("Starting REDUCE step...")
    combined_text = "\n---\n".join(mapped_results)
    reduce_prompt = reduce_prompt_template.format(combined_text=combined_text, entity_name=entity_name)

    try:
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert report writer that responds in Markdown."}, {"role": "user", "content": reduce_prompt}],
            stream=True
        )
        logging.info("REDUCE step complete. Streaming final report.")
        return response_stream
    except Exception as e:
        logging.error(f"Error during REDUCE step: {e}")
        return "An error occurred while finalizing the report. Please check the logs."

def generate_timeline(weaviate_client, openai_client, model="gpt-4"):
    """
    Generates a chronological timeline of events from the documents.
    """
    logging.info("Generating timeline...")
    map_prompt_template = """
    You are a data extractor. Your task is to find and list any events with specific dates or clear time references (e.g., "last week," "January 2023") from the following text. For each event, provide the date and a brief, neutral description. If no specific events are found, respond with "No relevant information."

    Text:
    "{chunk_content}"
    """

    reduce_prompt_template = """
    You are a historian. You have been given an unordered list of events extracted from various documents. Your task is to organize these events into a single, coherent, and chronologically sorted timeline.

    - Merge duplicate or very similar events to create a clean narrative.
    - Format each event clearly with the date first, followed by the description.
    - Present the final output in clear Markdown format, using headings for months or years where appropriate.

    Here is the unsorted list of events:
    ---
    {combined_text}
    ---
    """

    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model)

def summarize_entity(entity_name, weaviate_client, openai_client, model="gpt-4"):
    """
    Finds all mentions of a specific person or entity and creates a summary.
    """
    logging.info(f"Generating summary for entity: {entity_name}...")
    map_prompt_template = """
    You are a data extractor. Your task is to read the following text and extract any information, events, or descriptions related to the entity: '{entity_name}'. If the text is not relevant to this entity, respond with "No relevant information."

    Text:
    "{chunk_content}"
    """

    reduce_prompt_template = """
    You are a biographer. You have been given a collection of notes and mentions about '{entity_name}'. Your task is to synthesize this information into a concise and well-structured summary.

    - Start with a brief overview of the person or entity.
    - Organize the information thematically or chronologically, whichever makes more sense.
    - Merge duplicate information and resolve any minor contradictions to create a coherent narrative.
    - Present the final output in clear Markdown format.

    Here is the collection of notes:
    ---
    {combined_text}
    ---
    """

    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=entity_name)

def generate_report(report_type, weaviate_client, openai_client, model="gpt-4"):
    """
    Generates a specific report type (e.g., "Conflict Report") using the map-reduce framework.
    """
    logging.info(f"Generating report: {report_type}...")

    # The map prompt looks for information relevant to the general topic of the report.
    map_prompt_template = """
    You are a data extractor. Your task is to read the following text and extract any information relevant to the topic of '{entity_name}'. This could include events, statements, conflicts, communications, or other noteworthy details. If the text is not relevant to this topic, respond with "No relevant information."

    Text:
    "{chunk_content}"
    """

    # The reduce prompt assembles the information into a formal report.
    reduce_prompt_template = """
    You are a professional analyst. You have been given a collection of notes and information related to the topic of '{entity_name}'. Your task is to synthesize this information into a comprehensive and well-structured report.

    - Start with a clear introduction that defines the scope and purpose of the report.
    - Organize the information into logical sections with clear Markdown headings.
    - Provide a balanced and objective analysis based *only* on the information provided.
    - Conclude with a summary of the key findings.
    - Do not invent or infer information not present in the provided text.

    Here is the collection of information:
    ---
    {combined_text}
    ---
    """

    # We reuse the `entity_name` parameter to pass in the `report_type`.
    # This triggers a vector search for the topic, which is more efficient than a full scan.
    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=report_type)

def create_pdf(text_content, summary=None, sources=None):
    """
    Generates a PDF from text content, an optional summary, and a list of sources.
    Uses FPDF2 library. Handles basic UTF-8 characters.
    """
    logging.info("Generating PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()

        # Add a Unicode-compatible font
        # The font file needs to be available. Let's assume it's in the repo or installed.
        # For now, we'll stick to Arial which has some basic latin support.
        # A more robust solution would bundle a .ttf file.
        pdf.set_font("Arial", size=12)

        # Title
        if summary:
            pdf.set_font("Arial", 'B', 16)
            # Encode to latin-1, replacing unsupported characters
            pdf.cell(0, 10, summary.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'C')
            pdf.ln(10)

        # Body Content
        pdf.set_font("Arial", size=12)
        # Use multi_cell for automatic line wrapping
        pdf.multi_cell(0, 10, text_content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)

        # Sources
        if sources:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Sources", 0, 1)
            pdf.set_font("Arial", size=12)
            for source in sources:
                title = source.get('title', 'Unknown Source').encode('latin-1', 'replace').decode('latin-1')
                url = source.get('url', '')
                pdf.set_text_color(0, 0, 255) # Blue
                pdf.set_font('Arial', 'U', 12)
                pdf.cell(0, 7, f"- {title}", link=url)
                pdf.ln(5)
            pdf.set_text_color(0, 0, 0) # Reset color

        # Output to byte string
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        logging.info("PDF generation successful.")
        return pdf_bytes

    except Exception as e:
        logging.error("Failed to generate PDF.", exc_info=True)
        # Return an error message in bytes, as the caller expects a byte stream
        return b"Error: Could not generate the PDF file."

def fetch_report(report_name):
    """
    Fetches the content of a pre-generated report from the 'GeneratedReports' table in Airtable.
    """
    logging.info(f"Fetching pre-generated report: '{report_name}'")
    try:
        # Note: The report_name passed to this function should already be sanitized.
        reports_table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], "GeneratedReports")

        # Airtable formulas require you to escape single quotes if they appear in the value.
        # Although our sanitize_name function removes them, this is best practice.
        escaped_name = report_name.replace("'", "\\'")
        formula = f"{{ReportName}} = '{escaped_name}'"

        records = reports_table.all(formula=formula, max_records=1)

        if records and 'Content' in records[0]['fields']:
            logging.info(f"Successfully fetched report: '{report_name}'")
            return records[0]['fields']['Content']
        else:
            logging.warning(f"Report not found in Airtable: '{report_name}'")
            return f"Could not find a pre-generated report named '{report_name}'. It might still be generating or it may have failed to create."

    except Exception as e:
        logging.error(f"An error occurred while fetching report '{report_name}' from Airtable.", exc_info=True)
        return f"An error occurred while trying to fetch the report. Please check the application logs."
