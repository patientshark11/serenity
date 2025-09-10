import os
import weaviate
import openai
from pyairtable import Table
import uuid
from pyairtable.formulas import match
import re
import logging
import json
from fpdf import FPDF
from io import BytesIO
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_env_var(name: str) -> str:
    """Fetch an environment variable or raise a helpful error."""
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks of given size."""
    return [text[i:i + size] for i in range(0, len(text), size)]

def connect_to_weaviate():
    """Establishes a connection to the Weaviate instance."""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
            headers={'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]}
            cluster_url=_get_env_var("WEAVIATE_URL"),
            auth_credentials=Auth.api_key(_get_env_var("WEAVIATE_API_KEY")),
            headers={'X-OpenAI-Api-Key': _get_env_var("OPENAI_API_KEY")}
        )
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Weaviate: {e}")
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise

def get_embedding(text, openai_client):
    """Generates an embedding for a given text using OpenAI."""
    model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
    return response.data[0].embedding
    try:
        response = openai_client.embeddings.create(
            input=[text.replace("\n", " ")], model=model_name
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=2000):
    """Ingests data from Airtable into Weaviate, creating a new schema."""
    collection_name = "CustodyDocs"
    logging.info(f"Starting Airtable ingestion process for collection '{collection_name}'.")
    logger.info(f"Starting Airtable ingestion process for collection '{collection_name}'.")
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
    table = Table(
        _get_env_var("AIRTABLE_API_KEY"),
        _get_env_var("AIRTABLE_BASE_ID"),
        _get_env_var("AIRTABLE_TABLE_NAME"),
    )
    records = table.iterate()
    with custody_docs.batch.dynamic() as batch:
        for item in records:
            fields = item.get("fields", {})
            full_content = " ".join(str(v) for v in fields.values() if v)
            source_url = fields.get("Primary Source Content", "")
            summary_title = fields.get("Summary Title", "Untitled Source")
            if not full_content: continue
            
            chunks = (lambda text, n: [text[i:i+n] for i in range(0, len(text), n)])(full_content, chunk_size)
            
            for chunk in chunks:

            for chunk in _chunk_text(full_content, chunk_size):
                emb = get_embedding(chunk, openai_client)
                data_obj = {"chunk_content": chunk, "airtable_record_id": item["id"], "primary_source_content": source_url, "summary_title": summary_title}
                data_obj = {
                    "chunk_content": chunk,
                    "airtable_record_id": item["id"],
                    "primary_source_content": source_url,
                    "summary_title": summary_title,
                }
                batch.add_object(properties=data_obj, vector=emb)
    if batch.number_errors > 0:
        logging.error(f"Batch import finished with {batch.number_errors} errors.")
        logger.error(f"Batch import finished with {batch.number_errors} errors.")
    return "Sync successful!"

def generative_search(query, weaviate_client, openai_client, model="gpt-4"):
def generative_search(query, weaviate_client, openai_client, model="gpt-4", hyde_model="gpt-3.5-turbo"):
    """Performs a search using the HyDE technique."""
    logging.info(f"Performing HyDE search for query: {query}")
    
    logger.info(f"Performing HyDE search for query: {query}")

    hyde_prompt = (
        "Write a detailed, factual paragraph that directly answers the following question. "
        "Do not say 'this is a hypothetical answer' or similar. "
        "Just provide the answer as if it were an excerpt from a definitive source.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            model=hyde_model,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": hyde_prompt}]
        )
        hypothetical_answer = response.choices[0].message.content
        logging.info(f"Generated hypothetical answer: {hypothetical_answer[:100]}...")
        logger.info(f"Generated hypothetical answer: {hypothetical_answer[:100]}...")
    except Exception as e:
        logging.error(f"Failed to generate hypothetical answer: {e}. Falling back to original query.")
        logger.error(f"Failed to generate hypothetical answer: {e}. Falling back to original query.")
        hypothetical_answer = query

    query_vector = get_embedding(hypothetical_answer, openai_client)
    

    if not weaviate_client.collections.exists("CustodyDocs"):
        return None, [], "The document collection does not exist. Please run the data sync first."

    collection = weaviate_client.collections.get("CustodyDocs")
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=5,
        return_properties=["chunk_content", "primary_source_content", "summary_title"]
    )
    results = response.objects
    

    if not results:
        return "I couldn't find a relevant answer in the documentation.", [], ""
        return None, [], "I couldn't find a relevant answer in the documentation."

    context = "\n---\n".join([obj.properties["chunk_content"] for obj in results])
    final_prompt = (
        "Based ONLY on the following context, please provide a comprehensive answer to the user's original question.\n\n"
        f"Context:\n{context}\n\n"
        f"Original Question: {query}\n\nAnswer:"
    )
    
    answer_stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."}, {"role": "user", "content": final_prompt}],
        stream=True
    )
    
    sources_raw = [{"title": obj.properties.get("summary_title"), "url": obj.properties.get("primary_source_content")} for obj in results if obj.properties.get("primary_source_content")]
    unique_sources = {s["url"]: s for s in sources_raw}.values()
    sources = list(unique_sources)
    
    summary = f'Response to: "{query[:40]}..."'
    return answer_stream, sources, summary

def sanitize_name(name):
    """Removes characters that are problematic for API calls or filenames."""
    # Corrected line:
    return re.sub(r"[/'\"]", "", name)

def create_pdf(text_content, summary=None, sources=None):
    """Generates a PDF from text content, an optional summary, and a list of sources."""
    logging.info("Generating PDF report...")
    logger.info("Generating PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        if summary:
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, summary.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'C')
            pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text_content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)

        if sources:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Sources", 0, 1)
            pdf.set_font("Arial", size=12)
            for source in sources:
                title = source.get('title', 'Unknown Source').encode('latin-1', 'replace').decode('latin-1')
                url = source.get('url', '')
                pdf.set_text_color(0, 0, 255)
                pdf.set_font('Arial', 'U', 12)
                pdf.cell(0, 7, f"- {title}", link=url)
                pdf.ln(5)
            pdf.set_text_color(0, 0, 0)

        pdf_bytes = pdf.output()
        logging.info("PDF generation successful.")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        logger.info("PDF generation successful.")
        return pdf_bytes
    except Exception as e:
        logging.error("Failed to generate PDF.", exc_info=True)
        logger.error("Failed to generate PDF.", exc_info=True)
        return b"Error: Could not generate the PDF file."

def _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model="gpt-4", entity_name=None):
    collection_name = "CustodyDocs"
    if not weaviate_client.collections.exists(collection_name):
        return "The document collection does not exist. Please run the data sync first."
    collection = weaviate_client.collections.get(collection_name)
    
    items_to_process = []
    if entity_name:
        logging.info(f"Starting targeted search for entity: {entity_name}")
        logger.info(f"Starting targeted search for entity: {entity_name}")
        query_vector = get_embedding(entity_name, openai_client)
        response = collection.query.near_vector(near_vector=query_vector, limit=50)
        items_to_process = response.objects
    else:
        logging.info("Starting full collection iteration...")
        logger.info("Starting full collection iteration...")
        items_to_process = collection.iterator()

    mapped_results = []
    logging.info("Starting MAP step...")
    logger.info("Starting MAP step...")
    for item in items_to_process:
        chunk_content = item.properties['chunk_content']
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
            logger.warning(f"Skipping a chunk due to an error during map stage: {e}")
            continue
    
    if not mapped_results:
        return f"Could not find any relevant information for '{entity_name}'." if entity_name else "Could not find any relevant information in the documents."

    logging.info(f"MAP step complete. Found {len(mapped_results)} relevant pieces of information.")
    logging.info("Starting REDUCE step...")
    logger.info(f"MAP step complete. Found {len(mapped_results)} relevant pieces of information.")
    logger.info("Starting REDUCE step...")
    combined_text = "\n---\n".join(mapped_results)
    reduce_prompt = reduce_prompt_template.format(combined_text=combined_text, entity_name=entity_name)

    try:
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert report writer that responds in Markdown."}, {"role": "user", "content": reduce_prompt}],
            stream=True
        )
        logging.info("REDUCE step complete. Streaming final report.")
        logger.info("REDUCE step complete. Streaming final report.")
        return response_stream
    except Exception as e:
        logging.error(f"Error during REDUCE step: {e}")
        logger.error(f"Error during REDUCE step: {e}")
        return "An error occurred while finalizing the report. Please check the logs."

def generate_timeline(weaviate_client, openai_client, model="gpt-4"):
    logging.info("Generating timeline...")
    logger.info("Generating timeline...")
    map_prompt_template = (
        'Extract any events with specific dates or clear time references (e.g., "yesterday," "last week," "January 2023") '
        'from the following text. For each event, provide the date and a brief, neutral description. '
        'If no specific events are found, respond with "No relevant information."\n\n'
        'Text:\n"{chunk_content}"'
    )
    reduce_prompt_template = (
        'You are a historian. You have been given a list of events extracted from various documents. '
        'Your task is to organize these events into a single, coherent, and chronologically sorted timeline.\n\n'
        '- Merge duplicate or very similar events.\n'
        '- Format each event clearly with the date first, followed by the description.\n'
        '- Present the final output in clear Markdown format, using headings for months or years where appropriate.\n\n'
        'Here is the unsorted list of events:\n---\n{combined_text}\n---'
    )
    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model)

def summarize_entity(entity_name, weaviate_client, openai_client, model="gpt-4"):
    logging.info(f"Generating summary for entity: {entity_name}...")
    logger.info(f"Generating summary for entity: {entity_name}...")
    map_prompt_template = (
        "You are a data extractor. Your task is to read the following text and extract any information, events, or descriptions "
        "related to the entity: '{entity_name}'. If the text is not relevant to this entity, respond with "
        '"No relevant information."\n\nText:\n"{chunk_content}"'
    )
    reduce_prompt_template = (
        "You are a biographer. You have been given a collection of notes and mentions about '{entity_name}'. "
        "Your task is to synthesize this information into a concise and well-structured summary.\n\n"
        "- Start with a brief overview of the person or entity.\n"
        "- Organize the information thematically or chronologically, whichever makes more sense.\n"
        "- Merge duplicate information and resolve any minor contradictions to create a coherent narrative.\n"
        "- Present the final output in clear Markdown format.\n\n"
        "Here is the collection of notes:\n---\n{combined_text}\n---"
    )
    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=entity_name)

def generate_report(report_type, weaviate_client, openai_client, model="gpt-4"):
    logging.info(f"Generating report: {report_type}...")
    logger.info(f"Generating report: {report_type}...")
    map_prompt_template = (
        "You are a data extractor. Your task is to read the following text and extract any information relevant to the topic of "
        "'{entity_name}'. This could include events, statements, conflicts, communications, or other noteworthy details. "
        'If the text is not relevant to this topic, respond with "No relevant information."\n\n'
        'Text:\n"{chunk_content}"'
    )
    reduce_prompt_template = (
        "You are a professional analyst. You have been given a collection of notes and information related to the topic of '{entity_name}'. "
        "Your task is to synthesize this information into a comprehensive and well-structured report.\n\n"
        "- Start with a clear introduction that defines the scope and purpose of the report.\n"
        "- Organize the information into logical sections with clear Markdown headings.\n"
        "- Provide a balanced and objective analysis based *only* on the information provided.\n"
        "- Conclude with a summary of the key findings.\n"
        "- Do not invent or infer information not present in the provided text.\n\n"
        "Here is the collection of information:\n---\n{combined_text}\n---"
    )
    return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=report_type)

def fetch_report(report_name):
    """Fetches the content of a pre-generated report from the 'GeneratedReports' table in Airtable."""
    logging.info(f"Fetching pre-generated report: '{report_name}'")
    logger.info(f"Fetching pre-generated report: '{report_name}'")
    try:
        reports_table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], "GeneratedReports")
        escaped_name = report_name.replace("'", "\\'")
        formula = f"{{ReportName}} = '{escaped_name}'"
        
        reports_table = Table(
            _get_env_var("AIRTABLE_API_KEY"),
            _get_env_var("AIRTABLE_BASE_ID"),
            "GeneratedReports",
        )
        formula = match({"ReportName": report_name})

        records = reports_table.all(formula=formula, max_records=1)
        

        if records and 'Content' in records[0]['fields']:
            logging.info(f"Successfully fetched report: '{report_name}'")
            logger.info(f"Successfully fetched report: '{report_name}'")
            return records[0]['fields']['Content']
        else:
            logging.warning(f"Report not found in Airtable: '{report_name}'")
            return f"Could not find a pre-generated report named '{report_name}'. It might still be generating or it may have failed to create."
            logger.warning(f"Report not found in Airtable: '{report_name}'")
            return (
                f"Could not find a pre-generated report named '{report_name}'. It might still be generating or it may have failed to create."
            )
            
    except Exception as e:
        logging.error(f"An error occurred while fetching report '{report_name}' from Airtable.", exc_info=True)
        return f"An error occurred while trying to fetch the report. Please check the application logs."
        logger.error(
            f"An error occurred while fetching report '{report_name}' from Airtable.",
            exc_info=True,
        )
        return (
            "An error occurred while trying to fetch the report. Please check the application logs."
        )
