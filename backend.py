import os
import weaviate
from openai import OpenAI
from pyairtable import Table
import uuid
import re
import logging
import json
try:
    from fpdf import FPDF
except Exception:  # pragma: no cover - optional dependency
    FPDF = None
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

def ingest_airtable_to_weaviate(limit=None, chunk_size=2000):
    """Ingests data from Airtable into Weaviate, creating a new schema.

    Parameters
    ----------
    limit : int | None
        Maximum number of Airtable records to ingest. If ``None`` all records
        are processed.
    chunk_size : int
        Size of text chunks when splitting large documents.

    Returns
    -------
    dict
        Status information including number of records ingested or an error
        message if ingestion fails.
    """
    weaviate_client = None
    try:
        weaviate_client = connect_to_weaviate()
        openai_client = OpenAI()

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

        table = Table(
            os.environ["AIRTABLE_API_KEY"],
            os.environ["AIRTABLE_BASE_ID"],
            os.environ["AIRTABLE_TABLE_NAME"],
        )
        table_kwargs = {"max_records": limit} if limit is not None else {}
        records = table.all(**table_kwargs)
        ingested = 0
        with custody_docs.batch.dynamic() as batch:
            for item in records:
                fields = item.get("fields", {})
                full_content = " ".join(str(v) for v in fields.values() if v)
                source_url = fields.get("Primary Source Content", "")
                summary_title = fields.get("Summary Title", "Untitled Source")
                if not full_content:
                    continue

                # Simple text splitting for now, can be improved later
                chunks = (lambda text, n: [text[i:i+n] for i in range(0, len(text), n)])(full_content, chunk_size)

                for chunk in chunks:
                    emb = get_embedding(chunk, openai_client)
                    data_obj = {
                        "chunk_content": chunk,
                        "airtable_record_id": item["id"],
                        "primary_source_content": source_url,
                        "summary_title": summary_title,
                    }
                    batch.add_object(properties=data_obj, vector=emb)
                ingested += 1

        result = {
            "status": "ok",
            "records_ingested": ingested,
            "errors": getattr(batch, "number_errors", 0),
        }
    except Exception as e:
        logging.error(f"Ingestion failed: {e}", exc_info=True)
        result = {"status": "error", "message": str(e)}
    finally:
        try:
            weaviate_client.close()
        except Exception:
            pass

    return result

def generative_search(query, weaviate_client, openai_client, model="gpt-4"):
    """
    Performs a search using the HyDE technique.
    1. Generates a hypothetical answer to the user's query.
    2. Embeds the hypothetical answer to get a vector.
    3. Searches Weaviate for documents similar to that vector.
    4. Generates a final answer based on the retrieved documents.
    """
    logging.info(f"Performing HyDE search for query: {query}")

    # 1. Generate a hypothetical answer
    hyde_prompt = f"Write a detailed, factual paragraph that directly answers the following question. Do not say 'this is a hypothetical answer' or similar. Just provide the answer as if it were a[...]"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": hyde_prompt}]
        )
        hypothetical_answer = response.choices[0].message.content
        logging.info(f"Generated hypothetical answer: {hypothetical_answer[:100]}...")
    except Exception as e:
        logging.error(f"Failed to generate hypothetical answer: {e}. Falling back to original query.")
        hypothetical_answer = query

    # 2. Embed the hypothetical answer
    query_vector = get_embedding(hypothetical_answer, openai_client)

    # 3. Search Weaviate
    collection = weaviate_client.collections.get("CustodyDocs")
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=5,
        return_properties=["chunk_content", "primary_source_content", "summary_title"]
    )
    results = response.objects

    if not results:
        return "I couldn't find a relevant answer in the documentation.", [], ""

    # 4. Generate the final answer
    context = "\n---\n".join([obj.properties["chunk_content"] for obj in results])
    final_prompt = f"Based ONLY on the following context, please provide a comprehensive answer to the user's original question.\n\nContext:\n{context}\n\nOriginal Question: {query}\n\nAnswer:" 

    answer_stream = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."}, {"role": "user", "content": final_prompt}],
        stream=True
    )

    sources_raw = [{"title": obj.properties.get("summary_title"), "url": obj.properties.get("primary_source_content")} for obj in results if obj.properties.get("primary_source_content")]
    unique_sources = {s["url"]: s for s in sources_raw}.values()
    sources = list(unique_sources)

    summary = f"Response to: \"{query[:40]}...\""
    return answer_stream, sources, summary

def sanitize_name(name):
    """Removes characters that are problematic for API calls or filenames."""
    return re.sub(r"[/'\"]", "", name)

def create_pdf(text_content, summary=None, sources=None):
    """Generates a PDF from text content, an optional summary, and a list of sources."""
    logging.info("Generating PDF report...")
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

        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")
        else:
            pdf_bytes = bytes(pdf_bytes)
        logging.info("PDF generation successful.")
        return pdf_bytes
    except Exception as e:
        logging.error("Failed to generate PDF.", exc_info=True)
        return b"Error: Could not generate the PDF file."
def fetch_report(report_name):
    """Fetches a pre-generated report from the 'GeneratedReports' Airtable table."""
    logging.info(f"Fetching report '{report_name}' from Airtable.")
    try:
        reports_table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], "GeneratedReports")
        record = reports_table.first(formula=f"{{ReportName}}='{report_name}'")
        if record:
            return record.get("fields", {}).get("Content", "Report content not found.")
        return f"Report '{report_name}' not found."
    except Exception as e:
        logging.error(f"Failed to fetch report '{report_name}': {e}")
        return f"Error: Could not fetch report '{report_name}'."

