import os
import atexit
import threading
from collections.abc import Mapping
import weaviate
import openai
from pyairtable import Api
import uuid
import re
import logging
import json
from fpdf import FPDF, XPos, YPos
from io import BytesIO
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth, Timeout

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_WEAVIATE_CLIENT_LOCK = threading.RLock()
_cached_weaviate_client = None
_cached_weaviate_config = None


def _connect_to_weaviate_cloud(**kwargs):
    """Connect to Weaviate Cloud using the modern helper when available."""

    connect_helper = getattr(weaviate, "connect_to_weaviate_cloud", None)
    if connect_helper is None:
        connect_module = getattr(weaviate, "connect", None)
        if connect_module is not None:
            connect_helper = getattr(connect_module, "connect_to_weaviate_cloud", None)

    if connect_helper is not None:
        return connect_helper(**kwargs)

    timeout = kwargs.pop("timeout", None)
    kwargs.pop("grpc", None)

    try:
        from weaviate.config import AdditionalConfig
    except Exception:  # pragma: no cover - defensive import
        additional_config = None
    else:
        additional_config = (
            AdditionalConfig(timeout=timeout) if timeout is not None else None
        )

    return weaviate.connect_to_wcs(
        cluster_url=kwargs["cluster_url"],
        auth_credentials=kwargs.get("auth_credentials"),
        headers=kwargs.get("headers"),
        additional_config=additional_config,
    )


def _close_weaviate_client(client):
    """Close a Weaviate client instance, suppressing errors."""

    if client is None:
        return

    close_method = getattr(client, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.warning("Failed to close Weaviate client cleanly: %s", exc)

def connect_to_weaviate(force_refresh=False):
    """Return a cached connection to the Weaviate instance.

    Parameters
    ----------
    force_refresh : bool, optional
        When ``True``, always create a fresh client, replacing any cached
        instance.
    """

    global _cached_weaviate_client, _cached_weaviate_config

    desired_config = (
        os.environ["WEAVIATE_URL"],
        os.environ["WEAVIATE_API_KEY"],
        os.environ["OPENAI_API_KEY"],
    )

    with _WEAVIATE_CLIENT_LOCK:
        if (
            not force_refresh
            and _cached_weaviate_client is not None
            and _cached_weaviate_config == desired_config
        ):
            return _cached_weaviate_client

        client_to_close = _cached_weaviate_client
        _cached_weaviate_client = None
        _cached_weaviate_config = None
        if client_to_close is not None:
            _close_weaviate_client(client_to_close)

        try:
            timeout = Timeout(init=10, query=60, insert=120)

            client = _connect_to_weaviate_cloud(
                cluster_url=desired_config[0],
                auth_credentials=Auth.api_key(desired_config[1]),
                headers={"X-OpenAI-Api-Key": desired_config[2]},
                timeout=timeout,
                grpc=False,
            )
        except Exception as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            raise

        _cached_weaviate_client = client
        _cached_weaviate_config = desired_config
        return client


def close_cached_weaviate_client():
    """Close and clear the cached Weaviate client, if one exists."""

    global _cached_weaviate_client, _cached_weaviate_config

    with _WEAVIATE_CLIENT_LOCK:
        client = _cached_weaviate_client
        if client is None:
            return
        _cached_weaviate_client = None
        _cached_weaviate_config = None

    _close_weaviate_client(client)


atexit.register(close_cached_weaviate_client)

def get_embedding(text, openai_client):
    """Generates an embedding for a given text using OpenAI."""
    model_name = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model_name)
    return response.data[0].embedding

def ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=None, limit=None):
    """Ingests data from Airtable into Weaviate, creating a new schema.

    Optional behaviour such as limiting the number of ingested Airtable records or
    adjusting the text chunk size can be controlled via the ``SYNC_LIMIT`` and
    ``SYNC_CHUNK_SIZE`` environment variables or by passing explicit arguments.
    """

    if chunk_size is None:
        try:
            chunk_size = int(os.environ.get("SYNC_CHUNK_SIZE", "2000"))
        except ValueError:
            chunk_size = 2000

    if limit is None:
        try:
            limit = int(os.environ.get("SYNC_LIMIT", "0")) or None
        except ValueError:
            limit = None

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
    api = Api(os.environ["AIRTABLE_API_KEY"])
    try:
        table = api.table(
            os.environ["AIRTABLE_BASE_ID"],
            os.environ["AIRTABLE_TABLE_NAME"],
        )
        records = table.all(max_records=limit) if limit else table.all()
        with custody_docs.batch.dynamic() as batch:
            for item in records:
                fields = item.get("fields", {})
                full_content = " ".join(str(v) for v in fields.values() if v)
                source_url = fields.get("Primary Source Content", "")
                summary_title = fields.get("Summary Title", "Untitled Source")
                if not full_content:
                    continue

                # Simple text splitting for now, can be improved later
                chunks = (lambda text, n: [text[i:i + n] for i in range(0, len(text), n)])(
                    full_content, chunk_size
                )

                for chunk in chunks:
                    emb = get_embedding(chunk, openai_client)
                    data_obj = {
                        "chunk_content": chunk,
                        "airtable_record_id": item["id"],
                        "primary_source_content": source_url,
                        "summary_title": summary_title,
                    }
                    batch.add_object(properties=data_obj, vector=emb)
        if batch.number_errors > 0:
            logging.error(f"Batch import finished with {batch.number_errors} errors.")
        return "Sync successful!"
    finally:
        close_airtable_api(api)


def close_airtable_api(api):
    """Attempt to close an Airtable Api instance if supported."""

    if api is None:
        return

    close_method = getattr(api, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.warning(f"Failed to close Airtable API cleanly: {exc}")
        return

    session = getattr(api, "session", None)
    if hasattr(session, "close"):
        try:
            session.close()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.warning(f"Failed to close Airtable session cleanly: {exc}")


def generative_search(query, weaviate_client, openai_client, model="gpt-4", hyde_model=None):
    """Performs a search using the HyDE technique.

    Parameters
    ----------
    query : str
        The user's question.
    weaviate_client : weaviate.Client
        Client connected to a Weaviate instance.
    openai_client : openai.OpenAI
        OpenAI client for generating text and embeddings.
    model : str, optional
        Model used to generate the final answer. Defaults to ``"gpt-4"``.
    hyde_model : str, optional
        Model used for generating the hypothetical answer in the HyDE step.
        Defaults to the ``OPENAI_HYDE_MODEL`` environment variable or
        ``"gpt-3.5-turbo"`` if unset.
    """
    logging.info(f"Performing HyDE search for query: {query}")

    # Determine model for hypothetical answer
    hyde_model = hyde_model or os.getenv("OPENAI_HYDE_MODEL", "gpt-3.5-turbo")

    # 1. Generate a hypothetical answer
    hyde_prompt = (
        "Write a detailed, factual paragraph that directly answers the following question. "
        "Do not say 'this is a hypothetical answer' or similar. "
        "Just provide the answer as if it were a real answer.\n\nQuestion: "
        f"{query}"
    )
    try:
        response = openai_client.chat.completions.create(
            model=hyde_model,
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

def _collect_context(search_query, weaviate_client, openai_client, limit=20):
    """Retrieves relevant chunks from Weaviate for a given query."""
    try:
        hyde_prompt = (
            "Write a concise paragraph that answers the following question. "
            "Do not mention this is hypothetical. Question: " + search_query
        )
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": hyde_prompt},
            ],
        )
        hypo_answer = response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to generate hypothetical answer: {e}")
        hypo_answer = search_query

    try:
        query_vector = get_embedding(hypo_answer, openai_client)
        collection = weaviate_client.collections.get("CustodyDocs")
        resp = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_properties=["chunk_content"],
        )
        context = "\n---\n".join(
            obj.properties.get("chunk_content", "") for obj in resp.objects
        )
        return context
    except Exception as e:
        logging.error(f"Failed to retrieve context from Weaviate: {e}")
        return ""

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
        query_vector = get_embedding(entity_name, openai_client)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=50
        )
        items_to_process = response.objects
    else:
        logging.info("Starting full collection iteration...")
        items_to_process = collection.iterator()

    mapped_results = []
    logging.info("Starting MAP step...")
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
            continue

    if not mapped_results:
        return f"Could not find any relevant information for '{entity_name}'." if entity_name else "Could not find any relevant information in the documents."

    logging.info(f"MAP step complete. Found {len(mapped_results)} relevant pieces of information.")

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

def generate_timeline(weaviate_client, openai_client, model="gpt-4", mode="map-reduce"):
    """Generates a chronological timeline of events from stored documents."""
    if mode == "map-reduce":
        # Use map-reduce logic for best accuracy
        map_prompt_template = """
        You are a data extractor. Your task is to find and list any events with specific dates or clear time references (e.g., "last week," "January 2023") from the following text. For each event, provide the date and a brief description.
        Text:
        "{chunk_content}"
        """
        reduce_prompt_template = """
        You are a historian. You have been given an unordered list of events extracted from various documents. Your task is to organize these events into a single, coherent, and chronologically sorted timeline. Merge duplicates, format each event with the date first.
        Here is the unsorted list of events:
        ---
        {combined_text}
        ---
        """
        return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model)
    else:
        # Use simple context aggregation (fast, less accurate)
        context = _collect_context("Create a chronological timeline of key events.", weaviate_client, openai_client, limit=40)
        if not context:
            return "Error: No data available to generate a timeline."
        prompt = (
            "Using only the following context, create a chronological timeline of key events."
            "\n\n"
            f"Context:\n{context}\n\nTimeline:"
        )
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes timelines based on provided documentation."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

def generate_report(report_type, weaviate_client, openai_client, model="gpt-4", mode="map-reduce"):
    """Generates a specific report type using stored documents."""
    if mode == "map-reduce":
        map_prompt_template = """
        You are a data extractor. Your task is to read the following text and extract any information relevant to the topic of '{entity_name}'. This could include events, statements, conflicts, communications, etc.
        Text:
        "{chunk_content}"
        """
        reduce_prompt_template = """
        You are a professional analyst. You have been given a collection of notes and information related to the topic of '{entity_name}'. Your task is to synthesize this information into a comprehensive, well-structured report in Markdown.
        Here is the collection of information:
        ---
        {combined_text}
        ---
        """
        return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=report_type)
    else:
        query = f"Generate a {report_type} based on the documentation."
        context = _collect_context(query, weaviate_client, openai_client, limit=40)
        if not context:
            return f"Error: No data available to generate {report_type}."
        prompt = (
            f"Using only the following context, write a {report_type}."
            "\n\n"
            f"Context:\n{context}\n\n{report_type}:"
        )
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates analytical reports based on provided context."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

def summarize_entity(entity, weaviate_client, openai_client, model="gpt-4", mode="map-reduce"):
    """Summarizes information about a specific entity from the documents."""
    if mode == "map-reduce":
        map_prompt_template = """
        You are a data extractor. Your task is to read the following text and extract any information, events, or descriptions related to the entity: '{entity_name}'. If the text is not relevant to the entity, ignore it.
        Text:
        "{chunk_content}"
        """
        reduce_prompt_template = """
        You are a biographer. You have been given a collection of notes and mentions about '{entity_name}'. Your task is to synthesize this information into a concise and well-structured summary in Markdown.
        Here is the collection of notes:
        ---
        {combined_text}
        ---
        """
        return _map_reduce_query(weaviate_client, openai_client, map_prompt_template, reduce_prompt_template, model, entity_name=entity)
    else:
        query = f"Summarize all available information about {entity}."
        context = _collect_context(query, weaviate_client, openai_client, limit=40)
        if not context:
            return f"Error: No data available to summarize {entity}."
        prompt = (
            f"Using only the following context, provide a concise summary about {entity}."
            "\n\n"
            f"Context:\n{context}\n\nSummary:"
        )
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes entities based on provided documentation."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

def sanitize_name(name):
    """Removes characters that are problematic for API calls or filenames."""
    return re.sub("[/'\"]", "", name)

def create_pdf(text_content, summary=None, sources=None):
    """Generates a PDF from text content, an optional summary, and a list of sources."""
    logging.info("Generating PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)

        if summary:
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(
                0,
                10,
                summary.encode('latin-1', 'replace').decode('latin-1'),
                border=0,
                align='C',
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.ln(10)

        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text_content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)

        if sources:
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(
                0,
                10,
                "Sources",
                border=0,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_font("Helvetica", size=12)
            for source in sources:
                title = source.get('title', 'Unknown Source').encode('latin-1', 'replace').decode('latin-1')
                url = source.get('url', '')
                pdf.set_text_color(0, 0, 255)
                pdf.set_font('Helvetica', 'U', 12)
                pdf.cell(0, 7, f"- {title}", link=url)
                pdf.ln(5)
            pdf.set_text_color(0, 0, 0)

        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")
        else:
            pdf_bytes = bytes(pdf_bytes)
        logging.info("PDF generation successful.")
        return pdf_bytes
    except Exception as e:
        logging.error("Failed to generate PDF.", exc_info=True)
        return b"Error: Could not generate the PDF file."

def fetch_report(report_name, api=None):
    """Fetches a pre-generated report from the Airtable reports table.

    The table name defaults to ``"GeneratedReports"`` but can be overridden
    via the ``AIRTABLE_REPORTS_TABLE_NAME`` environment variable. The field
    containing the report name defaults to ``"Name"`` and can be overridden
    via ``AIRTABLE_REPORT_NAME_FIELD``.

    Parameters
    ----------
    report_name : str
        Name of the report to fetch.
    api : pyairtable.Api, optional
        Existing Airtable API instance to reuse. If ``None``, a new instance
        will be created and closed automatically.
    """
    logging.info(f"Fetching report '{report_name}' from Airtable.")
    manage_api = api is None
    if manage_api:
        api = Api(os.environ["AIRTABLE_API_KEY"])
    try:
        reports_table_name = os.environ.get(
            "AIRTABLE_REPORTS_TABLE_NAME", "GeneratedReports"
        )
        report_name_field = os.getenv("AIRTABLE_REPORT_NAME_FIELD", "Name")
        reports_table = api.table(
            os.environ["AIRTABLE_BASE_ID"],
            reports_table_name,
        )

        # Ensure the configured report name field exists in the Airtable table
        try:
            schema_info = reports_table.schema()
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve schema for table '{reports_table_name}': {e}"
            ) from e

        def _extract_field_names(schema):
            """Return a set of field names from a schema object or mapping."""

            names = set()

            def _iter_fields(candidate):
                if candidate is None:
                    return []
                if isinstance(candidate, Mapping):
                    return list(candidate.values())
                if isinstance(candidate, (list, tuple, set)):
                    return list(candidate)
                return [candidate]

            def _add_from(candidate):
                for field in _iter_fields(candidate):
                    name = None
                    if isinstance(field, Mapping):
                        name = field.get("name")
                    else:
                        getter = getattr(field, "get", None)
                        if callable(getter):
                            try:
                                name = getter("name")
                            except Exception:  # pragma: no cover - defensive
                                name = None
                        if not name:
                            name = getattr(field, "name", None)
                    if name:
                        names.add(name)

            if isinstance(schema, Mapping):
                _add_from(schema.get("fields"))
            else:
                fields_attr = getattr(schema, "fields", None)
                if fields_attr is not None:
                    _add_from(fields_attr)

                get_method = getattr(schema, "get", None)
                if callable(get_method):
                    try:
                        _add_from(get_method("fields", None))
                    except Exception:  # pragma: no cover - defensive
                        pass

                if isinstance(schema, (list, tuple, set)):
                    _add_from(schema)
                else:
                    schema_dict = getattr(schema, "__dict__", None)
                    if isinstance(schema_dict, dict) and "fields" in schema_dict:
                        _add_from(schema_dict.get("fields"))

            return names

        table_fields = _extract_field_names(schema_info)

        if report_name_field not in table_fields:
            raise ValueError(
                f"Field '{report_name_field}' not found in Airtable table '{reports_table_name}'."
            )

        sanitized = sanitize_name(report_name)
        if sanitized != report_name:
            escaped_raw = report_name.replace("'", "\\'")
            formula = (
                f"OR({{{report_name_field}}}='{sanitized}', {{{report_name_field}}}='{escaped_raw}')"
            )
        else:
            formula = f"{{{report_name_field}}}='{sanitized}'"

        record = reports_table.first(formula=formula)
        if record:
            return record.get("fields", {}).get("Content", "Report content not found.")
        return f"Report '{report_name}' not found."
    except ValueError:
        raise
    except Exception as e:
        logging.error(f"Failed to fetch report '{report_name}': {e}")
        return f"Error: Could not fetch report '{report_name}'."
    finally:
        if manage_api:
            close_airtable_api(api)


def fetch_reports(report_names):
    """Fetch multiple pre-generated reports using a single Airtable API instance.

    Parameters
    ----------
    report_names : Iterable[str]
        The report names to fetch.

    Returns
    -------
    dict
        Mapping of each requested report name to its content or an error
        message if not found.
    """
    api = Api(os.environ["AIRTABLE_API_KEY"])
    try:
        return {name: fetch_report(name, api=api) for name in report_names}
    finally:
        close_airtable_api(api)
