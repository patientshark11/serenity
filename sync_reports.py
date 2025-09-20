"""
Cron-friendly script to sync Airtable → Weaviate and pre-generate reports.
This script should be run on a schedule (e.g., daily) by a cron job.

It performs two main tasks:
1.  Syncs the main data table from Airtable to the Weaviate vector database.
2.  Generates all standard analysis reports (Timeline, Conflict Report, etc.) and
    a summary for each key person, then saves the output to a separate
    reports table in Airtable (default name: 'GeneratedReports') for instant
    retrieval by the app. The table name can be overridden via the
    ``AIRTABLE_REPORTS_TABLE_NAME`` environment variable.

The OpenAI models used for the MAP and REDUCE steps default to
``backend.DEFAULT_MAP_MODEL`` and ``backend.DEFAULT_REDUCE_MODEL`` but can be
overridden via the ``OPENAI_MAP_MODEL`` and ``OPENAI_REDUCE_MODEL``
environment variables.
"""
import os
import sys
import json
import logging
import backend
import openai
import traceback
from collections.abc import Mapping
from pyairtable import Api
from datetime import datetime

# --- CONFIGURATION ---
REPORT_MODE = os.environ.get("REPORT_MODE", "map-reduce")  # "map-reduce" or "simple"
MAP_MODEL = os.environ.get("OPENAI_MAP_MODEL", backend.DEFAULT_MAP_MODEL)
REDUCE_MODEL = os.environ.get("OPENAI_REDUCE_MODEL", backend.DEFAULT_REDUCE_MODEL)
GENERATE_PDF = True  # Set to False to skip PDF generation

def _safe_int_env(var_name, default):
    """Return an integer from the environment or a fallback default."""

    value = os.environ.get(var_name)
    if value in (None, ""):
        return default

    try:
        return int(value)
    except ValueError:
        logging.warning(
            "Environment variable %s must be an integer. Using default %s.",
            var_name,
            default,
        )
        return default


def _load_json_mapping_env(var_name):
    """Parse a JSON mapping from the environment, returning an empty dict on error."""

    raw_value = os.environ.get(var_name)
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        logging.warning(
            "Environment variable %s must contain valid JSON. Ignoring value.",
            var_name,
        )
        return {}

    if isinstance(parsed, Mapping):
        return dict(parsed)

    logging.warning(
        "Environment variable %s must decode to a mapping. Ignoring value.",
        var_name,
    )
    return {}


def _parse_timeline_search_config():
    """Return configuration for the timeline report fallback search."""

    limit = _safe_int_env("TIMELINE_CHUNK_LIMIT", 50)
    query = os.environ.get("TIMELINE_SEARCH_QUERY")
    if query:
        query = query.strip() or None

    search_type = os.environ.get("TIMELINE_SEARCH_TYPE")
    if search_type:
        search_type = search_type.strip().lower() or None

    options = _load_json_mapping_env("TIMELINE_SEARCH_OPTIONS")

    return {
        "limit": limit,
        "query": query,
        "type": search_type,
        "options": options,
    }


def build_reports_to_generate(weaviate_client, openai_client):
    """Return the report generators configured for the nightly job."""

    timeline_config = _parse_timeline_search_config()

    reports = {
        "Timeline": lambda cfg=timeline_config: backend.generate_timeline(
            weaviate_client,
            openai_client,
            model=REDUCE_MODEL,
            mode=REPORT_MODE,
            map_model=MAP_MODEL,
            search_limit=cfg["limit"],
            search_query=cfg["query"],
            search_type=cfg["type"],
            search_options=cfg["options"],
        ),
        "Conflict Report": lambda: backend.generate_report(
            "Conflict Report",
            weaviate_client,
            openai_client,
            model=REDUCE_MODEL,
            mode=REPORT_MODE,
            map_model=MAP_MODEL,
        ),
        "Legal Communication Summary": lambda: backend.generate_report(
            "Legal Communication Summary",
            weaviate_client,
            openai_client,
            model=REDUCE_MODEL,
            mode=REPORT_MODE,
            map_model=MAP_MODEL,
        ),
    }

    key_people = get_key_people()
    for person in key_people:
        reports[f"Summary for {person}"] = lambda p=person: backend.summarize_entity(
            p,
            weaviate_client,
            openai_client,
            model=REDUCE_MODEL,
            mode=REPORT_MODE,
            map_model=MAP_MODEL,
        )

    return reports


def get_key_people():
    """Try to fetch key people from Airtable, otherwise fallback to static list."""
    api = None
    try:
        api = Api(os.environ["AIRTABLE_API_KEY"])
        table = api.table(
            os.environ["AIRTABLE_BASE_ID"],
            os.environ["AIRTABLE_TABLE_NAME"],
        )
        people_records = table.all(formula="FIND('person', {Entity Type})")
        key_people = []
        for rec in people_records:
            name = rec.get("fields", {}).get("Name")
            if name and name not in key_people:
                key_people.append(name)
        if key_people:
            print(f"Fetched key people from Airtable: {key_people}")
            return key_people
    except Exception as e:
        print(f"Could not fetch key people from Airtable, falling back to static list. Error: {e}", file=sys.stderr)
    finally:
        backend.close_airtable_api(api)
    # Fallback static list
    return [
        "Kim", "Diego", "Kim's family/friends", "YWCA Staff",
        "Heather Ulrich", "DSS/Youth Villages", "Diego's mom"
    ]

def _extract_text_fragment(obj, *, _seen=None):
    """Recursively extract text content from OpenAI streaming chunks."""

    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return ""
    _seen.add(obj_id)

    pieces = []

    if isinstance(obj, Mapping):
        for key in ("content", "text", "value", "delta", "message"):
            if key in obj:
                pieces.append(_extract_text_fragment(obj[key], _seen=_seen))
        if not pieces:
            for value in obj.values():
                pieces.append(_extract_text_fragment(value, _seen=_seen))
        return "".join(piece for piece in pieces if piece)

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            pieces.append(_extract_text_fragment(item, _seen=_seen))
        return "".join(piece for piece in pieces if piece)

    for attr in ("content", "text", "value", "delta", "message", "choices"):
        if hasattr(obj, attr):
            pieces.append(
                _extract_text_fragment(getattr(obj, attr), _seen=_seen)
            )

    if not pieces:
        try:
            iterator = iter(obj)
        except TypeError:
            return ""
        else:
            for item in iterator:
                pieces.append(_extract_text_fragment(item, _seen=_seen))

    return "".join(piece for piece in pieces if piece)


def _stringify_report_content(raw_content):
    """Convert streamed or static report content into a single string."""

    if raw_content is None:
        return ""

    if isinstance(raw_content, str):
        return raw_content

    if isinstance(raw_content, bytes):
        return raw_content.decode("utf-8", errors="ignore")

    try:
        iterator = iter(raw_content)
    except TypeError:
        return str(raw_content)

    pieces = []
    for chunk in iterator:
        text = _extract_text_fragment(chunk)
        if text:
            pieces.append(text)

    combined = "".join(pieces)
    if combined:
        return combined

    return str(raw_content)


def generate_and_save_report(reports_table, name, generator_func):
    sanitized_name = backend.sanitize_name(name)
    print(f"\nGenerating report: '{name}' (Sanitized: '{sanitized_name}')...")
    try:
        report_content = generator_func()
        report_content = _stringify_report_content(report_content)

        if "error" in report_content.lower() or "could not find" in report_content.lower():
            print(f"WARNING: Report for '{name}' generation resulted in a non-content message: {report_content}")

        if GENERATE_PDF:
            try:
                pdf_bytes = backend.create_pdf(report_content, summary=name)
                print(
                    f"PDF generated for '{name}' ({len(pdf_bytes)} bytes). Attachment upload is currently disabled."
                )
            except Exception as pdf_err:
                logging.error("Failed to generate PDF for '%s': %s", name, pdf_err)

        report_name_field = os.environ.get("AIRTABLE_REPORT_NAME_FIELD", "Name")

        record_to_save = {
            report_name_field: sanitized_name,
            "Content": report_content,
            "LastGenerated": datetime.now().isoformat(),
        }
        key_fields = [report_name_field]
        try:
            response = reports_table.batch_upsert(
                [{"fields": record_to_save}], key_fields=key_fields
            )
            # Ensure the Airtable API returned a created or updated record
            if (
                not response
                or not isinstance(response, list)
                or not any(r.get("id") for r in response)
            ):
                logging.error(
                    "Airtable upsert for '%s' returned unexpected response: %s",
                    name,
                    response,
                )
                raise RuntimeError(
                    f"Airtable upsert for '{name}' did not return a record"
                )
        except Exception as upsert_err:
            server_response = getattr(upsert_err, "body", upsert_err)
            logging.error(
                "Failed to save report for '%s' to Airtable: %s", name, server_response
            )
            raise
        print(f"Successfully generated and saved report for '{name}'")
        return True
    except Exception as e:
        logging.error(
            "Failed to generate or save report for '%s'. Error: %s\n%s",
            name,
            e,
            traceback.format_exc(),
        )
        raise

def main():
    print("--- Starting nightly job: Data Sync and Report Generation ---")

    # --- 1. Initialize Clients ---
    weaviate_client = None
    try:
        weaviate_client = backend.connect_to_weaviate()
        openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print("Successfully connected to Weaviate and OpenAI.")
    except Exception as e:
        print(f"FATAL: Could not connect to backend services. Aborting job. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Sync Main Data from Airtable to Weaviate ---
    try:
        print("Starting main data sync to Weaviate...")
        sync_status = backend.ingest_airtable_to_weaviate(weaviate_client, openai_client, chunk_size=2000)
        print(f"Data sync finished. Status: {sync_status}")
    except Exception as e:
        print(f"ERROR: Weaviate data sync failed. Reports will be generated with existing data. Error: {e}", file=sys.stderr)

    # --- 3. Pre-generate Analysis Reports ---
    print("\nStarting report pre-generation...")
    api = None
    try:
        reports_table_name = os.environ.get(
            "AIRTABLE_REPORTS_TABLE_NAME", "GeneratedReports"
        )
        api = Api(os.environ["AIRTABLE_API_KEY"])
        reports_table = api.table(
            os.environ["AIRTABLE_BASE_ID"],
            reports_table_name,
        )
        print(f"Connected to '{reports_table_name}' Airtable table.")
    except Exception as e:
        print(
            f"FATAL: Could not connect to '{reports_table_name}' table in Airtable. Aborting job. Error: {e}",
            file=sys.stderr,
        )
        backend.close_airtable_api(api)
        backend.close_cached_weaviate_client()
        sys.exit(1)

    # --- 4. Reports Definition (easy to modify/expand) ---
    reports_to_generate = build_reports_to_generate(weaviate_client, openai_client)

    # --- 5. Generate & Save Reports ---
    failed_reports = []
    for name, generator_func in reports_to_generate.items():
        try:
            if not generate_and_save_report(reports_table, name, generator_func):
                failed_reports.append(name)
        except Exception as loop_err:
            print(
                f"ERROR: Unexpected failure while processing '{name}': {loop_err}",
                file=sys.stderr,
            )
            failed_reports.append(name)

    if failed_reports:
        print(
            "The following reports failed to generate or save: " + ", ".join(failed_reports),
            file=sys.stderr,
        )

    print("\n--- Nightly job finished ---")
    backend.close_airtable_api(api)
    backend.close_cached_weaviate_client()

if __name__ == "__main__":
    main()
