"""
Cron-friendly script to sync Airtable → Weaviate and pre-generate reports.
This script should be run on a schedule (e.g., daily) by a cron job.

It performs two main tasks:
1.  Syncs the main data table from Airtable to the Weaviate vector database.
2.  Generates all standard analysis reports (Timeline, Conflict Report, etc.) and
    a summary for each key person, then saves the output to a separate
    'GeneratedReports' table in Airtable for instant retrieval by the app.
"""
import os
import sys
import backend
import openai
import traceback
from pyairtable import Table
from datetime import datetime
import base64

# --- CONFIGURATION ---
REPORT_MODE = os.environ.get("REPORT_MODE", "map-reduce")  # "map-reduce" or "simple"
GENERATE_PDF = True  # Set to False to skip PDF generation

def get_key_people():
    """Try to fetch key people from Airtable, otherwise fallback to static list."""
    try:
        table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], os.environ["AIRTABLE_TABLE_NAME"])
        people_records = table.all(formula="FIND('person', {EntityType})")
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
    # Fallback static list
    return [
        "Kim", "Diego", "Kim's family/friends", "YWCA Staff",
        "Heather Ulrich", "DSS/Youth Villages", "Diego's mom"
    ]

def generate_and_save_report(reports_table, name, generator_func):
    sanitized_name = backend.sanitize_name(name)
    print(f"\nGenerating report: '{name}' (Sanitized: '{sanitized_name}')...")
    try:
        report_content = generator_func()
        # If report_content is a streamed object, convert to string
        if not isinstance(report_content, str):
            report_content = "".join(str(c) for c in report_content)

        if "error" in report_content.lower() or "could not find" in report_content.lower():
            print(f"WARNING: Report for '{name}' generation resulted in a non-content message: {report_content}")

        pdf_bytes_b64 = ""
        if GENERATE_PDF:
            try:
                pdf_bytes = backend.create_pdf(report_content, summary=name)
                pdf_bytes_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                print(f"PDF generated for '{name}' ({len(pdf_bytes)} bytes).")
            except Exception as pdf_err:
                print(f"Failed to generate PDF for '{name}': {pdf_err}", file=sys.stderr)
        
        record_to_save = {
            "ReportName": sanitized_name,
            "Content": report_content,
            "LastGenerated": datetime.now().isoformat()
        }
        if GENERATE_PDF:
            record_to_save["PDF"] = pdf_bytes_b64
        try:
            result = reports_table.batch_upsert(
                [{"fields": record_to_save}], key_fields=["ReportName"]
            )
            if not result or not result[0].get("id"):
                raise RuntimeError(
                    f"Airtable batch_upsert returned unexpected result: {result}"
                )
        except Exception as upsert_err:
            print(
                f"ERROR: Failed to save report for '{name}' to Airtable: {upsert_err}",
                file=sys.stderr,
            )
            raise
        print(f"Successfully generated and saved report for '{name}'")
    except Exception as e:
        print(f"ERROR: Failed to generate or save report for '{name}'. Error: {e}\n{traceback.format_exc()}", file=sys.stderr)

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
    try:
        reports_table = Table(os.environ["AIRTABLE_API_KEY"], os.environ["AIRTABLE_BASE_ID"], "GeneratedReports")
        print("Connected to 'GeneratedReports' Airtable table.")
    except Exception as e:
        print(f"FATAL: Could not connect to 'GeneratedReports' table in Airtable. Aborting job. Error: {e}", file=sys.stderr)
        if weaviate_client and getattr(weaviate_client, "is_connected", lambda: False)():
            weaviate_client.close()
        sys.exit(1)

    # --- 4. Reports Definition (easy to modify/expand) ---
    reports_to_generate = {
        "Timeline": lambda: backend.generate_timeline(weaviate_client, openai_client, mode=REPORT_MODE),
        "Conflict Report": lambda: backend.generate_report("Conflict Report", weaviate_client, openai_client, mode=REPORT_MODE),
        "Legal Communication Summary": lambda: backend.generate_report("Legal Communication Summary", weaviate_client, openai_client, mode=REPORT_MODE),
    }

    key_people = get_key_people()
    for person in key_people:
        reports_to_generate[f"Summary for {person}"] = lambda p=person: backend.summarize_entity(p, weaviate_client, openai_client, mode=REPORT_MODE)

    # --- 5. Generate & Save Reports ---
    for name, generator_func in reports_to_generate.items():
        generate_and_save_report(reports_table, name, generator_func)

    print("\n--- Nightly job finished ---")
    if weaviate_client and getattr(weaviate_client, "is_connected", lambda: False)():
        weaviate_client.close()

if __name__ == "__main__":
    main()