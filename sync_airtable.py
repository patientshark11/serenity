"""Cron-friendly script to sync Airtable → Weaviate and pre-generate reports.
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
from pyairtable import Table
from datetime import datetime

def main():
    print("--- Starting nightly job: Data Sync and Report Generation ---")

    # --- 1. Initialize Clients ---
    weaviate_client = None  # Initialize to None
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
        # Note: ingest_airtable_to_weaviate is in backend.py
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
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
        sys.exit(1)

    reports_to_generate = {
        "Timeline": lambda: backend.generate_timeline(weaviate_client, openai_client),
        "Conflict Report": lambda: backend.generate_report("Conflict Report", weaviate_client, openai_client),
        "Legal Communication Summary": lambda: backend.generate_report("Legal Communication Summary", weaviate_client, openai_client),
    }

    key_people = ["Kim", "Diego", "Kim's family/friends", "YWCA Staff", "Heather Ulrich", "DSS/Youth Villages", "Diego's mom"]
    for person in key_people:
        reports_to_generate[f"Summary for {person}"] = lambda p=person: backend.summarize_entity(p, weaviate_client, openai_client)

    for name, generator_func in reports_to_generate.items():
        sanitized_name = backend.sanitize_name(name)
        print(f"Generating report: '{name}' (Sanitized: '{sanitized_name}')...")
        try:
            response_stream = generator_func()
            if isinstance(response_stream, str):
                report_content = response_stream
            else:
                report_content = "".join(c for c in response_stream)
            
            if "error" in report_content.lower() or "could not find" in report_content.lower():
                 print(f"WARNING: Report for '{name}' generation resulted in a non-content message: {report_content}")

            record_to_save = {
                "ReportName": sanitized_name,
                "Content": report_content,
                "LastGenerated": datetime.now().isoformat()
            }
            reports_table.upsert([record_to_save], key_fields=["ReportName"])
            print(f"Successfully generated and saved report for '{name}'")
        except Exception as e:
            print(f"ERROR: Failed to generate or save report for '{name}'. Error: {e}", file=sys.stderr)
            
    print("\n--- Nightly job finished ---")
    if weaviate_client and weaviate_client.is_connected():
        weaviate_client.close()

if __name__ == "__main__":
    main()
