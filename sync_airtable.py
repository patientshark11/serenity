"""Cron-friendly script to sync Airtable → Weaviate on a schedule.
Create a Render *Cron Job* service that runs:
    python sync_airtable.py
"""
import os
import sys
import backend
import openai

if __name__ == "__main__":
    weaviate_client = None
    exit_code = 0
    try:
        weaviate_client = backend.connect_to_weaviate()
        openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        res = backend.ingest_airtable_to_weaviate(weaviate_client, openai_client)
        print({"status": "ok", "result": res})
    except Exception as e:
        print({"status": "error", "error": str(e)}, file=sys.stderr)
        exit_code = 1
    finally:
        if weaviate_client and hasattr(weaviate_client, "close"):
            weaviate_client.close()
    sys.exit(exit_code)
