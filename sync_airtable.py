"""Cron-friendly script to sync Airtable → Weaviate on a schedule.
Create a Render *Cron Job* service that runs:  
    python sync_airtable.py
"""
import os
import sys
from streamlit_app import ingest_airtable_to_weaviate  # reuse the same function

if __name__ == "__main__":
    try:
        limit = int(os.environ.get("SYNC_LIMIT", "0")) or None
    except ValueError:
        limit = None
    res = ingest_airtable_to_weaviate(limit=limit)
    print({"status": "ok", **res})
    sys.exit(0)