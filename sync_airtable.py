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
    try:
        # Initialize usage tracking
        usage_tracker = backend.get_usage_tracker()
        usage_tracker.reset_counters()
        
        weaviate_client = backend.connect_to_weaviate()
        openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        res = backend.ingest_airtable_to_weaviate(
            weaviate_client, openai_client, chunk_size=2000
        )
        usage_tracker.log_summary()
        print({"status": "ok", "result": res})
        sys.exit(0)
    except RuntimeError as e:
        if "API limit reached" in str(e):
            backend.get_usage_tracker().log_summary()
            print({"status": "error", "error": f"API limit reached: {e}"}, file=sys.stderr)
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print({"status": "error", "error": str(e)}, file=sys.stderr)
        sys.exit(1)
    finally:
        if weaviate_client and getattr(weaviate_client, "is_connected", lambda: False)():
            weaviate_client.close()
