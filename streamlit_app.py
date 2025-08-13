import streamlit as st
import os # Import the os library
import weaviate
import openai
from pyairtable import Table
import uuid

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Project Serenity - Diagnostic Mode",
    page_icon="🐛",
    layout="centered"
)

st.title("🐛 Diagnostic Mode")
st.warning("The application is currently in diagnostic mode. We are checking for environment variables.")

# --- 2. THE NEW DIAGNOSTIC FUNCTION ---
def run_diagnostics():
    """
    This function will directly check for environment variables and print results to the logs.
    """
    st.header("Checking Environment Variables...")
    
    # List of keys we expect to find
    required_keys = [
        "WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENAI_API_KEY",
        "AIRTABLE_API_KEY", "AIRTABLE_BASE_ID", "AIRTABLE_TABLE_NAME"
    ]
    
    all_found = True
    
    for key in required_keys:
        # We use os.environ.get() which is the standard Python way to read env vars
        value = os.environ.get(key)
        
        if value:
            # Found the key, display it masked for security
            st.success(f"✅ Found key: {key}")
            print(f"✅ Found key: {key}")
        else:
            # Did not find the key
            st.error(f"❌ MISSING key: {key}")
            print(f"❌ MISSING key: {key}")
            all_found = False

    if all_found:
        st.success("🎉 All required environment variables were found! The configuration seems correct.")
        st.info("You can now replace this diagnostic code with the original application code.")
    else:
        st.error("One or more environment variables are missing. Please check your Render dashboard settings.")
        st.info("The issue is with how variables are passed to the app, not the app code itself.")
    
    # Stop the app after diagnostics
    st.stop()


# --- RUN THE DIAGNOSTICS ---
run_diagnostics()


# The original application code is below this line but will not be executed because of st.stop()
# You can restore it after the diagnostics are complete.

# @st.cache_resource ... etc.
