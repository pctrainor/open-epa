import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback
import datetime  # For timestamping logs

# Clear existing API key from environment if present
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]
    print("Cleared existing API key from environment")

# Load environment variables with override
load_dotenv(override=True)

# Verify key was loaded (safely)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"API key loaded successfully: {api_key[:5]}...{api_key[-4:]} (length: {len(api_key)})")
else:
    print("WARNING: No API key found in .env file")

# Logging function for queries and responses
def log_query_response(query_data, query_analysis, response, num_records, error=None):
    """Log query and response data to a file for training purposes"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory at {logs_dir}")
    
    # Use current date in filename for easier management
    log_file = os.path.join(logs_dir, f'query_logs_{datetime.datetime.now().strftime("%Y%m%d")}.jsonl')
    
    # Create a log entry as JSON
    log_entry = {
        "timestamp": timestamp,
        "state": query_data.get('state', ''),
        "county": query_data.get('county', ''),
        "tract_id": query_data.get('tract_id', ''),
        "user_query": query_data.get('query', ''),
        "records_found": num_records,
        "query_analysis": query_analysis,
        "response": response,
        "error": error
    }
    
    # Append to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"Query logged to {log_file}")
    except Exception as e:
        print(f"ERROR: Failed to write to log file: {e}")

app = Flask(__name__)