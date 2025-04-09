import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback
import datetime  # For timestamping logs
import numpy as np # For handling potential inf values

# --- Constants ---
# Define the configuration for calculating summary statistics
# Based on the provided codebook
STATS_CONFIG = [
    # --- Demographics ---
    {'csv_col': 'Total population', 'output_key': 'Total Population (Sum)', 'agg': 'sum'},
    {'csv_col': 'Percent Hispanic or Latino', 'output_key': 'Average Percent Hispanic or Latino', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent Black or African American alone', 'output_key': 'Average Percent Black or African American alone', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent White', 'output_key': 'Average Percent White', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent American Indian / Alaska Native', 'output_key': 'Average Percent American Indian/Alaska Native', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent Asian', 'output_key': 'Average Percent Asian', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent Native Hawaiian or Pacific', 'output_key': 'Average Percent Native Hawaiian/Pacific Islander', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent two or more races', 'output_key': 'Average Percent Two or More Races', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent other races', 'output_key': 'Average Percent Other Races', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent age under 10', 'output_key': 'Average Percent Under Age 10', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent age over 64', 'output_key': 'Average Percent Over Age 64', 'agg': 'average', 'multiplier': 100},

    # --- Disadvantage & Income ---
    {'csv_col': 'Identified as disadvantaged', 'output_key': 'Number of Disadvantaged Tracts (Identified)', 'agg': 'count_true'},
    {'csv_col': 'Is low income?', 'output_key': 'Number of Low Income Tracts', 'agg': 'count_true'},
    {'csv_col': 'Adjusted percent of individuals below 200% Federal Poverty Line', 'output_key': 'Average Percent Below 200% FPL', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent of individuals < 100% Federal Poverty Line', 'output_key': 'Average Percent Below 100% FPL', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Unemployment (percent)', 'output_key': 'Average Unemployment Rate (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Median household income as a percent of area median income', 'output_key': 'Average Median Household Income (% of AMI)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Linguistic isolation (percent)', 'output_key': 'Average Linguistic Isolation Rate (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Percent individuals age 25 or over with less than high school degree', 'output_key': 'Average Percent Less Than High School Degree', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Total threshold criteria exceeded', 'output_key': 'Average Total Threshold Criteria Exceeded (per tract)', 'agg': 'average'},
    {'csv_col': 'Total categories exceeded', 'output_key': 'Average Total Categories Exceeded (per tract)', 'agg': 'average'},
    {'csv_col': 'Identified as disadvantaged based on neighbors and relaxed low income threshold only', 'output_key': 'Count Tracts Disadvantaged by Neighbors/Relaxed Income', 'agg': 'count_true'},
    {'csv_col': 'Identified as disadvantaged due to tribal overlap', 'output_key': 'Count Tracts Disadvantaged by Tribal Overlap', 'agg': 'count_true'},
    {'csv_col': 'Identified as disadvantaged solely due to status in v1.0 (grandfathered)', 'output_key': 'Count Tracts Disadvantaged (Grandfathered)', 'agg': 'count_true'},

    # --- Health ---
    {'csv_col': 'Current asthma among adults aged greater than or equal to 18 years', 'output_key': 'Average Percent With Asthma', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Diagnosed diabetes among adults aged greater than or equal to 18 years', 'output_key': 'Average Percent With Diabetes', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Coronary heart disease among adults aged greater than or equal to 18 years', 'output_key': 'Average Heart Disease Rate (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Life expectancy (years)', 'output_key': 'Average Life Expectancy (Years)', 'agg': 'average'},

    # --- Environment, Climate, Energy ---
    {'csv_col': 'Energy burden', 'output_key': 'Average Energy Burden (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'PM2.5 in the air', 'output_key': 'Average PM2.5 in the Air', 'agg': 'average'},
    {'csv_col': 'Diesel particulate matter exposure', 'output_key': 'Average Diesel Particulate Matter Exposure', 'agg': 'average'},
    {'csv_col': 'Traffic proximity and volume', 'output_key': 'Average Traffic Proximity/Volume Score', 'agg': 'average'},
    {'csv_col': 'Share of properties at risk of flood in 30 years', 'output_key': 'Average Share Properties at Flood Risk (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Share of properties at risk of fire in 30 years', 'output_key': 'Average Share Properties at Wildfire Risk (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Expected agricultural loss rate (Natural Hazards Risk Index)', 'output_key': 'Average Expected Agricultural Loss Rate', 'agg': 'average'}, # Rate might need scaling? Check units/meaning.
    {'csv_col': 'Expected building loss rate (Natural Hazards Risk Index)', 'output_key': 'Average Expected Building Loss Rate', 'agg': 'average'},
    {'csv_col': 'Expected population loss rate (Natural Hazards Risk Index)', 'output_key': 'Average Expected Population Loss Rate', 'agg': 'average'},
    {'csv_col': 'Proximity to hazardous waste sites', 'output_key': 'Average Proximity to Hazardous Waste Sites', 'agg': 'average'},
    {'csv_col': 'Proximity to NPL (Superfund) sites', 'output_key': 'Average Proximity to NPL (Superfund) Sites', 'agg': 'average'},
    {'csv_col': 'Proximity to Risk Management Plan (RMP) facilities', 'output_key': 'Average Proximity to RMP Facilities', 'agg': 'average'},
    {'csv_col': 'Wastewater discharge', 'output_key': 'Average Wastewater Discharge Score', 'agg': 'average'},
    {'csv_col': 'Leaky underground storage tanks', 'output_key': 'Average Leaky Underground Storage Tank Score', 'agg': 'average'},
    {'csv_col': 'Share of the tract\'s land area that is covered by impervious surface or cropland as a percent', 'output_key': 'Average % Land Impervious/Cropland', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Is there at least one Formerly Used Defense Site (FUDS) in the tract?', 'output_key': 'Count Tracts with FUDS', 'agg': 'count_true'},
    {'csv_col': 'Is there at least one abandoned mine in this census tract?', 'output_key': 'Count Tracts with Abandoned Mines', 'agg': 'count_true'},

    # --- Housing & Transportation ---
    {'csv_col': 'Housing burden (percent)', 'output_key': 'Average Housing Burden (%)', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'DOT Travel Barriers Score', 'output_key': 'Average DOT Travel Barriers Score', 'agg': 'average'},
    {'csv_col': 'Percent pre-1960s housing (lead paint indicator)', 'output_key': 'Average Percent Pre-1960s Housing', 'agg': 'average', 'multiplier': 100},
    {'csv_col': 'Median value ($) of owner-occupied housing units', 'output_key': 'Average Median Home Value ($)', 'agg': 'average'},
    {'csv_col': 'Share of homes with no kitchen or indoor plumbing (percent)', 'output_key': 'Average % Homes Lacking Kitchen/Plumbing', 'agg': 'average', 'multiplier': 100}, # Raw is likely a float like 0.01, needs *100

    # --- Threshold Counts (Boolean Fields) ---
    {'csv_col': 'Greater than or equal to the 90th percentile for energy burden and is low income?', 'output_key': 'Count Tracts Exceeding Energy Burden Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for PM2.5 exposure and is low income?', 'output_key': 'Count Tracts Exceeding PM2.5 Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for diesel particulate matter and is low income?', 'output_key': 'Count Tracts Exceeding Diesel PM Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for traffic proximity and is low income?', 'output_key': 'Count Tracts Exceeding Traffic Proximity Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for housing burden and is low income?', 'output_key': 'Count Tracts Exceeding Housing Burden Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for share of properties at risk of flood in 30 years and is low income?', 'output_key': 'Count Tracts Exceeding Flood Risk Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for share of properties at risk of fire in 30 years and is low income?', 'output_key': 'Count Tracts Exceeding Wildfire Risk Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for asthma and is low income?', 'output_key': 'Count Tracts Exceeding Asthma Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for diabetes and is low income?', 'output_key': 'Count Tracts Exceeding Diabetes Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for heart disease and is low income?', 'output_key': 'Count Tracts Exceeding Heart Disease Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for low life expectancy and is low income?', 'output_key': 'Count Tracts Exceeding Low Life Expectancy Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for unemployment and has low HS attainment?', 'output_key': 'Count Tracts Exceeding Unemployment Threshold & Low HS Attainment', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for households in linguistic isolation and has low HS attainment?', 'output_key': 'Count Tracts Exceeding Linguistic Isolation Threshold & Low HS Attainment', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for households at or below 100% federal poverty level and has low HS attainment?', 'output_key': 'Count Tracts Exceeding Poverty Threshold & Low HS Attainment', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for proximity to hazardous waste facilities and is low income?', 'output_key': 'Count Tracts Exceeding Haz Waste Proximity Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for proximity to superfund sites and is low income?', 'output_key': 'Count Tracts Exceeding Superfund Proximity Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for proximity to RMP sites and is low income?', 'output_key': 'Count Tracts Exceeding RMP Proximity Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for wastewater discharge and is low income?', 'output_key': 'Count Tracts Exceeding Wastewater Threshold & Low Income', 'agg': 'count_true'},
    {'csv_col': 'Greater than or equal to the 90th percentile for leaky underground storage tanks and is low income?', 'output_key': 'Count Tracts Exceeding Leaky UST Threshold & Low Income', 'agg': 'count_true'},
    # Add more boolean threshold columns from the codebook here if needed...
]


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
    # Use absolute path for logs directory based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')

    # Create logs directory if it doesn't exist
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"Created logs directory at {logs_dir}")
    except Exception as e:
        print(f"ERROR: Could not create logs directory at {logs_dir}: {e}")
        return # Cannot log if directory creation fails

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
        "query_analysis": query_analysis, # Store the dict directly
        "response": response,
        "error": error
    }

    # Append to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            # Ensure analysis is serializable (it should be if it's from JSON)
            f.write(json.dumps(log_entry, default=str) + '\n') # Use default=str for safety
        print(f"Query logged to {log_file}")
    except Exception as e:
        print(f"ERROR: Failed to write to log file {log_file}: {e}")
        # Attempt to log the error itself to console
        try:
            print(f"Failed log entry data: {json.dumps(log_entry, default=str)}")
        except Exception as inner_e:
            print(f"Could not even serialize the failed log entry: {inner_e}")


app = Flask(__name__)

# --- Data Loading ---
df = None
try:
    # Use absolute path for CSV based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '2.0-communities.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        # Data Cleaning - Apply to all relevant columns at once
        str_cols = ['State/Territory', 'County Name', 'Census tract 2010 ID']
        for col in str_cols:
            if col in df.columns:
                # Ensure it's string type before using .str accessor
                df[col] = df[col].astype(str).str.strip().str.lower() # Standardize to lower for matching
            else:
                print(f"Warning: Expected string column '{col}' not found in CSV.")

        # Convert potential boolean-like columns more robustly
        bool_like_cols = [config['csv_col'] for config in STATS_CONFIG if config['agg'] == 'count_true']
        for col in bool_like_cols:
             if col in df.columns:
                 # Map common true/false strings/numbers to boolean, treat errors as False
                 df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else 'false')
                 df[col] = df[col].map({'true': True, 't': True, 'yes': True, 'y': True, '1': True,
                                        'false': False, 'f': False, 'no': False, 'n': False, '0': False, '':False}).fillna(False)
                 df[col] = df[col].astype(bool)

        # Attempt to convert numeric columns, coercing errors to NaN
        numeric_cols = [config['csv_col'] for config in STATS_CONFIG if config['agg'] in ['average', 'sum']]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace potential infinity values introduced by calculations/conversions
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print(f"DataFrame loaded successfully with {len(df)} records from {csv_path}.")
        print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        # print(df.info(verbose=True)) # Optional: Print detailed info for debugging
    else:
        print(f"ERROR: CSV file not found at {csv_path}. Application cannot function without data.")
        df = pd.DataFrame() # Create empty DF to prevent crashes later
except FileNotFoundError:
     print(f"CRITICAL Error: CSV file not found at {csv_path}. Please ensure the file exists.")
     df = pd.DataFrame()
except Exception as e:
    print(f"CRITICAL Error loading or processing DataFrame: {e}")
    print(traceback.format_exc())
    df = pd.DataFrame()

# --- OpenAI Client Initialization ---
client = None
try:
    # api_key already loaded and checked above
    if not api_key:
        raise ValueError("OpenAI API key not found or failed to load.")
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"CRITICAL Error initializing OpenAI client: {e}")
    client = None # Ensure client is None if initialization fails

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    # Check if data loading failed and pass status to template (optional)
    data_loaded = df is not None and not df.empty
    return render_template('index.html', data_loaded=data_loaded)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the analysis request:
    1. Filters data based on user input (state, county, tract).
    2. Dynamically calculates summary statistics based on STATS_CONFIG.
    3. Uses Step 1 LLM call to analyze user query intent and map to available stats.
    4. Uses Step 2 LLM call to generate a concise analysis based on query intent and stats.
    """
    # Initial error checking
    if df is None or df.empty:
        error_msg = "Dataset not loaded or is empty on the server. Cannot perform analysis."
        # Log attempt even if data isn't loaded
        log_query_response(request.get_json() or {}, {}, error_msg, 0, error=error_msg)
        return jsonify({"error": error_msg}), 500

    if not client:
        error_msg = "OpenAI client not configured on the server. Cannot perform analysis."
        log_query_response(request.get_json() or {}, {}, error_msg, 0, error=error_msg)
        return jsonify({"error": error_msg}), 500

    try:
        request_data = request.get_json()
        if not request_data:
             return jsonify({"error": "Invalid request format. Expected JSON."}), 400

        print(f"\n--- New Request ---")
        print(f"Received request data: {request_data}")

        state = request_data.get('state', '').strip().lower()
        county = request_data.get('county', '').strip().lower()
        # Keep tract ID as string, strip whitespace, but don't lowercase (often numeric/specific case)
        tract_id = request_data.get('tract_id', '').strip()
        user_query = request_data.get('query', '').strip()

        if not user_query:
            error_msg = "User query cannot be empty."
            log_query_response(request_data, {}, error_msg, 0, error=error_msg)
            return jsonify({"error": error_msg}), 400

        # --- Filtering Logic ---
        print(f"Starting filtering with state='{state}', county='{county}', tract_id='{tract_id}'")
        filtered_df = df.copy() # Start with a copy of the full dataframe

        # Build filter step-by-step
        if state:
            # Ensure comparison column is also lowercase
            filtered_df = filtered_df[filtered_df['State/Territory'] == state]
            print(f"After state filter ('{state}'): {len(filtered_df)} records")
            if filtered_df.empty:
                available_states = df['State/Territory'].unique().tolist()
                msg = f"No data found for state: '{request_data.get('state', '')}'. Check spelling. Available states might include: {', '.join(available_states[:10])}..."
                log_query_response(request_data, {}, msg, 0)
                return jsonify({"analysis": msg}) # Return message, not error

        if county:
             # Ensure comparison column is also lowercase
            filtered_df = filtered_df[filtered_df['County Name'] == county]
            print(f"After county filter ('{county}'): {len(filtered_df)} records")
            if filtered_df.empty:
                state_context = f" in state '{state}'" if state else " in the dataset"
                # Get counties available *within the current filter level*
                counties_in_scope = df['County Name'][df['State/Territory'] == state].unique().tolist() if state else df['County Name'].unique().tolist()
                county_list_str = f" Available counties{state_context} might include: {', '.join(counties_in_scope[:10])}..." if counties_in_scope else ""
                msg = f"No data found for county: '{request_data.get('county', '')}'{state_context}. Check spelling.{county_list_str}"
                log_query_response(request_data, {}, msg, 0)
                return jsonify({"analysis": msg})

        if tract_id:
            # Compare with the cleaned tract ID column (already string and stripped)
            filtered_df = filtered_df[filtered_df['Census tract 2010 ID'] == tract_id]
            print(f"After tract filter ('{tract_id}'): {len(filtered_df)} records")
            # No special message if tract filter yields empty, main message handles it

        num_tracts_found = len(filtered_df)
        print(f"Final filter resulted in {num_tracts_found} tracts.")

        if filtered_df.empty:
            msg = "No data found matching the specified combination of State, County, and/or Tract ID."
            log_query_response(request_data, {}, msg, 0)
            return jsonify({"analysis": msg})

        # --- Dynamic Data Summarization ---
        print(f"Calculating summary stats for {num_tracts_found} tracts based on STATS_CONFIG.")
        summary_stats = {"Number of Census Tracts Found": num_tracts_found}

        for config in STATS_CONFIG:
            col = config['csv_col']
            key = config['output_key']
            agg_type = config['agg']
            multiplier = config.get('multiplier', 1)

            if col in filtered_df.columns:
                # Select the column data for calculation
                column_data = filtered_df[col]

                if agg_type == 'sum':
                    # Ensure numeric, drop NaN before sum
                    numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
                    if not numeric_data.empty:
                        summary_stats[key] = int(numeric_data.sum()) # Use int for counts/population
                    else:
                        print(f"Debug: Column '{col}' for sum had no valid numeric data after filtering/dropna.")

                elif agg_type == 'average':
                    numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
                    if not numeric_data.empty:
                        avg = numeric_data.mean()
                        scaled_avg = avg # Default to the calculated average

                        # Check if a percentage multiplier is configured
                        if multiplier == 100:
                            # Heuristic: If avg is already > 1.5 (likely already a percentage), don't multiply.
                            # Adjust threshold (e.g., 1.0 or 2.0) if needed based on data characteristics.
                            if avg <= 1.5:
                                scaled_avg = avg * 100 # Assume it's a fraction, convert to percentage
                            # else: Keep scaled_avg = avg (it's already likely a percentage)

                        elif multiplier != 1: # Handle potential future non-100 multipliers
                            scaled_avg = avg * multiplier

                        summary_stats[key] = round(scaled_avg, 1) # Round the final value
                    else:
                        print(f"Debug: Column '{col}' for average had no valid numeric data after filtering/dropna.")

                elif agg_type == 'count_true':
                    # Assumes column is already boolean or mapped to boolean during load
                    # Fill potential NaNs introduced *after* load with False before summing
                    summary_stats[key] = int(column_data.fillna(False).astype(bool).sum())

                # Can add other agg types like 'median', 'stddev' here if needed
            else:
                # Only print warning if column is *expected* but missing
                # print(f"Warning: Column '{col}' configured in STATS_CONFIG not found in DataFrame.")
                pass # Don't warn for every single unused config item

        # Filter out stats that couldn't be calculated (value is None or key wasn't added)
        # Start with the guaranteed tract count
        calculated_summary_stats = {"Number of Census Tracts Found": num_tracts_found}
        # Add other successfully calculated stats
        for k, v in summary_stats.items():
            if k != "Number of Census Tracts Found" and pd.notna(v): # Check for non-null/NaN values
                 calculated_summary_stats[k] = v

        # print(f"Calculated Summary Stats: {json.dumps(calculated_summary_stats, indent=2)}") # Debug print

        if len(calculated_summary_stats) <= 1: # Only contains the tract count
            msg = "Could not calculate any summary statistics for the selected area (beyond the tract count)."
            log_query_response(request_data, {}, msg, num_tracts_found)
            return jsonify({"analysis": msg})

        # --- STEP 1: Analyze the User Query with an LLM ---
        print("--- Starting Step 1: Query Analysis ---")
        available_data_points = list(calculated_summary_stats.keys()) # Use keys from successfully calculated stats

        # Updated prompt with more mappings based on STATS_CONFIG output keys
        step1_system_prompt = f"""You are an expert query analysis assistant for US census and environmental justice data. Analyze the user query to understand the core intent (action) and identify EXACTLY which of the strictly defined 'Available data points' are relevant.
        Available data points: {', '.join(available_data_points)}

        Mapping Guidelines & Examples (Use the EXACT key names provided):
        - **Demographics:** Map 'population'/'how many people'/'amount of people' to 'Total Population (Sum)'. Map ethnicity/race terms ('hispanic', 'latino', 'black', 'african american', 'white', 'asian', 'native american' etc.) to the corresponding 'Average Percent...' key. Map 'age' to 'Average Percent Under Age 10' or 'Average Percent Over Age 64'.
        - **Disadvantage/Income:** Map 'disadvantaged'/'burdened communities'/'EJ communities' to 'Number of Disadvantaged Tracts (Identified)'. Map 'low income'/'poverty'/'Federal Poverty Line'/'FPL' to 'Number of Low Income Tracts', 'Average Percent Below 200% FPL', or 'Average Percent Below 100% FPL'. Map 'unemployment'/'jobs' to 'Average Unemployment Rate (%)'. Map 'linguistic isolation' to 'Average Linguistic Isolation Rate (%)'. Map 'income level'/'area median income'/'AMI' to 'Average Median Household Income (% of AMI)'. Map 'education'/'high school' to 'Average Percent Less Than High School Degree'. Map 'thresholds exceeded'/'criteria met' to 'Average Total Threshold Criteria Exceeded (per tract)'.
        - **Health:** Map 'asthma' to 'Average Percent With Asthma', 'diabetes' to 'Average Percent With Diabetes', 'heart disease'/'cardiovascular' to 'Average Heart Disease Rate (%)', 'life expectancy' to 'Average Life Expectancy (Years)'.
        - **Energy & Air Quality:** Map 'energy burden'/'energy costs' to 'Average Energy Burden (%)'. Map 'air quality'/'pollution'/'PM2.5'/'particulate' to 'Average PM2.5 in the Air'. Map 'diesel'/'diesel pm' to 'Average Diesel Particulate Matter Exposure'.
        - **Climate Risk:** Map 'flood'/'flooding' to 'Average Share Properties at Flood Risk (%)'. Map 'fire'/'wildfire' to 'Average Share Properties at Wildfire Risk (%)'. Map 'agricultural loss' to 'Average Expected Agricultural Loss Rate'. Map 'building loss' to 'Average Expected Building Loss Rate'.
        - **Housing & Transportation:** Map 'housing cost'/'housing burden' to 'Average Housing Burden (%)'. Map 'transportation'/'transit'/'travel barriers' to 'Average DOT Travel Barriers Score'. Map 'traffic'/'road proximity' to 'Average Traffic Proximity/Volume Score'. Map 'old houses'/'lead paint risk' to 'Average Percent Pre-1960s Housing'. Map 'home value' to 'Average Median Home Value ($)'. Map 'plumbing'/'kitchen' to 'Average % Homes Lacking Kitchen/Plumbing'.
        - **Pollution/Waste:** Map 'hazardous waste' to 'Average Proximity to Hazardous Waste Sites'. Map 'superfund'/'NPL' to 'Average Proximity to NPL (Superfund) Sites'. Map 'RMP sites' to 'Average Proximity to RMP Facilities'. Map 'wastewater' to 'Average Wastewater Discharge Score'. Map 'leaky tanks'/'UST' to 'Average Leaky Underground Storage Tank Score'. Map 'FUDS'/'defense sites' to 'Count Tracts with FUDS'. Map 'mines' to 'Count Tracts with Abandoned Mines'.
        - **Threshold Counts:** Map queries like "how many tracts exceed energy threshold?" to the relevant 'Count Tracts Exceeding... Threshold...' key.
        - **General:** Map 'summarize'/'describe'/'overview'/'composition'/'makeup'/'size' to the action 'summarize'. If no specific field is mentioned, 'relevant_fields' can include primary fields like population, disadvantage count, and key demographics.

Output Instructions:
- Output ONLY a valid JSON object. Do not include any text before or after the JSON.
- Use the following JSON structure:
{{
  "action": "<identified action e.g., summarize, list_value, compare, check_feasibility, unknown>",
  "relevant_fields": ["<list of EXACT key names from Available data points that directly match the query's core concepts>"],
  "unmatched_query_parts": ["<list ONLY substantive concepts from the query that represent data types genuinely unavailable in the provided list>"]
}}
- **Crucially:** Do NOT put common words (like 'how many', 'percentage of', 'people', 'rate', 'level', 'amount', 'of', 'in', 'as', 'opposed', 'to', 'have', 'average') in 'unmatched_query_parts' if the main data concept *was* successfully mapped to a 'relevant_field'. Only list concepts representing *data types* that are genuinely unavailable (e.g., 'crime rates', 'specific company names', 'future projections').
- If the query asks for something clearly unavailable, set action to 'check_feasibility' and list the concept in 'unmatched_query_parts'. If unsure, lean towards 'summarize' or 'list_value' if some relevant fields are found.
"""
        step1_user_content = f"Analyze this user query: \"{user_query}\""

        query_analysis = {"action": "summarize", "relevant_fields": [], "unmatched_query_parts": ["Query analysis failed (default)"]} # Default
        try:
            step1_completion = client.chat.completions.create(
                model="gpt-3.5-turbo", # Or potentially gpt-4-turbo if complexity requires
                messages=[
                    {"role": "system", "content": step1_system_prompt},
                    {"role": "user", "content": step1_user_content}
                ],
                temperature=0.0, # Low temp for deterministic mapping
                max_tokens=250,  # Increased slightly for potentially longer lists
                response_format={"type": "json_object"}
            )
            step1_result_str = step1_completion.choices[0].message.content
            print(f"Step 1 Raw Result: {step1_result_str}")
            query_analysis = json.loads(step1_result_str) # Parse JSON response
            print(f"Step 1 Parsed Analysis: {query_analysis}")

        except json.JSONDecodeError as json_e:
             print(f"Error parsing Step 1 JSON response: {json_e}")
             print(f"Raw response was: {step1_result_str}")
             # Keep the default query_analysis
        except Exception as e:
            print(f"Error during Step 1 (Query Analysis) API call: {e}")
            # Keep the default query_analysis

        # --- Check Feasibility / Prepare for Step 2 ---
        action = query_analysis.get("action", "unknown")
        unmatched = query_analysis.get("unmatched_query_parts", [])
        relevant = query_analysis.get("relevant_fields", [])

        # Refined check: If action suggests infeasibility OR if it's unknown/generic AND there are significant unmatched parts
        if action == "check_feasibility" or (action in ["unknown", "summarize"] and unmatched and not relevant):
            missing_parts_str = f"'{', '.join(unmatched)}'" if unmatched else "the specific topic"
            analysis_result = f"Specific data for {missing_parts_str} needed for your query is not available in the calculated summary statistics for this area."
            if not relevant and unmatched: # If we couldn't match *anything* relevant
                 analysis_result += " Available data includes topics like population, income, health, environment, and housing metrics."

            print("Step 1 indicated query requires unavailable data or analysis failed. Returning message.")
            log_query_response(request_data, query_analysis, analysis_result, num_tracts_found)
            return jsonify({"analysis": analysis_result})

        # --- STEP 2: Perform Data Analysis ---
        print("--- Starting Step 2: Data Analysis ---")

        # Decide which stats to send: Filter calculated stats based on Step 1's relevant fields
        final_summary_stats_to_send = {}
        if relevant:
            # Filter the relevant fields identified by Step 1 to ensure they were actually calculated
            filtered_relevant = [f for f in relevant if f in calculated_summary_stats]
            if filtered_relevant:
                 # Include relevant fields + always include the tract count
                final_summary_stats_to_send = {k: v for k, v in calculated_summary_stats.items() if k in filtered_relevant or k == "Number of Census Tracts Found"}
                print(f"Sending {len(final_summary_stats_to_send)} relevant stats to Step 2.")
            else:
                # Relevant fields were identified, but *none* could be calculated for this specific filtered data
                # Send all calculated stats instead as a fallback for a general summary
                final_summary_stats_to_send = calculated_summary_stats
                print("Warning: Relevant fields identified, but none were calculable. Sending all available stats for general summary.")
        else:
            # No specific fields identified by Step 1 (e.g., generic "summarize" query) - send all calculated stats
            final_summary_stats_to_send = calculated_summary_stats
            print("No specific relevant fields identified by Step 1. Sending all available stats.")


        if not final_summary_stats_to_send or len(final_summary_stats_to_send) <= 1:
             msg = "Could not gather sufficient summary statistics for the selected area to generate an analysis."
             log_query_response(request_data, query_analysis, msg, num_tracts_found)
             return jsonify({"analysis": msg})


        final_summary_data_string = json.dumps(final_summary_stats_to_send, indent=2)

        # Step 2 System Prompt (Remains largely the same)
        step2_system_prompt = """You are an AI assistant analyzing US Census tract data summaries.
- Provide a concise summary addressing the user's query based *only* on the provided summary statistics for the specified location.
- Do NOT list individual census tracts.
- State results directly and naturally. Avoid repetitive introductory phrases like "Based on the data..." or "The data shows...". Frame the answer as if you are directly informing the user about the area.
- If the query asks about something specific that is *not* present in the summary statistics provided to you (even if Step 1 thought it might be relevant), clearly state "Specific data for [topic] is not available in this summary." Do not invent data.
- Keep the response focused on the overall area described by the statistics. If only one tract was found, you can state the values directly. If multiple tracts were found, refer to averages or counts as appropriate (e.g., "The average unemployment rate is X%", "Y tracts were identified as disadvantaged").
- Use short paragraphs or bullet points for clarity.
- Mention the number of census tracts the analysis is based on if it's greater than 1 (e.g., "Across the Z census tracts found...").
        """

        # Determine location name for the prompt
        location_parts = [request_data.get('tract_id',''), request_data.get('county',''), request_data.get('state','')]
        location_name = next((part for part in location_parts if part), "Selected Area") # First non-empty part or default
        full_location_desc = f"{request_data.get('tract_id', '') or 'Area'} in {request_data.get('county', '') or ''}{', ' if request_data.get('county','') and request_data.get('state','') else ''}{request_data.get('state', '') or 'Selected Region'}".replace(" in ,", " in").strip()
        if full_location_desc.startswith("Area in"): full_location_desc = location_name # Clean up if only state/county missing

        action_verb = query_analysis.get("action", "summarize") # Get action from Step 1

        # Step 2 User Content
        step2_user_content = f"""
Analysis Request:
Location Description: {full_location_desc}
Number of Census Tracts Found: {num_tracts_found}
User Query: "{user_query}"
Identified Action from Query Analysis: {action_verb}
Identified Relevant Fields by Query Analysis: {json.dumps(relevant)}

Summary Statistics Provided for Analysis (Use ONLY these values):
{final_summary_data_string}

---
Task: Provide a concise response answering the User Query for the Location based *strictly* on the statistics provided. Address the 'Identified Action'. Do not list individual tracts. Mention the number of tracts if greater than 1. If the provided statistics don't contain the specific information needed for the query (even if listed in 'Relevant Fields'), state that clearly.
"""

        print(f"Sending summarized prompt for Step 2 to OpenAI (User Content Length: {len(step2_user_content)} chars)")
        # print(f"Step 2 User Content Preview:\n{step2_user_content[:500]}...") # Optional preview

        step2_completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or gpt-4-turbo for potentially better nuance
            messages=[
                {"role": "system", "content": step2_system_prompt},
                {"role": "user", "content": step2_user_content}
            ],
            temperature=0.3, # Slightly higher temp for more natural language
            max_tokens=400   # Allow for slightly longer summaries if needed
        )

        analysis_result = step2_completion.choices[0].message.content.strip()
        print("Received analysis from OpenAI (Step 2).")
        # print(f"Step 2 Result: {analysis_result}") # Debug print

        # Log the successful interaction
        log_query_response(
            request_data,
            query_analysis, # Log the structured analysis from Step 1
            analysis_result,
            num_tracts_found
        )

        return jsonify({"analysis": analysis_result})

    except Exception as e:
        print(f"--- ERROR in /analyze route ---")
        # Print traceback to console for debugging
        traceback.print_exc()
        error_msg = f"An unexpected server error occurred during analysis: {str(e)}"
        # Log the error, including traceback if possible
        log_query_response(request.get_json() or {}, {}, "", 0, error=f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    # Check if crucial components are loaded before starting
    if df is None or df.empty:
         print("CRITICAL: DataFrame failed to load. Flask app starting, but /analyze will fail.")
    if client is None:
         print("CRITICAL: OpenAI client failed to initialize. Flask app starting, but /analyze will fail.")

    print("Starting Flask server...")
    # Consider security implications of debug=True in production
    # Use host='0.0.0.0' to make accessible on network if needed (e.g., Docker)
    app.run(debug=True, host='127.0.0.1', port=5000)