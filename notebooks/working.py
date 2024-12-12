import requests
import json
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv  # Add python-dotenv for better env management
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

load_dotenv()
BEARER_TOKEN = os.getenv('JIRA_BEARER_TOKEN')
JIRA_URL = os.getenv('JIRA_URL')
 
session = requests.Session()
session.headers.update({
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Content-Type': 'application/json'  # Ensure Content-Type is set to application/json
    })

def login_to_jira():
    response = session.get(JIRA_URL + '/rest/api/2/user?username=SVC_IT_Automation')
    
    if response.status_code == 200:
        print("Successfully logged in to Jira")
        #print(response.json())
        return True
    else:
        print("Failed to log in to Jira")
        print(response.status_code, response.text)
        return False


# Login to JIRA
if not login_to_jira():
    raise Exception("Failed to log in to JIRA")

# Data retrieval variables
start_at = 0
max_results = 500
total = None
fields = ['key', 'project', 'summary', 'description', 'assignee', 'httpUrl', 'customfield_19900', 'customfield_15404', 'customfield_14201']
issues_list = []

def fetch_batch(start_at, max_results, fields, jira_url, session):
    search_data = {
        'jql': "project='29-IT Service Desk' AND status=closed AND created >= startOfYear(-1y)",
        'fields': fields,
        'startAt': start_at,
        'maxResults': max_results
    }
    
    try:
        search_response = session.post(
            f"{jira_url}/rest/api/2/search",
            data=json.dumps(search_data),
            timeout=300
        )
        
        if search_response.status_code == 200:
            return search_response.json().get('issues', [])
        elif search_response.status_code == 401:
            if login_to_jira():
                # Retry once after re-authentication
                return fetch_batch(start_at, max_results, fields, jira_url, session)
        return []
    except Exception as e:
        print(f"Error fetching batch at {start_at}: {str(e)}")
        return []

# Get total count first
initial_search = {
    'jql': "project='29-IT Service Desk' AND status=closed AND created >= startOfYear(-1y)",
    'fields': ['key'],
    'maxResults': 1
}
total_response = session.post(
    f"{JIRA_URL}/rest/api/2/search",
    data=json.dumps(initial_search)
)
total = total_response.json()['total']

# Setup parallel processing
max_workers = 10  # Adjust based on your needs and API limits
issues_list = []
batch_starts = range(0, total, max_results)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_batch = {
        executor.submit(fetch_batch, start, max_results, fields, JIRA_URL, session): start 
        for start in batch_starts
    }
    
    with tqdm(total=len(batch_starts), desc="Downloading batches") as pbar:
        for future in as_completed(future_to_batch):
            batch_issues = future.result()
            issues_list.extend(batch_issues)
            pbar.update(1)

# Convert collected data to DataFrame
df = pd.json_normalize(issues_list)