import requests
import json

# Google Custom Search API URL
url = "https://www.googleapis.com/customsearch/v1"

# API key and search engine ID
api_key = "YOUR_API_KEY"
search_engine_id = "YOUR_SEARCH_ENGINE_ID"

# Search parameters
params = {
    "q": "Body Worn Cameras",
    "cx": search_engine_id,
    "key": api_key
}

# Send a GET request to the Google Custom Search API
response = requests.get(url, params=params)

# Parse the JSON response
data = response.json()

# Extract the product details for the first 5 results
for result in data["items"][:5]:
    # Extract the product title
    title = result["title"]
    
    # Extract the product link
    link = result["link"]
    
    # Print the product details
    print(f"Title: {title}")
    print(f"Link: {link}")
    print("")