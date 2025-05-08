import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/jahnavipoloju/llm_search/.env")


SERPER_API_KEY = os.getenv("SERPER_API_KEY")
print(SERPER_API_KEY)
import requests


headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
}

params = {
    "q": "openai chatgpt latest features",
      "num": 10  # your query
}

response = requests.post("https://google.serper.dev/search", headers=headers, json=params)

# Check the response
if response.status_code == 200:
    data = response.json()
    for result in data.get("organic", []):  # organic search results
        print(result["title"])
        print(result["link"])
        print()
else:
    print("Error:", response.status_code, response.text)
