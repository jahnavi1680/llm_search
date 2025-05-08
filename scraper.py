import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/jahnavipoloju/llm_search/.env")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

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
    text = ''
    for result in data.get("organic", []):  # organic search results
        print(result["title"])
        print(result["link"])
        #lets now get the html content of te webpages
        response = requests.get(result["link"])
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tags = soup.find_all(['h1', 'h2', 'h3', 'p'])
            for tag in tags:
                text += tag.name + ': ' + tag.get_text()
    text_file = open("output.txt", "w")

    text_file.write(text)

    text_file.close()


        
    


else:
    print("Error:", response.status_code, response.text)
