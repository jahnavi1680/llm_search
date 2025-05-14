import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import openai
import textwrap
import httpx
#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import together

#from sentence_transformers import SentenceTransformer
import sentence_transformers as st
import chromadb
#from chromadb.config import Settings

load_dotenv(dotenv_path="/Users/jahnavipoloju/llm_search/.env")


model = st.SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, decent quality

# Set up ChromaDB client (local in-memory DB)
chroma_client = chromadb.PersistentClient(path="db")  # This is the new correct way
collection = chroma_client.get_or_create_collection(name="temporary")

# ---- STEP 1: Load your data ----
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
}
params = {
    "q": "openai chatgpt latest features",
      "num": 1  # your query
}
response = requests.post("https://google.serper.dev/search", headers=headers, json=params)

# Check the response
'''if response.status_code == 200:
    data = response.json()
    all_text = ''
    for result in data.get("organic", []):  # organic search results
        print(result["title"])
        print(result["link"])
        #lets now get the html content of te webpages
        response = requests.get(result["link"])
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tags = soup.find_all(['h1', 'h2', 'h3', 'p'])
            for tag in tags:
                all_text += f"{tag.name}: {tag.get_text(strip=True)}\n"
    print(all_text)

else:
    print("Error:", response.status_code, response.text)'''
all_text=''
url='https://en.wikipedia.org/wiki/Donkey'
response = requests.get(url)
print(response)
if response.status_code == 200:
    current_chunk = ''
    chunks = []
    heading_stack = []  # Stack of (level, text), e.g., [(1, "Intro"), (2, "Background")]
    current_h=''
    
    soup = BeautifulSoup(response.content, 'html.parser')
    tags = soup.find_all(['h1', 'h2', 'h3','h4','h5','h6', 'p'])
    maps = {'h1': 1,
        'h2': 2,
        'h3': 3,
        'h4': 4,
        'h5': 5,
        'h6': 6
    }
    chunks = []
    current_chunk = {'heading': None, 'paragraphs': []}
    all_text=[]
    headings=[]
    for idx,element in enumerate(tags):
        tag = element.name
        text = element.get_text().strip()

        if tag.startswith('h') and tag[1:].isdigit():
            # Save previous chunk if it had a heading and content
            if current_chunk['heading'] or current_chunk['paragraphs']:
                chunks.append(current_chunk)
                all_text.append(current_chunk['heading']+':'+' '.join(current_chunk['paragraphs']))
                headings.append({"type": tag, "id": idx})
            # Start a new chunk
            current_chunk = {'heading': text, 'paragraphs': []}
        elif tag == 'p':
            if current_chunk['heading'] is None:
                # Paragraph without heading — optional: skip or group under 'untitled'
                current_chunk['heading'] = 'untitled'
            current_chunk['paragraphs'].append(text)

    # Append the last chunk if not empty
    if current_chunk['heading'] or current_chunk['paragraphs']:
        chunks.append(current_chunk)


print(len(all_text))


embeddings = model.encode(all_text).tolist()
ids = [f"item_{i}" for i in range(len(all_text))]

collection.add(
    documents=all_text,
    embeddings=embeddings,
    metadatas=headings,
    ids=ids
)           

model = st.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

query = "what do donkeys eat as their main source of energy?"
query_embedding = model.encode(query).tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

documents = results['documents'][0]  # list of top matching docs
prompt = f"""
You are an expert question-answer assistant. Based on the information below, answer the query.

Context:
{documents}

Question:
{query}
"""

client = together.Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
response = client.chat.completions.create(
model="Qwen/Qwen3-235B-A22B-fp8-tput",
messages=[{"role": "user", "content":prompt}]
)


print(response.choices[0].message.content)



# Load model and tokenizer
#model_id = "mistralai/Mistral-7B-Instruct-v0.1"
##tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
#summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)


class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

CHUNK_SIZE = 1000  # Words per chunk
MODEL = "gpt-3.5-turbo"
def split_text(all_text, max_words=CHUNK_SIZE):
    words = all_text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def summarize_chunk(chunk, model=MODEL):
    openai.api_key=os.getenv("openai")
    client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")
    client = openai.OpenAI(http_client=CustomHTTPClient())
    response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-fp8-tput",
    messages=[{"role": "user", "content":  f"Please summarize the following:\n\n{chunk}"}]
    )
    '''messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Please summarize the following:\n\n{chunk}"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )'''
    return response.choices[0].message.content


def map_reduce_summarize(long_text, model="gpt-3.5-turbo"):
    chunks = split_text(long_text)
    
    print(f"⏳ Mapping {len(chunks)} chunks...")
    chunk_summaries = [summarize_chunk(chunk, model=model) for chunk in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    print("📦 Reducing...")
    combined_summary_input = "\n\n".join(chunk_summaries)
    final_summary = summarize_chunk("Summarize the following summaries:\n\n" + combined_summary_input, model=model)
    
    return final_summary







        
    



