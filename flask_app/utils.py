import os
import requests
from bs4 import BeautifulSoup
import sentence_transformers as st
import chromadb

# Load API keys from environment variables
SERPER_API_KEY = None
OPENAI_API_KEY = None


def search_articles(query):
    """
    Searches for articles related to the query using Serper API.
    Returns a list of dictionaries containing article URLs, headings, and text.
    """
    articles=[]
    headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
    }
    params = {
        "q": "openai chatgpt latest features",
        "num": 10  # your query
    }
    response = requests.post("https://google.serper.dev/search", headers=headers, json=params)
    data = response.json()
    all_content = []
    all_headings = []
    for result in data.get("organic", []):  # organic search results
        print(result["link"])
        content,headings=fetch_article_content(result["link"])
        all_content+=content
        all_headings+=headings
    # implement the search logic - retrieves articles
    return all_content, all_headings


def fetch_article_content(url):
    """
    Fetches the article content, extracting headings and text.
    """
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

    # implementation of fetching headings and content from the articles

    return all_text,headings


def chroma(all_content,all_headings):
    """
    Concatenates the content of the provided articles into a single string.
    """
    model = st.SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, decent quality

    # Set up ChromaDB client (local in-memory DB)
    chroma_client = chromadb.PersistentClient(path="db")  # This is the new correct way
    collection = chroma_client.get_or_create_collection(name="temporary")
    embeddings = model.encode(all_content).tolist()
    ids = [f"item_{i}" for i in range(len(all_content))]

    collection.add(
        documents=all_content,
        embeddings=embeddings,
        metadatas=all_headings,
        ids=ids
    )           


    # formatting + concatenation of the string is implemented here

    return collection


def generate_answer(collection, query):
    """
    Generates an answer from the concatenated content using GPT-4.
    The content and the user's query are used to generate a contextual answer.
    """
    model = st.SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, decent quality
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
    # Create the prompt based on the content and the query
    
    # implement openai call logic and get back the response
    return response.choices[0].message.content
