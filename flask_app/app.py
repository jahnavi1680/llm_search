import os
from flask import Flask, request, jsonify
import utils

# Load environment variables from .env file

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'. Extracts the query from the request,
    processes it through the search, concatenate, and generate functions,
    and returns the generated answer.
    """
    data = request.get_json()  # Parse the JSON body of the request
    user_query = data.get('query')  # Extract 'query' field

    print("Received query:", user_query)

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    # get the data/query from streamlit app
    print("Received query: ", query)

    
    # Step 1: Search and scrape articles based on the query
    print("Step 1: searching articles")

    # Step 2: Concatenate content from the scraped articles
    print("Step 2: concatenating content")

    # Step 3: Generate an answer using the LLM
    print("Step 3: generating answer")
    answer = utils.search_articles(user_query)
    print(answer)
    # return the jsonified text back to streamlit
    return jsonify({"answer":answer})

if __name__ == '__main__':
    app.run(host='localhost', port=5001)
