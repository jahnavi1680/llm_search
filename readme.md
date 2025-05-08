
# LLM-Based RAG System

## Overview

This project is designed to create a Retrieval-Augmented Generation (RAG) system using a Large Language Model (LLM). The system integrates with an API to scrape content from the internet and uses an API to serve the LLM-generated answers. A simple front-end interface is provided to interact with the system. 
Note: ONLY use the packages provided in the requirements.txt file (similar/alternative packages are ok only if they perform similar task/function). 

## Process Overview

1. **User Input via Streamlit Interface**:
   - The user interacts with a Streamlit-based front-end where they can input their query.

2. **Query Sent to Flask Backend**:
   - The query entered by the user is sent from the Streamlit interface to a Flask backend via an API call.

3. **Internet Search and Article Scraping**:
   - The Flask backend searches the internet for the query using a designated API. It retrieves the top relevant articles and scrapes their content, extracting only the useful text (headings and paragraphs).

4. **Content Processing**:
   - The scraped content is processed to create a coherent input, which is then passed to the LLM for generating a response.

5. **LLM Response Generation**:
   - The processed content and the user's query are used to generate a contextual answer using the LLM. The LLM is accessed via an API, and the generated response is returned to the Flask backend.

6. **Response Sent Back to Streamlit Interface**:
   - The Flask backend sends the generated answer back to the Streamlit interface, where it is displayed to the user.

## What we expect?
We expect you to explore, understand the components and functionality, and demonstrate your ability to work with the provided tools and deliver a solution that meets the requirements. 

Bonus points: If you use Langchain (or similar tools to add memory to the system) to make the chatbot conversational.
## Prerequisites

- Python 3.8 or above

## Setup Instructions

### Step 1: Clone or download the Repository (if emailed)

```bash
git clone https://github.com/your-repo-url.git
cd project_name
```

Or download it

### Step 2: Set Up a Virtual Environment

You can use `venv` or `conda` to create an isolated environment for this project.

#### Using `venv`

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

#### Using `conda`

```bash
conda create --name project_env python=3.8
conda activate project_env
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys in a way it can be accessed in the app.


### Step 5: Run the Flask Backend

Navigate to the `flask_app` directory and start the Flask server:

```bash
cd flask_app
python app.py
```

### Step 6: Run the Streamlit Frontend

In a new terminal, run the Streamlit app:

```bash
cd streamlit_app
streamlit run app.py
```

### Step 7: Open the Application

Open your web browser and go to `http://localhost:8501`. You can now interact with the system by entering your query.

## Project Structure

- **flask_app/**: Contains the backend Flask API and utility functions.
- **streamlit_app/**: Contains the Streamlit front-end code.
- **.env**: Stores API keys (make sure this file is not included in version control).
- **requirements.txt**: Lists the project dependencies.

## Task Instructions for Candidates

You are required to complete the following:

1. Implement the functionality to fetch, process, and generate responses from an LLM using the provided APIs.
2. Integrate the APIs with the Flask backend.
3. Display the results in the Streamlit frontend.

Good luck!