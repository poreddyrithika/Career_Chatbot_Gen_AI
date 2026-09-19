# Career Guidance Chatbot

An AI-powered **Career Guidance Chatbot** that provides personalized and grounded career advice based on career-related PDF resources uploaded by the user.

The application uses **Retrieval-Augmented Generation (RAG)** with ChromaDB to retrieve relevant information from uploaded documents before generating career guidance using a Groq-powered LLM.

## Features

* Upload multiple career-related PDF resources
* Extract text from PDF documents
* Split documents into smaller chunks
* Store document chunks in ChromaDB
* Perform vector-based document retrieval
* Perform keyword-based filtering
* Combine vector and keyword search for hybrid retrieval
* Use an LLM to rerank retrieved documents
* Generate personalized career roadmaps
* Recommend skills to learn
* Suggest relevant companies
* Provide steps to improve career readiness
* Streamlit-based user interface
* Environment-variable based API key handling

## Technologies Used

* **Python**
* **Streamlit** – Web application interface
* **LangChain** – LLM and RAG workflow
* **ChromaDB** – Vector database
* **Groq** – LLM API
* **PyPDF2** – PDF text extraction
* **RecursiveCharacterTextSplitter** – Document chunking

## Architecture

```text
Career Resource PDFs
        ↓
     PDF Reader
        ↓
   Text Extraction
        ↓
      Chunking
        ↓
     ChromaDB
        ↓
   User Career Query
        ↓
 ┌─────────────────────┐
 │   Vector Search     │
 │   Keyword Search    │
 └─────────────────────┘
        ↓
   Hybrid Retrieval
        ↓
    LLM Reranking
        ↓
  Top Relevant Context
        ↓
      RAG Prompt
        ↓
      Groq LLM
        ↓
Personalized Career Advice
```

## How It Works

### 1. Upload Career Resources

The user uploads one or more PDF documents containing career-related information.

### 2. Extract Text

`PyPDF2` extracts readable text from each uploaded PDF.

### 3. Chunk the Documents

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
```

The overlap helps preserve context between neighboring chunks.

### 4. Store in ChromaDB

The chunks are stored in a ChromaDB collection named:

```text
career_knowledge_base
```

### 5. Retrieve Relevant Information

When the user asks a career question, the application performs vector search to retrieve relevant document chunks.

It also applies keyword-based filtering to improve retrieval.

### 6. Hybrid Search

The vector-search results and keyword-search results are combined while removing duplicates.

```text
Vector Search + Keyword Search
              ↓
         Hybrid Results
```

### 7. LLM Reranking

The retrieved documents are sent to the LLM, which identifies the three most relevant documents for the user's query.

This reduces the amount of unnecessary context sent to the final RAG prompt.

### 8. Generate Career Advice

The top retrieved documents are passed to the final RAG prompt.

The LLM generates guidance covering:

* Skills to learn
* Recommended companies
* Career-readiness steps
* A personalized roadmap

## Project Structure

```text
Career_Chatbot_Gen_AI/
│
├── career_chatbot_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/poreddyrithika/Career_Chatbot_Gen_AI.git
```

Move into the project directory:

```bash
cd Career_Chatbot_Gen_AI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## API Key Setup

The application reads the Groq API key from an environment variable.

The code uses:

```python
api_key = os.environ.get("GROQ_API_KEY")
```

### Windows PowerShell

Set the environment variable for the current terminal session:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Then run the application.

**Never commit your API key to GitHub.**

## Running the Application

Run:

```bash
streamlit run career_chatbot_app.py
```

The Streamlit application will open in your browser.

## Example Questions

You can ask questions such as:

```text
What skills should I learn for an AI/ML career?
```

```text
What companies hire freshers for software development roles?
```

```text
Give me a roadmap to prepare for AI/ML placements.
```

```text
What technologies should I learn for a data science career?
```

## RAG Workflow

The chatbot follows a Retrieval-Augmented Generation workflow:

```text
PDF Documents
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
ChromaDB
     ↓
User Query
     ↓
Hybrid Retrieval
     ↓
LLM Reranking
     ↓
Relevant Context
     ↓
RAG
     ↓
Career Advice
```

## Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Vector databases
* Document chunking
* Semantic retrieval
* Keyword search
* Hybrid search
* LLM-based reranking
* Prompt engineering
* PDF processing
* Environment-variable based secret management
* Streamlit application development

## Note

This project is developed for **learning and educational purposes** and demonstrates how RAG and LLMs can be used to build a career guidance application.

