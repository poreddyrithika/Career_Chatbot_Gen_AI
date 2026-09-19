import os
import re
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

# --------------------------------------------------
# Setup API key and LLM
# --------------------------------------------------

api_key = "your_groq_api_key"

if not api_key:
    st.error("GROQ_API_KEY not set. Please configure it as an environment variable.")
    st.stop()

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b",
    api_key=api_key,
)


# --------------------------------------------------
# Initialize Vector DB (created once, reused across reruns)
# --------------------------------------------------

@st.cache_resource
def get_collection():
    client = chromadb.Client()
    return client.get_or_create_collection("career_knowledge_base")


collection = get_collection()

# In no sequel collection=table

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()


# --------------------------------------------------
# Function to ingest PDF
# --------------------------------------------------

def ingest_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Career Guidance Chatbot")
st.markdown("Get personalized, grounded career advice based on your uploaded documents.")


# --------------------------------------------------
# Upload PDFs
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Career Resources (PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    newly_ingested = 0

    for file in uploaded_files:

        if file.name in st.session_state.ingested_files:
            continue

        text = ingest_pdf(file)

        if not text.strip():
            st.warning(f"No extractable text found in {file.name} (skipped).")
            continue

        # Chunking (chunk = multiple sentences)
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_text(text)

        if chunks:
            collection.add(
                documents=chunks,
                ids=[f"{file.name}_{i}" for i in range(len(chunks))]
            )

        st.session_state.ingested_files.add(file.name)
        newly_ingested += 1

    if newly_ingested:
        st.success(f"{newly_ingested} PDF(s) ingested successfully!")


# --------------------------------------------------
# User career query
# --------------------------------------------------

user_query = st.text_input("Ask your career question:")

if st.button("Get Advice") and user_query:

    if collection.count() == 0:
        st.warning("Please upload at least one PDF before asking a question.")
        st.stop()

    # Vector Search
    vector_results = collection.query(
        query_texts=[user_query],
        n_results=5
    )
    vector_docs = vector_results["documents"][0]

    # Keyword Search
    keywords = user_query.lower().split()
    keyword_docs = [
        doc for doc in vector_docs
        if any(k in doc.lower() for k in keywords)
    ]

    # hybrid search = vector search + keyword search
    # hybrid merge (order-preserving, no duplicates)
    hybrid_docs = list(dict.fromkeys(vector_docs + keyword_docs))

    # Reranking using LLM (to avoid hallucinations and to save token limit)
    # in llm perspective, prompt = instruction
    numbered_docs = "\n".join(
        f"{i + 1}. {doc}" for i, doc in enumerate(hybrid_docs)
    )

    rerank_prompt = PromptTemplate.from_template(
        """User Query:
        {query}

        Documents:
        {docs}

        Rank these documents from most relevant to least relevant for providing
        career advice, including skills and companies.
        Reply with ONLY the numbers of the top 3 documents, most relevant first,
        separated by commas (example: 2, 1, 4).
        """
    )

    rerank_chain = rerank_prompt | llm
    reranked_output = rerank_chain.invoke({
        "query": user_query,
        "docs": numbered_docs
    })

    # Parse the ranked numbers out of the LLM's reply and map back to real docs
    ranked_numbers = [int(n) for n in re.findall(r"\d+", reranked_output.content)]

    top_context = []
    for n in ranked_numbers:
        index = n - 1
        if 0 <= index < len(hybrid_docs):
            top_context.append(hybrid_docs[index])
        if len(top_context) == 3:
            break

    # Fallback in case parsing failed to find anything
    if not top_context:
        top_context = hybrid_docs[:3]

    # Final RAG prompt for career advice
    final_prompt = PromptTemplate.from_template(
        """
        You are a career guidance AI assistant.
        Based on the following resources:
        {context}
        Provide a personalized roadmap for the user:
        -Skills to learn
        -Recommended companies
        -Steps to improve career readiness
        User Query:
        {query}
        """
    )

    rag_chain = final_prompt | llm
    career_advice = rag_chain.invoke({
        "context": "\n".join(top_context),
        "query": user_query
    })

    st.subheader("Top Retrieved Context")
    for doc in top_context:
        st.write("-", doc)

    st.subheader("Personalized Career Advice")
    st.write(career_advice.content)

