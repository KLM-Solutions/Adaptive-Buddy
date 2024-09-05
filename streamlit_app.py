import streamlit as st
import os
from dotenv import load_dotenv
from docx import Document
import re
import time
from tqdm import tqdm

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    Document
)

load_dotenv()

# Initialize API keys and environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
INDEX_NAME = "adaptive"

# Set up LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Adaptive"

# Initialize embeddings and Pinecone index
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone_index = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    environment="us-east-1"
)

# Define the list of entities
ENTITIES = [
    "Consolidation Infographic",
    "Industry Use Case",
    "Administrator Guide and User Manual",
    "Data-Entry",    
    "Integration",
    "Model-Administration_July2024",
    "Product-Downloads",
    "Product-Support",
    "Reporting-and-Analysis",
    "Security-Administration",
    "Admin-Guide",
    "FDM Considerations",
    "Consolidation Quick Reference Guide",
    "Dashboards Analytics Datasheet",
    "Integration Datasheet",
    "OfficeConnect Datasheet",
    "Sales Planning Datasheet"
]

def extract_text_from_docx(file):
    doc = Document(file)
    paragraphs = [para.text for para in doc.paragraphs]
    return paragraphs

def generate_chunk_description(chunk):
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    system_message = SystemMessage(content="""
    You are an AI assistant designed to create concise summaries and descriptions of text chunks stored in Pinecone. Your task is to:
    1. Provide a brief summary of the main ideas and themes in the chunk.
    2. Describe the key topics and concepts covered, without directly quoting the text.
    3. Highlight the significance or context of the information, if apparent.
    4. Avoid mentioning any document names or titles in your description.
    5. Keep your response maximum 150 words to ensure it's concise yet informative.
    Your summary should give readers a clear understanding of the chunk's content without reproducing the exact text.
    """)
    human_message = HumanMessage(content=f"Please provide a concise summary and description of the following text chunk, without using direct quotes:\n\n{chunk}")
    response = chat([system_message, human_message])
    return response.content

def upsert_document(file, metadata, entity):
    paragraphs = extract_text_from_docx(file)
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    max_metadata_size = 40 * 1024  # 40 KB in bytes
    max_chunk_size = 35 * 1024  # 35 KB to leave room for other metadata

    for paragraph in paragraphs:
        paragraph_size = len(paragraph.encode('utf-8'))
        if current_chunk_size + paragraph_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_chunk_size = 0
        if paragraph_size > max_chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_size = len(sentence.encode('utf-8'))
                if current_chunk_size + sentence_size > max_chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0
                current_chunk.append(sentence)
                current_chunk_size += sentence_size
        else:
            current_chunk.append(paragraph)
            current_chunk_size += paragraph_size
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    total_chunks = len(chunks)
    st.write(f"Total chunks to process: {total_chunks}")

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 100  # Adjust based on your Pinecone plan and rate limits
    documents_to_upsert = []
    successful_upserts = 0

    for i, chunk in enumerate(chunks):
        try:
            chunk_description = generate_chunk_description(chunk)
            chunk_id = f"{metadata['title']}_chunk_{i}"
            chunk_metadata = {
                'chunk_id': chunk_id,
                'entity': entity,
                'description': chunk_description
            }

            # Create a Document object
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            documents_to_upsert.append(doc)

            # Batch upsert when we reach the batch size or on the last chunk
            if len(documents_to_upsert) == batch_size or i == total_chunks - 1:
                retry_count = 0
                while retry_count < 3:  # Retry up to 3 times
                    try:
                        pinecone_index.add_documents(documents_to_upsert, namespace=entity)
                        successful_upserts += len(documents_to_upsert)
                        documents_to_upsert = []  # Clear the batch after successful upsert
                        break
                    except Exception as e:
                        retry_count += 1
                        st.warning(f"Upsert attempt {retry_count} failed. Retrying in 5 seconds...")
                        time.sleep(5)

                if retry_count == 3:
                    st.error(f"Failed to upsert batch after 3 attempts. Skipping this batch.")
                    documents_to_upsert = []  # Clear the batch to continue with next chunks

            # Update progress
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total_chunks} chunks. Successfully upserted: {successful_upserts}")

        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")

    st.success(f"Document '{metadata['title']}' processing completed. {successful_upserts} out of {total_chunks} chunks successfully upserted for entity '{entity}'.")

def query_pinecone(query, entity):
    results = pinecone_index.similarity_search_with_score(
        query,
        k=2,  # Reduced from 3 to 2 for faster processing
        namespace=entity
    )
    return [result[0].page_content for result in results]

def get_answer(context, user_query, entity):
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY)
    system_message = SystemMessage(content=f"""You are an AI assistant designed to provide accurate and specific answers based solely on the given context. Follow these instructions strictly:
    Use ONLY the information provided in the context to answer the question.
    If the answer is not in the {entity}, say "I don't have enough information to answer accurately for {entity}."
    Do not use any external knowledge or make assumptions beyond what's explicitly stated in the context.
    If the context contains multiple relevant pieces of information, synthesize them into a coherent answer.
    If the question cannot be answered based on the context, explain why, referring to what information is missing.
    Remember, accuracy and relevance to the provided context are paramount.""")
    human_message = HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}")
    response = chat([system_message, human_message])
    return response.content

def process_query(query, entity):
    if query:
        with st.spinner(f"Searching for the best answer in {entity}..."):
            matches = query_pinecone(query, entity)
            if matches:
                context = "\n\n".join(matches)
                answer = get_answer(context, query, entity)
                st.write(answer)
            else:
                st.warning(f"No relevant information found in {entity}. Please try a different question or entity.")
    else:
        st.warning("Please enter a question before searching.")

def main():
    st.title("Adaptive-Buddy")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload DOCX Files", type="docx", accept_multiple_files=True)
        if uploaded_files:
            upload_entity = st.selectbox("Select Entity for Upload", ENTITIES, key="upload_entity")
            if st.button("Upload Documents"):
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        metadata = {"title": uploaded_file.name}
                        upsert_document(uploaded_file, metadata, upload_entity)

    # Main area for query interface
    query_entity = st.selectbox("Select Entity for Query", ENTITIES, key="query_entity")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        process_query(user_query, query_entity)

if __name__ == "__main__":
    main()
