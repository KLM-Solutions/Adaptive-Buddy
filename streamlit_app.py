import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import openai
import tiktoken
from tiktoken import get_encoding
import os
from dotenv import load_dotenv
from docx import Document as DocxDocument
import re
import time
from tqdm import tqdm

load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Adaptive"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "adaptive"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Define the list of entities
ENTITIES = [
    "Consolidation Infographic",
    "Industry Use Case ",
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
    doc = DocxDocument(file)
    paragraphs = [para.text for para in doc.paragraphs]
    return paragraphs

def generate_chunk_description(chunk):
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
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
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    return response.content

def upsert_document(file, metadata, entity):
    paragraphs = extract_text_from_docx(file)
    text = '\n'.join(paragraphs)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    total_chunks = len(chunks)
    st.write(f"Total chunks to process: {total_chunks}")

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_description = generate_chunk_description(chunk)
            chunk_id = f"{metadata['title']}_chunk_{i}"
            chunk_embedding = embeddings.embed_query(chunk)
            
            vectors_to_upsert.append((
                chunk_id,
                chunk_embedding,
                {
                    'text': chunk,
                    'entity': entity,
                    'description': chunk_description,
                    'title': metadata['title']
                }
            ))

            # Update progress
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total_chunks} chunks.")

            # Batch upsert when we reach 100 vectors or on the last chunk
            if len(vectors_to_upsert) == 100 or i == total_chunks - 1:
                index.upsert(vectors=vectors_to_upsert, namespace=entity)
                vectors_to_upsert = []

        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")

    st.success(f"Document '{metadata['title']}' processing completed. {total_chunks} chunks successfully upserted for entity '{entity}'.")

def process_query(query, entity):
    if query:
        with st.spinner(f"Searching for the best answer in {entity}..."):
            # Use LangChain's Pinecone integration for retrieval
            vector_store = Pinecone(index, embeddings.embed_query, "text", namespace=entity)
            
            # Create a retrieval chain
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1),
                chain_type="stuff",
                retriever=retriever,
            )

            # Get the answer
            result = qa_chain({"query": query})
            st.write(result['result'])
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
