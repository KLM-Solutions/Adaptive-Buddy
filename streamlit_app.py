import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Initialize API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]  # Add this to your secrets
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

# LangChain Tracing Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Adaptive"

# Initialize Pinecone
INDEX_NAME = "adaptive"

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone vector store
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text",
    namespace=PINECONE_ENV
)

# Define the list of entities (unchanged)
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

def upsert_document(file, metadata, entity):
    # Load the document
    loader = UnstructuredWordDocumentLoader(file)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata to chunks
    for chunk in chunks:
        chunk.metadata.update(metadata)
        chunk.metadata['entity'] = entity
    
    # Upsert to Pinecone
    vectorstore.add_documents(chunks)
    
    st.success(f"Document '{metadata['title']}' processing completed. {len(chunks)} chunks successfully upserted for entity '{entity}'.")

def query_pinecone(query, entity):
    # Search in Pinecone
    results = vectorstore.similarity_search(
        query,
        k=2,
        namespace=entity
    )
    return [doc.page_content for doc in results]

def get_answer(context, user_query, entity):
    chat = ChatOpenAI(model_name="gpt-4-0613", temperature=0.3)
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
