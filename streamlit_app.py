import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "gcp-starter"  # Replace with your actual Pinecone environment

# Initialize Pinecone
import pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "adaptive"

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

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def upsert_document(file, entity):
    # Load the document
    loader = UnstructuredWordDocumentLoader(file)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Initialize Pinecone vector store
    vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME, namespace=entity)
    
    return len(chunks)

def process_query(query, entity):
    # Initialize Pinecone vector store for querying
    vectorstore = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=entity)
    
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    
    # Create a retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    
    # Define a custom prompt
    custom_prompt = PromptTemplate(
        template="""You are an AI assistant designed to provide accurate and specific answers based solely on the given context. Use ONLY the information provided in the context to answer the question. If the answer is not in the {entity}, say "I don't have enough information to answer accurately for {entity}." Do not use any external knowledge or make assumptions beyond what's explicitly stated in the context. If the context contains multiple relevant pieces of information, synthesize them into a coherent answer. If the question cannot be answered based on the context, explain why, referring to what information is missing. Remember, accuracy and relevance to the provided context are paramount.

        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question", "entity"]
    )
    
    qa_chain.combine_documents_chain.llm_chain.prompt = custom_prompt
    
    # Run the query
    result = qa_chain({"query": query, "entity": entity})
    
    return result['result']

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
                        num_chunks = upsert_document(uploaded_file, upload_entity)
                        st.success(f"Uploaded {uploaded_file.name}: {num_chunks} chunks processed for entity '{upload_entity}'.")

    # Main area for query interface
    query_entity = st.selectbox("Select Entity for Query", ENTITIES, key="query_entity")
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_query:
            with st.spinner(f"Searching for the best answer in {query_entity}..."):
                answer = process_query(user_query, query_entity)
                st.write(answer)
        else:
            st.warning("Please enter a question before searching.")

if __name__ == "__main__":
    main()
