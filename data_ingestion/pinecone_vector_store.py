import time
import os
import re
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import hashlib
load_dotenv()

pinecone_index = "multi-doc-agentic-rag-dev-test"
pinecone_api = os.getenv("PINECONE_API_KEY")

def upsert_document(file_path, index_name=pinecone_index, dimension=1536):
    # Create Pinecone vector store
    vector_store = create_pinecone_vector_store(index_name=index_name, dimension=dimension)
    
    # Load the documents
    documents = load_documents(file_path)
    
    # Split the documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # Create a new dictionary with id and content
    ids = [str(uuid.uuid4()) for _ in docs]
    sentences = []
    
    
    # Loop through the split documents
    for doc in docs:
        # Access the content using the page_content attribute
        content = doc.page_content  # Use this instead of doc.content or doc['content']
        sentences.append(content)

    # Upsert the documents and their embeddings into Pinecone vector store
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents=docs,# Pass the new structure with IDs and content
        embedding=OpenAIEmbeddings(),
        index_name=index_name,
        ids=ids
    )

    print(f"Successfully upserted documents from: {file_path}")

# Function to load documents from various formats
def load_documents(file_path):
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.json'):
        loader = JSONLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return loader.load()


def create_pinecone_vector_store(
    index_name=pinecone_index, 
    dimension=1536,  
    metric="cosine", 
    cloud="aws", 
    region="us-east-1"
):
    # Enforce naming convention: lowercase alphanumeric and hyphens only
    if not re.match(r'^[a-z0-9-]+$', index_name):
        raise ValueError("Index name must consist of lower case alphanumeric characters or '-'.")

    # Initialize Pinecone client with API key from environment variable
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Check if the index already exists
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Create the index if it doesn't exist
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
        
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print(f"Index '{index_name}' is ready.")

    # Connect to the index and initialize the vector store
    index = pc.Index(index_name)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    
    print(f"Pinecone vector store for '{index_name}' is set up and ready.")
    return vector_store



file_path = "/home/ashish/Desktop/essay.pdf"  # Specify the path to your file
upsert_document(file_path)