import os
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")

# 1. Load and process the medical book (PDF)
def load_pdf(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# 2. Split the text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# 3. Download the pre-trained HuggingFace embedding model
def download_embeddings_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# 4. Create FAISS index from the embeddings
def create_faiss_index(text_chunks, embeddings):
    # Convert chunks into embeddings
    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    # Initialize FAISS index
    dimension = 384  # Embedding dimension from MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)

    # Convert embeddings to a numpy array and add to the FAISS index
    embedding_array = np.array(chunk_embeddings).astype('float32')
    index.add(embedding_array)

    # Return the index and text chunks for retrieval
    return index, chunk_texts


# 5. Search the FAISS index for the most similar text chunk based on query
def search_index(query, index, embeddings, chunk_texts):
    # Embed the query
    query_embedding = embeddings.embed_query(query)

    # Search for the most similar text in the index
    D, I = index.search(np.array([query_embedding], dtype='float32'), k=1)  # Find the top match

    # Get the matching chunk
    matching_chunk = chunk_texts[I[0][0]] if I[0].size > 0 else "No relevant information found."
    
    return matching_chunk

# 6. Display the response clearly and appropriately
def display_response(matching_chunk):
    # Ensure that the response is clear and concise
    max_length = 1000  # Define a maximum length for response
    if len(matching_chunk) > max_length:
        # Truncate long responses
        response = matching_chunk[:max_length] + "..."
    else:
        response = matching_chunk
    
    # Print the response
    print("Here is the relevant response to your query:\n")
    print(response)
    print("\nIf you need more details, feel free to ask another question!")
    
# Main function to run the chatbot
def run_chatbot():
    data_path = "data/"
    
    # 1. Load the PDF
    documents = load_pdf(data_path)

    # 2. Split the text into chunks
    text_chunks = split_text_into_chunks(documents)

    # 3. Download the HuggingFace embedding model
    embeddings = download_embeddings_model()

    # 4. Create FAISS index
    index, chunk_texts = create_faiss_index(text_chunks, embeddings)

    # 5. Simulate a chat session
    while True:
        query = input("Ask a medical question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the chatbot. Have a great day!")
            break
        
        matching_chunk = search_index(query, index, embeddings, chunk_texts)
        
        # Display the result in a clear and concise manner
        display_response(matching_chunk)
    
# Run the chatbot
run_chatbot()