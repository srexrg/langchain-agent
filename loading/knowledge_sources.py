from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os  
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Initialize embedding function
embedding_function = OpenAIEmbeddings()

# Load text document
current_dir = os.path.dirname(os.path.abspath(__file__))
text_filename = "../data.txt"
text_path = os.path.join(current_dir, text_filename)
loader = TextLoader(text_path)
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create and persist the vector store
vector_store = Chroma.from_documents(
    documents=texts, embedding=embedding_function, persist_directory="./chunk"
)

print(f"Populated Chroma with {len(texts)} text chunks from {text_path}.")

query = " Impression Metrics"
docs = vector_store.similarity_search(query)

print("\nTest Query:", query)
print("\nTop relevant chunk:")
print(docs[0].page_content)

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize embedding function
embedding_function = OpenAIEmbeddings()

# Load the persisted database
loaded_vector_store = Chroma(
    persist_directory="./chunk", embedding_function=embedding_function
)

# Test query
query = "What are the key metrics"
docs = loaded_vector_store.similarity_search(query)

print("Test Query:", query)
print("\nTop relevant chunk:")
print(docs[0].page_content)
