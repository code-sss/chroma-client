from chromadb import HttpClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
import os


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Get the ChromaDB host and port from environment variables
chromadb_host = os.getenv("CHROMADB_HOST", "localhost")
chromadb_port = int(os.getenv("CHROMADB_PORT", 8000))


def main():
    # Initialize the ChromaDB client
    client = HttpClient(
        host=chromadb_host,
        port=chromadb_port
    )

    print(f"Connected with chroma db at "
          f"{chromadb_host}:{chromadb_port} => {client.heartbeat()}")

    # List all collections
    collections = client.list_collections()
    print(f"Collections: {collections}")

    # Create a collection
    collection = client.get_or_create_collection(
        name="test",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda:0")   # Using GPU with CUDA
    )
    # Add documents to the collection
    collection.add(
        documents=["hello world", "goodbye world", "hello again"],
        metadatas=[{"source": "test"}, {
            "source": "test"}, {"source": "test"}],
        ids=["1", "2", "3"]
    )

    # List all collections
    collections = client.list_collections()
    print(f"Collections: {collections}")

    # Query the collection
    results = collection.query(
        query_texts=["hi there"],
        n_results=2
    )
    # Print the results
    print(f"Query results: {results}")

    # Delete the collection
    client.delete_collection(name="test")
    print("Collection deleted")

    # List all collections
    collections = client.list_collections()
    print(f"Collections: {collections}")


if __name__ == "__main__":
    main()
