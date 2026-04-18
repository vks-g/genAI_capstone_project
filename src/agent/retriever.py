import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_EMBEDDINGS_CACHE = None


def _get_embeddings():
    """Lazy singleton loader for the HuggingFace embedding model.
    Only downloads/loads the model on the first call, not at import time.
    This prevents slow cold starts on Streamlit Cloud.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name="intfloat/e5-small")
    return _EMBEDDINGS_CACHE


DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db"
)


def get_vector_store(use_cloud=False, cloud_api_key=None):
    """
    Get vector store from local or cloud ChromaDB.

    Args:
        use_cloud: If True, use Chroma Cloud; if False, use local ChromaDB
        cloud_api_key: Chroma Cloud API key (required if use_cloud=True)
    """
    if use_cloud:
        if not cloud_api_key:
            raise ValueError("Cloud mode requires api_key parameter")

        client = chromadb.CloudClient(
            api_key=cloud_api_key,
            tenant=os.getenv("CHROMA_TENANT", ""),
            database=os.getenv("CHROMA_DATABASE", "customer-churn"),
        )

        return Chroma(
            collection_name="saas_retention_strategies",
            embedding_function=_get_embeddings(),
            client=client,
        )
    else:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"chroma_db not found at {DB_PATH}. "
                "Run: python -m src.agent.build_vectorstore"
            )
        return Chroma(
            collection_name="saas_retention_strategies",
            embedding_function=_get_embeddings(),
            persist_directory=DB_PATH,
        )


def retrieve_strategies(query: str, k: int = 3, use_cloud=False) -> list[str]:
    """
    Retrieves top-k relevant text chunks.

    Args:
        query: Search query
        k: Number of results to return
        use_cloud: If True, query Chroma Cloud; if False, query local ChromaDB
    """
    try:
        # Get cloud credentials from environment if using cloud
        if use_cloud:
            cloud_api_key = os.getenv("CHROMA_API_KEY")

            vector_store = get_vector_store(use_cloud=True, cloud_api_key=cloud_api_key)
        else:
            vector_store = get_vector_store()

        retrieved_docs = vector_store.similarity_search(query, k=k)

        return [doc.page_content for doc in retrieved_docs]
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return []


if __name__ == "__main__":
    use_cloud = os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true"

    test_query = "What should I do for a customer with low engagement?"

    if use_cloud:
        print("Using Chroma Cloud...")
    else:
        print("Using local ChromaDB...")

    results = retrieve_strategies(test_query, use_cloud=use_cloud)
    for i, res in enumerate(results):
        print(f"--- Result {i + 1} ---\n{res}\n")
