import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.agent.document_loader import load_and_chunk_documents

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "chroma_db")


def create_and_persist_vector_store(
    kb_path="./knowledge_base", db_path=DB_PATH, use_cloud=False, cloud_api_key=None
):
    """
    Create and persist vector store to local or cloud ChromaDB.

    Args:
        kb_path: Path to knowledge base documents
        db_path: Path for local ChromaDB (used when use_cloud=False)
        use_cloud: If True, use Chroma Cloud; if False, use local ChromaDB
        cloud_api_key: Chroma Cloud API key (required if use_cloud=True)
    """
    chunks = load_and_chunk_documents(kb_path)
    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small")

    if use_cloud:
        if not cloud_api_key:
            raise ValueError("Cloud mode requires api_key parameter")

        client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT", ""),
            database=os.getenv("CHROMA_DATABASE", "customer-churn"),
        )

        print(f"Uploading {len(chunks)} chunks to Chroma Cloud in batches...")
        batch_size = 100

        collection = client.get_or_create_collection(name="saas_retention_strategies")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(
                f"  Uploading batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}..."
            )

            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                client=client,
                collection_name="saas_retention_strategies",
            )

        print(
            f"Successfully embedded and persisted {len(chunks)} chunks to Chroma Cloud."
        )
        return vector_store
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name="saas_retention_strategies",
        )
        print(f"Successfully embedded and persisted {len(chunks)} chunks to {db_path}.")

    return vector_store


if __name__ == "__main__":
    use_cloud = os.getenv("USE_CHROMA_CLOUD", "false").lower() == "true"

    if use_cloud:
        cloud_api_key = os.getenv("CHROMA_API_KEY")

        if not cloud_api_key:
            print("Error: CHROMA_API_KEY environment variable required for cloud mode")
            print("Falling back to local mode...")
            use_cloud = False

    if use_cloud:
        print("Uploading to Chroma Cloud...")
        create_and_persist_vector_store(use_cloud=True, cloud_api_key=cloud_api_key)
    else:
        print("Using local ChromaDB...")
        create_and_persist_vector_store()
