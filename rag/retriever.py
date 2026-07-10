"""
RAG-поиск по каталогу запчастей через ChromaDB.

Индексирует описания SKU (название, категория, характеристики) и позволяет
искать по ним на естественном языке — например, "аналоги тормозных колодок
для Lada Vesta".
"""

import os
import chromadb
import pandas as pd

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def index_catalog(df: pd.DataFrame, collection_name: str = "parts_catalog") -> None:
    """
    df: каталог SKU с колонками ['sku', 'name', 'category', 'description'].
    Индексирует каждую строку как отдельный документ.
    """
    client = get_client()
    collection = client.get_or_create_collection(collection_name)

    documents = (df["name"] + " | " + df["category"] + " | " + df["description"].fillna("")).tolist()
    ids = df["sku"].astype(str).tolist()
    metadatas = df[["sku", "category"]].to_dict("records")

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)


def search_catalog(query: str, n_results: int = 5, collection_name: str = "parts_catalog") -> list[dict]:
    """Поиск по каталогу на естественном языке. Возвращает список найденных SKU с метаданными."""
    client = get_client()
    collection = client.get_or_create_collection(collection_name)

    results = collection.query(query_texts=[query], n_results=n_results)

    return [
        {"id": doc_id, "document": doc, "metadata": meta}
        for doc_id, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]
