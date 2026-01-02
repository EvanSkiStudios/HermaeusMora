import os
import faiss
import numpy as np
import hashlib
import json
import ollama

from pathlib import Path
from utility_scripts.system_logging import setup_logger


# configure logging
logger = setup_logger(__name__)


# PATHS
base_dir = Path.cwd()
faiss_path = base_dir / "faiss.bin"
embeddings_path = base_dir / "embeddings.npy"
metadata_path = base_dir / "metadata.json"


def json_builder(faiss_index: int, chunk_index: int, content: str) -> dict:
    """
    Build a metadata entry for a chunk.
    :param faiss_index: Index in Faiss DB
    :param chunk_index: Index of the current Chunk
    :param content: The content of the chunk
    :return: dict to be json data
    """
    hash_content = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return {
        "faiss_index": faiss_index,
        "chunk_index": chunk_index,
        "hash": hash_content,
        "content": content,
    }


def save_metadata(chunk_metadata):
    """
    Save a single chunk metadata to metadata.json.
    Appends to existing list or creates a new one.
    """
    # Check if file exists and load existing data
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            json_data = json.load(f)
            if not isinstance(json_data, list):
                json_data = []
    else:
        json_data = []

    # Append new data
    json_data.append(chunk_metadata)

    # Write back to file
    with open(metadata_path, "w") as f:
        json.dump(json_data, f, indent=4)


def load_metadata():
    """
    Load existing metadata.json.
    Returns list of metadata entries or empty list if file doesn't exist.
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    return []


def chunk_loader(chunk_path):
    """
    Load chunk JSON file and return list of chunks.
    """
    with chunk_path.open("r", encoding="utf-8") as f:
        json_chunks = json.load(f)
    return json_chunks


def embed_chunk(chunk_content):
    """
    Generate a single embedding for a chunk and reshape for FAISS.
    Returns (1, dim) array and dim.
    """
    embedding_model = "embeddinggemma"
    resp = ollama.embed(model=embedding_model, input=chunk_content)
    embedding = resp["embeddings"][0]

    # reshape to 2D for FAISS
    embedding_vectors = np.array(embedding, dtype="float32").reshape(1, -1)
    dim = embedding_vectors.shape[1]

    return embedding_vectors, dim


def faiss_index_chunk(vectors, dim):
    """
    Create a FAISS index from a single chunk embedding.
    Returns the FAISS IndexFlatL2 object.
    """
    index = faiss.IndexFlatL2(dim)
    # noinspection PyArgumentList
    index.add(vectors)
    return index


def faiss_write(index):
    """
    Write a FAISS index to disk.
    """
    faiss.write_index(index, faiss_path)


def load_embeddings():
    """
    Load existing embeddings cache or return empty array.
    """
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path)
    return np.empty((0, 0), dtype="float32")


def save_embeddings(all_embeddings):
    """
    Save the embeddings cache to .npy.
    """
    np.save(embeddings_path, all_embeddings)