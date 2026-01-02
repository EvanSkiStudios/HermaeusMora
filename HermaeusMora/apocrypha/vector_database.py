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


def build_file_paths(base_dir: Path | None = None) -> tuple[Path, Path, Path]:
    base_dir = base_dir or Path.cwd()
    faiss_path = base_dir / "faiss.bin"
    embeddings_path = base_dir / "embeddings.npy"
    metadata_path = base_dir / "metadata.json"
    return faiss_path, embeddings_path, metadata_path


def json_builder(faiss_index: int, chunk_index: int, content: str) -> dict:
    """
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


def chunk_loader(chunk_path):
    with chunk_path.open("r", encoding="utf-8") as f:
        json_chunks = json.load(f)
    return json_chunks


def embed_chunk(chunk_content):
    embedding_model = "embeddinggemma"
    resp = ollama.embed(model=embedding_model, input=chunk_content)
    embedding = resp["embeddings"][0]

    # reshape to 2D for FAISS
    embedding_vectors = np.array(embedding, dtype="float32").reshape(1, -1)
    dim = embedding_vectors.shape[1]

    return embedding_vectors, dim


def faiss_index(vectors, dim):
    index = faiss.IndexFlatL2(dim)
    # noinspection PyArgumentList
    index.add(vectors)
    return index
