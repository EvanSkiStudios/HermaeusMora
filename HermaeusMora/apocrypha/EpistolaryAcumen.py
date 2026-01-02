import logging
import numpy as np

from utility_scripts.system_logging import setup_logger
from apocrypha.vector_database import chunk_loader, load_embeddings, load_metadata, embed_chunk, \
    load_or_create_faiss_index, append_to_faiss, json_builder, save_metadata, save_embeddings, save_faiss

# configure logging
logger = setup_logger(__name__)

# Suppress Ollama HTTP request logs
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def RetainKnowledge(path):
    """
    Process a chunk file and retain its knowledge.
    :param path: The chunk file to process
    """
    logger.info(f"Retaining Knowledge > {path}")

    chunks = chunk_loader(path)
    all_embeddings = load_embeddings()
    metadata = load_metadata()

    # Determine dimension from first chunk if embeddings are empty
    first_dim = None
    if all_embeddings.shape[0] > 0:
        first_dim = all_embeddings.shape[1]

    index = None

    logger.info("Processing Chunks...")
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        # Embed
        vectors, dim = embed_chunk(content)

        # Initialize FAISS if first time
        if index is None:
            if all_embeddings.shape[0] == 0:
                first_dim = dim
            else:
                first_dim = all_embeddings.shape[1]
            index = load_or_create_faiss_index(first_dim)

        # Check dimension consistency
        if dim != first_dim:
            raise ValueError(f"Embedding dimension mismatch: {dim} != {first_dim}")

        # Append vectors to embeddings cache
        if all_embeddings.shape[0] == 0:
            all_embeddings = vectors
        else:
            all_embeddings = np.vstack([all_embeddings, vectors])

        # Append vectors to FAISS index
        append_to_faiss(index, vectors)

        # Create metadata entry
        faiss_idx = all_embeddings.shape[0] - 1
        chunk_metadata = json_builder(faiss_idx, chunk_id, content)
        save_metadata(chunk_metadata)

    logger.info("Finished Chunks, Saving...")
    # Save everything
    save_embeddings(all_embeddings)
    save_faiss(index)

    logger.info(f"Finished > {path}")


