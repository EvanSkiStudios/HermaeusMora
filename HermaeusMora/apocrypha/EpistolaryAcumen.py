from vector_database import *


def RetainKnowledge(path):
    """
    Process a chunk file and retain its knowledge.
    :param path: The chunk file to process
    """

    chunks = chunk_loader(path)
    all_embeddings = load_embeddings()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        # Generate embeddings for chunk
        vectors, dim = embed_chunk(content)

        # If first time, all_embeddings is empty → reshape to (0, dim), then Append to embeddings cache
        if all_embeddings.shape[0] == 0:
            all_embeddings = np.empty((0, dim), dtype="float32")
        all_embeddings = np.vstack([all_embeddings, vectors])

        # Create FAISS index for this chunk (currently separate index per chunk)
        faiss_index = faiss_index_chunk(vectors, dim)
        faiss_write(faiss_index)

        # Save metadata
        chunk_metadata = json_builder(len(all_embeddings) - 1, chunk_id, content)
        save_metadata(chunk_metadata)

    # Save embeddings cache
    save_embeddings(all_embeddings)

