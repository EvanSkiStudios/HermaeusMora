from vector_database import *


def EpistolaryAcumen(path):
    """
    :param path: The chunk file to process
    :return:
    """
    faiss_path, embeddings_path, metadata_path = build_file_paths()

    chunks = chunk_loader(path)
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        vectors, dim = embed_chunk(content)
        # faiss index chunk
        # save metadata
        # write chunk to faiss



