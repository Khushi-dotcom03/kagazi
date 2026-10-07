import faiss
import numpy as np

def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))
    return index


def search(index, query_embedding, top_k=3):
    distances, indices = index.search(
        np.array(query_embedding, dtype="float32"),
        top_k
    )
    return distances[0], indices[0]

if __name__ == "__main__":
    import numpy as np

    embeddings = np.random.rand(5, 384).astype("float32")

    index = create_index(embeddings)

    print("Vectors stored:", index.ntotal)

    query = np.random.rand(1, 384).astype("float32")

    distances, indices = search(index, query)

    print("Indices:", indices)
    print("Distances:", distances)