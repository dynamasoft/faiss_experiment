import faiss
import numpy as np

# Define the dimension of the vectors
dimension = 2  # Example dimension size
num_vectors = 10  # Number of vectors in the database

# Generate random vectors (simulating embeddings)
np.random.seed(42)
database_vectors = np.random.random((num_vectors, dimension)).astype("float32")

print("Vectors :", database_vectors)


# Create a FAISS index (L2 distance-based index)
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance

# Add vectors to the index
index.add(database_vectors)

# Generate a random query vector
query_vector = np.random.random((1, dimension)).astype("float32")
print("query vector:", query_vector)

# Search for the 5 nearest neighbors
k = 5
distances, indices = index.search(query_vector, k)

# Print results
print("Nearest Neighbors' Indices:", indices)
print("Distances:", distances)
