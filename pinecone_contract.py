import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Ensure API key is set
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")  # Use latest available region

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define index name and embedding model
INDEX_NAME = "smart-contracts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Ensure it matches the model's output

# Check if the index exists, create it if not
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=EMBEDDING_DIMENSION, metric="cosine")

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Sample smart contracts
smart_contracts = {
    "erc20": {
        "code": """
        pragma solidity ^0.8.0;
        import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
        contract MyToken is ERC20 {
            constructor() ERC20("MyToken", "MTK") {
                _mint(msg.sender, 1000000 * 10 ** decimals());
            }
        }
        """,
        "metadata": {"type": "ERC-20"},
    },
    "erc1155": {
        "code": """
        pragma solidity ^0.8.0;
        import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
        contract MyNFT is ERC1155 {
            constructor() ERC1155("https://example.com/api/item/{id}.json") {}
        }
        """,
        "metadata": {"type": "ERC-1155"},
    },
}

# Encode and upsert embeddings
vectors = [
    (key, model.encode(data["code"]).tolist(), data["metadata"])
    for key, data in smart_contracts.items()
]
index.upsert(vectors)

# Query: Modified ERC-1155 contract
erc1155_variant = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
contract CustomNFT is ERC1155 {
    constructor() ERC1155("https://custom-nft.com/api/item/{id}.json") {}
}
"""

# Generate and execute query
query_vector = model.encode(erc1155_variant).tolist()
query_result = index.query(vector=query_vector, top_k=1, include_metadata=True)

# Extract and display results
if query_result["matches"]:
    match = query_result["matches"][0]
    print(f"Nearest Neighbor ID: {match['id']}")
    print(f"Predicted Contract Type: {match['metadata']['type']}")
    print(f"Similarity Score: {match['score']:.4f}")
else:
    print("No close match found.")
