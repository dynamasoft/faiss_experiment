import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB Client (persistent or in-memory)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)  # Change path for persistence

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="smart_contracts")

# Load a text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample smart contracts (ERC-20 and ERC-1155)
erc20_contract = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
contract MyToken is ERC20 {
    constructor() ERC20("MyToken", "MTK") {
        _mint(msg.sender, 1000000 * 10 ** decimals());
    }
}
"""

erc1155_contract = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
contract MyNFT is ERC1155 {
    constructor() ERC1155("https://example.com/api/item/{id}.json") {}
}
"""

# Generate embeddings
erc20_vector = model.encode(erc20_contract).tolist()
erc1155_vector = model.encode(erc1155_contract).tolist()

# Add contracts to ChromaDB
collection.add(
    ids=["erc20", "erc1155"],  # Unique IDs
    embeddings=[erc20_vector, erc1155_vector],  # Vectors
    metadatas=[{"type": "ERC-20"}, {"type": "ERC-1155"}],  # Metadata for filtering
)

# Query: Modified ERC-1155 contract (variant)
erc1155_variant = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
contract CustomNFT is ERC1155 {
    constructor() ERC1155("https://custom-nft.com/api/item/{id}.json") {}
}
"""

# Generate query embedding
query_vector = model.encode(erc1155_variant).tolist()

# Search for the closest match
results = collection.query(
    query_embeddings=[query_vector], n_results=1  # Retrieve the most similar contract
)

# Extract and print the closest match
nearest_contract_type = results["metadatas"][0][0]["type"]
nearest_contract_id = results["ids"][0][0]
nearest_distance = results["distances"][0][0]

print("Query Vector:", query_vector)
print("Nearest Neighbor ID:", nearest_contract_id)
print("Predicted Contract Type:", nearest_contract_type)
print("Distance:", nearest_distance)
