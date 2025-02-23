import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model (use OpenAI's API or a local model like SBERT)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient text embedding model

# Sample smart contracts (simplified)
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

# Generate embeddings for contracts
erc20_vector = model.encode(erc20_contract).astype("float32")
erc1155_vector = model.encode(erc1155_contract).astype("float32")

# Create FAISS index
dimension = erc20_vector.shape[0]
index = faiss.IndexFlatL2(dimension)
contract_labels = ["ERC-20", "ERC-1155"]  # Store contract type for interpretation

# Add vectors to FAISS index
index.add(np.array([erc20_vector, erc1155_vector]))

# Query: Modified ERC-1155 contract (variant)
erc1155_variant = """
pragma solidity ^0.8.0;
import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
contract CustomNFT is ERC1155 {
    constructor() ERC1155("https://custom-nft.com/api/item/{id}.json") {}
}
"""

# Generate query vector
query_vector = model.encode(erc1155_variant).astype("float32")

# Search for nearest contract
k = 1  # Find the closest match
distances, indices = index.search(query_vector.reshape(1, -1), k)

# Retrieve the closest contract type
nearest_contract = contract_labels[indices[0][0]]

# Print results
print("Query Vector:", query_vector)
print("Nearest Neighbor Index:", indices[0][0])
print("Predicted Contract Type:", nearest_contract)
print("Distance:", distances[0][0])
