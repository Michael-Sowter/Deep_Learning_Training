from sentence_transformers import SentenceTransformer
import torch

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define two example sentences
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A quick brown dog jumps over the lazy fox"

# Encode sentences and compute similarity score
embeddings1 = model.encode([sentence1], convert_to_tensor=True)
embeddings2 = model.encode([sentence2], convert_to_tensor=True)
cosine_similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

# Print similarity score
print(f"Similarity score: {cosine_similarities.item()}")


