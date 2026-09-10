from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load document
loader = TextLoader("Documents/factory_sop.txt")
docs = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = model.encode(chunk_texts)

# User query
query = "What orders should be escalated?"

query_embedding = model.encode([query])

# Similarity search
scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)

best_match = scores.argmax()

print("Question:", query)
print("\nBest Match:\n")
print(chunk_texts[best_match])