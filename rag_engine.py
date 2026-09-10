from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

loader = TextLoader("Documents/factory_sop.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts)

question = input("Ask FactoryOS: ")

question_embedding = model.encode([question])

scores = np.dot(embeddings, question_embedding.T).flatten()

best_match = texts[np.argmax(scores)]

prompt = f"""
You are an AI Manufacturing Operations Copilot.

Use the context below to answer the question.

Context:
{best_match}

Question:
{question}
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nFactoryOS Answer:\n")
print(response.choices[0].message.content)