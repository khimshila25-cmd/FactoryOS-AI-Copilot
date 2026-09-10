import streamlit as st
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

st.set_page_config(page_title="FactoryOS AI Copilot")

st.title("🏭 FactoryOS AI Operations Copilot")

question = st.text_input(
    "Ask a manufacturing question:"
)

if question:

    loader = TextLoader("Documents/factory_sop.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(docs)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    question_embedding = model.encode(
        [question]
    )

    scores = np.dot(
        embeddings,
        question_embedding.T
    ).flatten()

    best_match = texts[np.argmax(scores)]

    prompt = f"""
    You are an AI Manufacturing Operations Copilot.

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

    st.subheader("FactoryOS Answer")

    st.write(
        response.choices[0].message.content
    )