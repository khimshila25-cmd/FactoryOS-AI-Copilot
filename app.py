import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import os
import glob

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

    docs = []

    pdf_files = glob.glob("Documents/*.pdf")

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

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

    top_indices = np.argsort(scores)[-5:]

    context = "\n\n".join(
        [texts[i] for i in top_indices]
    )

    prompt = f"""
    You are an AI Manufacturing Operations Copilot.

    Use the context below to answer the question.

    Context:
    {context}

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