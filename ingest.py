import os

import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Pages: {len(docs)}")
print(f"Chunks: {len(chunks)}")

#load for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings loaded!")

vector_db = FAISS.from_documents(chunks, embeddings)
print("FAISS database created!")

vector_db.save_local("faiss_index")
print("FAISS database saved locally!")
