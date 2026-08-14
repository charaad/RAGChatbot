import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS
from langchain_postgres import PGVector



load_dotenv()

connection = os.getenv("SUPABASE_DB_URL")
if not connection:
    raise ValueError("SUPABASE_DB_URL is not set")

loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Pages: {len(docs)}")
print(f"Chunks: {len(chunks)}")

#load for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings loaded!")


vector_db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_documents",
    connection=connection,
    use_jsonb=True,
)

print("Documents embedded and stored in Supabase!")

#vector_db = FAISS.from_documents(chunks, embeddings)
#print("FAISS database created!")

#vector_db.save_local("faiss_index")
#print("FAISS database saved locally!")
