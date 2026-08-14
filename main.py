import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path

from dotenv import load_dotenv



#load api
load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#load faiss for search
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 8})

#load model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def rag_run(question: str, history: list[dict] = None) -> tuple[str, list[str]]:
    """Retrieves context and generates an answer using Gemini."""
 
    if history is None:
        history = []

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Format historical convo data (FIXED TYPO: message instead of messege)
    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history[-6:]
    )

    # Format Prompt (FIXED: Included history_text so the model actually sees conversation history)
    prompt = f"""Answer the question using ONLY the provided context and conversation history.

If the answer isn't directly mentioned, say:
"I don't know based on the provided documents."

Conversation History:
{history_text if history_text else "No prior history."}

Context:
{context}

Question:
{question}

Answer:"""

    # Call LLM
    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if item.get("type") == "text"
        )

    return answer, sources

