import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_postgres import PGVector

from dotenv import load_dotenv



#load api
env_path = Path(".") / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    raise ValueError("SUPABASE_DB_URL is not configured in environment variables.")


# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#load faiss for search
#vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#retriever = vector_db.as_retriever(search_kwargs={"k": 8})

vector_db = PGVector(
    embeddings=embeddings,
    collection_name="rag_documents",
    connection=SUPABASE_DB_URL,
    use_jsonb=True,
)
retriever = vector_db.as_retriever(search_kwargs={"k": 8})


#load model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",
temperature =0.45,
top_p=0.8,
top_k=70
)


def rag_run(question: str, history: list[dict] = None) -> str:
    """Retrieves context and generates an answer using Gemini."""
 
    if history is None:
        history = []

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
   # sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Format historical convo data (FIXED TYPO: message instead of messege)
    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history[-6:]
    )

    # Format Prompt 
    prompt = f"""Answer the question using the provided context and ALWAYS read and account for the conversation history.
       If no keywords are provided in current context, answer the question based previous context and conversation history. 

If the answer isn't clearly mentioned, look for adjacent information and attempt to answer. If no adjacent information is available, say:
"This is not in my field of expertise."

Do not mention the documents in the response, avoid saying "According to the documents" or similar phrases. 

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

    return answer

