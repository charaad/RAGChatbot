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
temperature =0.55,
top_p=0.75,
top_k=60
)


def rag_run(question: str, history: list[dict] = None) -> str:
    """Retrieves context and generates an answer using Gemini."""
 
    if history is None:
        history = []

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
   # sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    history_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history[-8:]
    )

    # Format Prompt 
    prompt = f"""You are a helpful assistant. Use the provided context and conversation history to answer the user's question.

Guidelines:
1. Primary Source: Ground your answer in the provided context first. Then also look at the conversation history and prioritise the more recent past messeges over the older ones.
2. Fallback/Inference: If the exact answer isn't explicitly stated, use general domain knowledge or logical inference to provide a useful answer, but make sure 
   it is related to any key words the current or recent past messeges use. Do not provide answers that are unrelated to the question.
3. Fallback Refusal: Only state "I don't have enough context to answer that accurately" if the query is completely outside the domain of the provided materials.
   Do not answer questions completely unrelated to the provided context, instead say "I don't have enough context to answer that accurately".
4. Tone: Speak directly and naturally. Never mention things similar to "documents", "context", "provided text", or "conversation history" in your response.
5. Multi-Turn Coreference: When the user query uses pronouns or collective references (e.g., "both", "these", "they", "the former", "it"), resolve them using 
   ALL relevant terms mentioned across the conversation history before formulating your answer.

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

