import streamlit as st
import requests


st.set_page_config( page_title="RAG Chatbot", page_icon="🤖", layout="centered" )

st.title("🤖 RAG Chatbot") 
st.caption("Ask questions about the Economics, Astronomy, and AI documents.")

if "messages" not in st.session_state: st.session_state.messages = []

for message in st.session_state.messages: 
    with st.chat_message(message["role"]): st.markdown(message["content"])


if prompt := st.chat_input("Ask a question..."): 
   with st.chat_message("user"): 
       st.markdown(prompt)

   history = st.session_state.messages[-6:]
    
   try:
#keep last 6 messeges history in mind for context window

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"question": prompt,
                  "history": st.session_state.messages[-6:]}
        )

        response.raise_for_status()

        data = response.json()

        answer = data["answer"]
        sources = data["sources"]


        with st.chat_message("assistant"):

            st.markdown(answer)

            if sources:

                with st.expander("📚 Sources"):

                    for source in sources:
                        st.write(f"- {source}")
    
    
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

   except requests.exceptions.RequestException as e:

        st.error(
            f"Could not connect to the FastAPI server: {e}"
        )