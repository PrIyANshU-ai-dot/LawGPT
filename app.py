# app.py — LawGPT RAG Chatbot + Document Summarizer (Ollama + FAISS)
# -------------------------------------------------------------------
# A local generative AI legal chatbot and document summarizer using the Indian Penal Code data.

import os
import time
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------------------------------
# 🎨 Streamlit Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="LawGPT", page_icon="⚖️", layout="wide")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
        color: black;
        font-weight: 600;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    footer, #MainMenu, .stDeployButton, #stDecoration {
        visibility: hidden;
    }
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# 🧠 Initialize Session Memory & Chat State
# ---------------------------------------------------
def reset_conversation():
    st.session_state.messages = []
    if "memory" in st.session_state:
        st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True
    )

# ---------------------------------------------------
# 🔎 Load Embeddings and FAISS Database
# ---------------------------------------------------
st.sidebar.subheader("📚 IPC Database Status")

try:
    embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    st.sidebar.success("✅ FAISS DB Loaded Successfully")
except Exception as e:
    st.sidebar.error("❌ Failed to load FAISS DB. Run `python ingest.py` first.")
    st.sidebar.write(str(e))
    st.stop()

# ---------------------------------------------------
# 🤖 Prompt Template
# ---------------------------------------------------
prompt_template = """<s>[INST]You are a legal assistant specializing in the Indian Penal Code.
Use the provided context to answer the user's question precisely and concisely.
If you are unsure, say so. Do not fabricate answers.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ---------------------------------------------------
# 🧠 LLM and RAG Chain Setup
# ---------------------------------------------------
llm = Ollama(model="llama3")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ---------------------------------------------------
# 💬 Streamlit Chat UI
# ---------------------------------------------------
st.title("⚖️ LawGPT — Your Legal AI Assistant")
st.caption("Ask questions related to the Indian Penal Code or upload documents to summarize them.")

# File uploader on main screen
uploaded_files = st.file_uploader(
    "📄 Upload PDF or TXT files to summarize (optional)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("🧠 Summarizing your uploaded documents... please wait.")
    summaries = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text = " ".join([p.page_content for p in pages])
            os.remove(tmp_path)
        else:
            text = uploaded_file.read().decode("utf-8")

        summarize_prompt = f"""
        You are a helpful AI assistant that summarizes documents into simple, plain English.
        Summarize the key points of this document clearly and concisely.
        Document:
        {text[:6000]}
        """

        try:
            summary = llm.invoke(summarize_prompt)
            summaries.append(f"**📘 {uploaded_file.name} Summary:**\n\n{summary.strip()}")
        except Exception as e:
            summaries.append(f"⚠️ Error summarizing {uploaded_file.name}: {e}")

    st.success("✅ Summaries generated successfully!")
    for summary in summaries:
        st.markdown(summary)
    st.divider()

# Show previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input area
user_query = st.chat_input("Ask your legal question here...")

if user_query:
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            try:
                result = qa_chain.invoke({"question": user_query})
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Error generating answer: {e}"

            message_placeholder = st.empty()
            full_response = "⚠️ **Note:** This is not legal advice. \n\n"
            for chunk in answer:
                full_response += chunk
                time.sleep(0.015)
                message_placeholder.markdown(full_response + " ▌")

            message_placeholder.markdown(full_response)

        st.button("🔄 Reset Conversation", on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
