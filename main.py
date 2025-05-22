import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Utility Functions ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    return CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )

# === Streamlit UI ===
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title(":blue[PDF Chatbot] 📢")

# Sidebar upload
st.sidebar.title("Upload PDF")
pdf_docs = st.sidebar.file_uploader("Choose your PDF file(s)", accept_multiple_files=True)
if st.sidebar.button("Process PDF") and pdf_docs:
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.chat_history = []
        st.success("PDF processed. Start chatting!")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Interface
if st.session_state.conversation:
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        response = st.session_state.conversation({"question": user_input})
        st.session_state.chat_history.append((user_input, response['answer']))

    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
else:
    st.markdown(
        """
        #### 🔄 Upload a PDF from the sidebar to start chatting.
        """
    )
