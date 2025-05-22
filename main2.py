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

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_txt(txt_files):
    text = ""
    for txt in txt_files:
        string = txt.read()
        if isinstance(string, bytes):
            string = string.decode("utf-8", errors="ignore")
        text += string + "\n"
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def main():
    st.set_page_config(page_title="📄 Chat with Files", page_icon="💬", layout="wide")
    st.title("💬 AI File Chat Assistant")

    # Sidebar File Upload
    st.sidebar.header("📁 Upload Your Files")
    pdf_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    txt_files = st.sidebar.file_uploader("Upload TXT files", accept_multiple_files=True, type=["txt"])

    if st.sidebar.button("📥 Process Files"):
        with st.spinner("Processing files..."):
            raw_text = ""
            if pdf_files:
                raw_text += extract_text_from_pdfs(pdf_files)
            if txt_files:
                raw_text += extract_text_from_txt(txt_files)

            chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.chat_history = []
            st.success("✅ Files processed! You can start chatting below.")

    # Chat Interface
    st.divider()
    st.subheader("💬 Ask anything from your uploaded files")

    if "conversation" not in st.session_state:
        st.info("Please upload files from the sidebar to begin.")
        return

    user_query = st.chat_input("Type your question here...")
    if user_query:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({"question": user_query})
            st.session_state.chat_history.append((user_query, response['answer']))

    for user_msg, bot_msg in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

if __name__ == "__main__":
    main()
